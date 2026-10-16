"""Object detection + localization node.

Pipeline: RGB image -> detections (color/shape or YOLO) -> ground-plane
projection through TF (object center at z = support_height + object_height / 2 in
base_footprint) -> size consistency gate -> tracking in the fixed frame.

Publishes
  /perception/observations  ObjectObservation (pose in base_footprint, every frame)
  /perception/target        PoseStamped (best target track, fixed frame)
  /perception/markers       MarkerArray (tracks, for RViz)
  /perception/debug_image   Image
  /perception/tables        PointCloud2 (fixed frame, the tables under known objects)
  TF  <fixed_frame> -> target_object
Subscribes
  /perception/reject_target std_msgs/Empty   mark the current target as not-the-target
  /perception/forget        geometry_msgs/PointStamped  drop tracks near a point (picked/delivered)
  /perception/picked        geometry_msgs/PointStamped  the target was picked here: keep its table

The tables are 7 cm tall, below the LiDAR plane, so Nav2 cannot see them. Each
known table is published as a patch of points for the costmaps to mark. The
target's own table is left out until it is picked, so the robot can drive up
to it; every other object's table (distractors, rejected tracks) is in from the
moment its track is confirmed.
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from fetch_deliver_interfaces.msg import ObjectObservation
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, Header
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .color_detector import ColorShapeDetector, draw, label_matches
from .geometry import center_from_height, ground_contact_center
from .ros_utils import apply, lookup_matrix
from .tracker import ObjectTracker


class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__('object_detector')
        self.declare_parameter('target_label', 'red_cylinder')
        self.declare_parameter('fixed_frame', 'map')
        self.declare_parameter('fallback_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('object_height', 0.10)
        self.declare_parameter('support_height', 0.07)  # objects stand on a 7 cm table
        self.declare_parameter('object_size', 0.03)
        self.declare_parameter('max_range', 3.5)
        self.declare_parameter('backend', 'color')       # color | yolo
        self.declare_parameter('yolo_weights', '')
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('table_half_size', 0.10)   # 0.16 m table + 2 cm margin each side

        self.target = self.get_parameter('target_label').value
        self.base = self.get_parameter('base_frame').value
        self.h = self.get_parameter('object_height').value
        self.support = self.get_parameter('support_height').value
        self.size = self.get_parameter('object_size').value
        self.max_range = self.get_parameter('max_range').value
        backend = self.get_parameter('backend').value
        if backend == 'yolo':
            from .yolo_detector import YoloDetector
            self.detector = YoloDetector(self.get_parameter('yolo_weights').value,
                                         self.get_parameter('yolo_confidence').value)
        else:
            self.detector = ColorShapeDetector()
        self.get_logger().info(f'object detector: backend={backend}, target={self.target}')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tracker = ObjectTracker()
        self.K = None
        self.camera_frame = None

        self.obs_pub = self.create_publisher(ObjectObservation, '/perception/observations', 20)
        self.target_pub = self.create_publisher(PoseStamped, '/perception/target', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/markers', 5)
        self.debug_pub = self.create_publisher(Image, '/perception/debug_image', 2)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 5)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.create_subscription(Empty, '/perception/reject_target', self.on_reject, 5)
        self.create_subscription(PointStamped, '/perception/forget', self.on_forget, 5)
        self.create_subscription(PointStamped, '/perception/picked', self.on_picked, 5)
        self.current_target_id = None
        self.picked_tables = []   # (x, y) in the fixed frame
        self.tables_pub = self.create_publisher(PointCloud2, '/perception/tables', 2)
        self.create_timer(0.5, self.publish_tables)

    # ------------------------------------------------------------------ io
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.camera_frame = msg.header.frame_id

    def fixed_frame(self) -> str:
        ff = self.get_parameter('fixed_frame').value
        if self.tf_buffer.can_transform(ff, self.base, Time()):
            return ff
        return self.get_parameter('fallback_frame').value

    def on_reject(self, _msg):
        if self.current_target_id is not None:
            self.tracker.reject(self.current_target_id)
            self.get_logger().info(f'track {self.current_target_id} rejected as target')
            self.current_target_id = None

    def on_forget(self, msg: PointStamped):
        T = lookup_matrix(self.tf_buffer, self.fixed_frame(), msg.header.frame_id or self.fixed_frame())
        if T is None:
            return
        p = apply(T, (msg.point.x, msg.point.y, msg.point.z))
        self.tracker.remove_near(p[0], p[1], 0.35)

    def on_picked(self, msg: PointStamped):
        T = lookup_matrix(self.tf_buffer, self.fixed_frame(), msg.header.frame_id or self.fixed_frame())
        if T is None:
            return
        p = apply(T, (msg.point.x, msg.point.y, msg.point.z))
        self.picked_tables.append((float(p[0]), float(p[1])))
        self.tracker.remove_near(p[0], p[1], 0.35)

    def publish_tables(self):
        centers = list(self.picked_tables)
        centers += [(t.x, t.y) for t in self.tracker.confirmed()
                    if t.rejected or not label_matches(t.label, self.target)]
        half = self.get_parameter('table_half_size').value
        grid = np.arange(-half, half + 1e-6, 0.025)
        pts = [(cx + dx, cy + dy, 0.04) for cx, cy in centers for dx in grid for dy in grid]
        header = Header(frame_id=self.fixed_frame(), stamp=self.get_clock().now().to_msg())
        self.tables_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))

    # ------------------------------------------------------------ pipeline
    def on_image(self, msg: Image):
        if self.K is None:
            return
        stamp = Time.from_msg(msg.header.stamp)
        T_base_cam = lookup_matrix(self.tf_buffer, self.base, self.camera_frame or msg.header.frame_id,
                                   stamp)
        fixed = self.fixed_frame()
        T_fixed_base = lookup_matrix(self.tf_buffer, fixed, self.base, stamp)
        if T_base_cam is None or T_fixed_base is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = self.detector.detect(bgr)
        T_cam_base = np.linalg.inv(T_base_cam)
        fx = self.K[0, 0]
        observations, kept = [], []
        for d in detections:
            x0, y0, w0, h0 = d.bbox
            center_z = self.support + self.h / 2
            if d.shape == 'cylinder' and y0 > 1 and y0 + h0 < msg.height - 1:
                # whole tall target in view: range from its apparent height
                p = center_from_height(x0 + w0 / 2.0, y0, y0 + h0, self.K, T_base_cam,
                                       self.h, self.size / 2, center_z)
            else:
                p = ground_contact_center(x0 + w0 / 2.0, y0 + h0 - 0.5, self.K, T_base_cam,
                                          self.size / 2, center_z, self.support)
            if p is None:
                continue
            rng = float(math.hypot(p[0], p[1]))
            if rng > self.max_range:
                continue
            depth = apply(T_cam_base, p)[2]
            expected_px = fx * self.size / max(depth, 1e-3)
            # width only: up close the top of a tall object is cut off by the image edge
            if not 0.45 * expected_px <= d.bbox[2] <= 2.6 * expected_px:
                continue  # wrong size for a graspable object at that distance (e.g. furniture)
            kept.append(d)
            obs = ObjectObservation()
            obs.header.stamp = msg.header.stamp
            obs.header.frame_id = self.base
            obs.label = d.label
            obs.confidence = float(d.confidence)
            obs.pose.position.x, obs.pose.position.y, obs.pose.position.z = map(float, p)
            obs.pose.orientation.w = 1.0
            obs.range = rng
            obs.bearing = float(math.atan2(p[1], p[0]))
            obs.bbox = [int(v) for v in d.bbox]
            obs.is_target = label_matches(d.label, self.target)
            self.obs_pub.publish(obs)
            pf = apply(T_fixed_base, p)
            observations.append((d.color, d.shape, float(pf[0]), float(pf[1]), float(pf[2]), rng))

        self.tracker.update(observations, stamp.nanoseconds * 1e-9)
        robot = T_fixed_base[:3, 3]
        best = self.tracker.best_target(self.target, (robot[0], robot[1]))
        if best is not None:
            self.current_target_id = best.track_id
            ps = PoseStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = fixed
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = best.x, best.y, best.z
            ps.pose.orientation.w = 1.0
            self.target_pub.publish(ps)
            t = TransformStamped()
            t.header = ps.header
            t.child_frame_id = 'target_object'
            t.transform.translation.x, t.transform.translation.y = best.x, best.y
            t.transform.translation.z = best.z
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)
        self.publish_markers(fixed, msg.header.stamp)
        if self.get_parameter('publish_debug_image').value and self.debug_pub.get_subscription_count():
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(draw(bgr, kept, self.target), 'bgr8'))

    def publish_markers(self, frame, stamp):
        arr = MarkerArray()
        colors = {'red': (1, 0, 0), 'green': (0, 0.8, 0), 'blue': (0, 0.2, 1), 'yellow': (1, 0.9, 0)}
        for t in self.tracker.confirmed():
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, stamp
            m.ns, m.id = 'tracks', t.track_id
            m.type = Marker.CUBE if t.shape != 'cylinder' else Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = t.x, t.y, t.z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.06
            r, g, b = colors.get(t.color, (1, 1, 1))
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            m.color.a = 0.4 if t.rejected else 0.9
            arr.markers.append(m)
            txt = Marker()
            txt.header = m.header
            txt.ns, txt.id, txt.type = 'labels', t.track_id, Marker.TEXT_VIEW_FACING
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = t.x, t.y, t.z + 0.15
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.08
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'{t.label} ({t.hits})'
            arr.markers.append(txt)
        self.marker_pub.publish(arr)


def main():
    # OpenCV's worker threads busy-wait between small per-frame jobs and burn a whole
    # core for nothing next to Gazebo; one thread is plenty at camera rate.
    cv2.setNumThreads(1)
    rclpy.init()
    node = ObjectDetectorNode()
    try:
        # Multi-threaded so TF keeps arriving while the image callback waits for the
        # transform at the image's capture time.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
