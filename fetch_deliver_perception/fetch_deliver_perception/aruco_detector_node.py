"""ArUco drop-off marker detection.

Detects DICT_4X4_50 markers, estimates their pose with solvePnP, and keeps a
filtered estimate of the drop-off marker in the fixed frame.

Publishes /perception/dropoff_marker (PoseStamped, fixed frame; the marker's
x axis points out of the wall, towards the robot side) and TF dropoff_marker.
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from .geometry import matrix_to_quat
from .ros_utils import lookup_matrix

# solvePnP marker frame (IPPE_SQUARE corner order): x right, y up, z out of the marker.
# Re-express it so x points out of the wall (z_cv), z up (y_cv) and y = z cross x (x_cv).
# Columns are the new axes written in the OpenCV marker frame.
R_CV_TO_MARKER = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)


def make_aruco_detector():
    """Return gray -> (corners, ids) for DICT_4X4_50, for both OpenCV APIs.

    OpenCV >= 4.7 has cv2.aruco.ArucoDetector; Ubuntu 22.04's python3-opencv
    (4.5.4) only has the legacy functions.
    """
    aruco = cv2.aruco
    if hasattr(aruco, 'ArucoDetector'):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
        return detect
    dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    def detect_legacy(gray):
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids
    return detect_legacy


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')
        self.declare_parameter('marker_id', 7)
        self.declare_parameter('marker_size', 0.20)
        self.declare_parameter('fixed_frame', 'map')
        self.declare_parameter('fallback_frame', 'odom')
        self.declare_parameter('max_distance', 4.0)
        self.declare_parameter('max_rate', 5.0)   # Hz; the marker is static, no need for every frame
        self.last_processed = None
        self.marker_id = self.get_parameter('marker_id').value
        self.size = self.get_parameter('marker_size').value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.detect_markers = make_aruco_detector()
        s = self.size / 2
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        self.K = None
        self.D = None
        self.estimate = None
        self.count = 0
        self.pub = self.create_publisher(PoseStamped, '/perception/dropoff_marker', 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 5)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.create_timer(0.2, self.republish)

    def on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
        self.camera_frame = msg.header.frame_id

    def fixed_frame(self):
        ff = self.get_parameter('fixed_frame').value
        return ff if self.tf_buffer.can_transform(ff, 'base_footprint', Time()) else \
            self.get_parameter('fallback_frame').value

    def on_image(self, msg):
        if self.K is None:
            return
        stamp = Time.from_msg(msg.header.stamp)
        if self.last_processed is not None and \
                (stamp - self.last_processed).nanoseconds < 1e9 / self.get_parameter('max_rate').value:
            return
        self.last_processed = stamp
        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)
        if ids is None:
            return
        for c, i in zip(corners, ids.flatten()):
            if int(i) != self.marker_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(self.obj_pts, c.reshape(4, 2).astype(np.float32),
                                          self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or float(np.linalg.norm(tvec)) > self.get_parameter('max_distance').value:
                continue
            R_cam_marker = cv2.Rodrigues(rvec)[0] @ R_CV_TO_MARKER
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3], T_cam_marker[:3, 3] = R_cam_marker, tvec.flatten()
            fixed = self.fixed_frame()
            T_fixed_cam = lookup_matrix(self.tf_buffer, fixed, self.camera_frame,
                                        Time.from_msg(msg.header.stamp))
            if T_fixed_cam is None:
                return
            T = T_fixed_cam @ T_cam_marker
            pos = T[:3, 3]
            yaw = math.atan2(T[1, 0], T[0, 0])  # marker x axis heading (out of the wall)
            self.update(fixed, pos, yaw, float(np.linalg.norm(tvec)))

    def update(self, frame, pos, yaw, dist):
        w = 1.0 / max(1.0, dist)            # closer observations count more
        if self.estimate is None or self.estimate[0] != frame:
            self.estimate = [frame, np.array(pos), yaw, w]
        else:
            _, p0, y0, w0 = self.estimate
            tot = w0 + w
            p = (p0 * w0 + pos * w) / tot
            dy = math.atan2(math.sin(yaw - y0), math.cos(yaw - y0))
            self.estimate = [frame, p, y0 + dy * w / tot, min(tot, 20.0)]
        self.count += 1
        if self.count == 1:
            self.get_logger().info(f'drop-off marker {self.marker_id} found at '
                                   f'({pos[0]:.2f}, {pos[1]:.2f}) in {frame}')

    def republish(self):
        if self.estimate is None:
            return
        frame, p, yaw, _ = self.estimate
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        q = matrix_to_quat(R)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        self.pub.publish(ps)
        t = TransformStamped()
        t.header = ps.header
        t.child_frame_id = 'dropoff_marker'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
        t.transform.rotation = ps.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main():
    # OpenCV's worker threads busy-wait between small per-frame jobs and burn a whole
    # core for nothing next to Gazebo; one thread is plenty at camera rate.
    cv2.setNumThreads(1)
    rclpy.init()
    node = ArucoDetectorNode()
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
