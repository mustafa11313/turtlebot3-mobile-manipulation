"""ROS-side handles shared by all mission behaviours (actions, topics, TF, costmap)."""

import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from fetch_deliver_interfaces.action import Align, Pick, Place
from fetch_deliver_interfaces.msg import MissionEvent, MissionStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import BackUp, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from .geometry import CostmapView, Pose2D, quat_to_yaw, yaw_to_quat


@dataclass
class MissionContext:
    """Mission state shared between behaviours (the tree's blackboard)."""
    target_label: str
    map_frame: str = 'map'
    home: Optional[Pose2D] = None
    target_map: Optional[Tuple[float, float, float]] = None   # tracked target (map)
    target_stamp: float = 0.0
    aligned_pose: Optional[PoseStamped] = None                 # final estimate before pick
    approach_excludes: List[Pose2D] = field(default_factory=list)
    marker_map: Optional[Pose2D] = None                       # drop-off marker pose (map)
    dropoff_place: Optional[Tuple[float, float]] = None        # where to put the object (map)
    holding: bool = False
    phase: str = 'INIT'
    start_time: float = 0.0
    pick_attempts: int = 0
    nav_failures: int = 0
    recoveries: int = 0
    first_detection_time: Optional[float] = None
    failure_reason: str = ''


class RobotIO:

    def __init__(self, node, ctx: MissionContext, cb_group):
        self.node = node
        self.ctx = ctx
        self.log = node.get_logger()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.nav = ActionClient(node, NavigateToPose, '/navigate_to_pose', callback_group=cb_group)
        self.spin = ActionClient(node, Spin, '/spin', callback_group=cb_group)
        self.backup = ActionClient(node, BackUp, '/backup', callback_group=cb_group)
        self.align = ActionClient(node, Align, '/mission/align', callback_group=cb_group)
        self.pick = ActionClient(node, Pick, '/manipulation/pick', callback_group=cb_group)
        self.place = ActionClient(node, Place, '/manipulation/place', callback_group=cb_group)
        self.clear_global = node.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap', callback_group=cb_group)
        self.clear_local = node.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap', callback_group=cb_group)
        self.event_pub = node.create_publisher(MissionEvent, '/fetch_deliver/events', 20)
        self.status_pub = node.create_publisher(MissionStatus, '/fetch_deliver/status', 10)
        self.reject_pub = node.create_publisher(Empty, '/perception/reject_target', 5)
        self.forget_pub = node.create_publisher(PointStamped, '/perception/forget', 5)
        node.create_subscription(PoseStamped, '/perception/target', self._on_target, 10,
                                 callback_group=cb_group)
        node.create_subscription(PoseStamped, '/perception/dropoff_marker', self._on_marker, 10,
                                 callback_group=cb_group)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._on_costmap, latched,
                                 callback_group=cb_group)
        node.create_subscription(OccupancyGrid, '/map', self._on_map, latched, callback_group=cb_group)
        self.costmap: Optional[CostmapView] = None
        self.map_msg: Optional[OccupancyGrid] = None

    # ------------------------------------------------------------- inputs
    def _transform_xy(self, frame, x, y, z=0.0):
        if frame == self.ctx.map_frame:
            return (x, y, z)
        tf = self.tf_buffer.lookup_transform(self.ctx.map_frame, frame, Time(),
                                             timeout=Duration(seconds=0.1))
        q = tf.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        t = tf.transform.translation
        return (t.x + math.cos(yaw) * x - math.sin(yaw) * y,
                t.y + math.sin(yaw) * x + math.cos(yaw) * y, t.z + z)

    def _on_target(self, msg: PoseStamped):
        try:
            p = self._transform_xy(msg.header.frame_id, msg.pose.position.x, msg.pose.position.y,
                                   msg.pose.position.z)
        except Exception:  # noqa: BLE001
            return
        self.ctx.target_map = p
        self.ctx.target_stamp = time.monotonic()
        if self.ctx.first_detection_time is None:
            self.ctx.first_detection_time = time.monotonic()
            self.event('target_detected', json.dumps({'x': p[0], 'y': p[1]}))

    def _on_marker(self, msg: PoseStamped):
        try:
            x, y, _ = self._transform_xy(msg.header.frame_id, msg.pose.position.x, msg.pose.position.y)
        except Exception:  # noqa: BLE001
            return
        o = msg.pose.orientation
        yaw = quat_to_yaw(o.x, o.y, o.z, o.w)
        if msg.header.frame_id != self.ctx.map_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.ctx.map_frame, msg.header.frame_id, Time())
                q = tf.transform.rotation
                yaw += quat_to_yaw(q.x, q.y, q.z, q.w)
            except Exception:  # noqa: BLE001
                return
        self.ctx.marker_map = (x, y, yaw)

    def _on_costmap(self, msg: OccupancyGrid):
        data = np.array(msg.data, np.int16).reshape(msg.info.height, msg.info.width)
        self.costmap = CostmapView(data, msg.info.resolution,
                                   (msg.info.origin.position.x, msg.info.origin.position.y))

    def _on_map(self, msg: OccupancyGrid):
        self.map_msg = msg

    def robot_pose(self) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(self.ctx.map_frame, 'base_footprint', Time(),
                                                 timeout=Duration(seconds=0.05))
        except Exception:  # noqa: BLE001
            return None
        q = tf.transform.rotation
        return (tf.transform.translation.x, tf.transform.translation.y,
                quat_to_yaw(q.x, q.y, q.z, q.w))

    # ------------------------------------------------------------ outputs
    def pose_stamped(self, pose: Pose2D) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.ctx.map_frame
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(pose[0]), float(pose[1])
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        return ps

    def event(self, name: str, detail: str = '', value: float = 0.0):
        msg = MissionEvent(event=name, phase=self.ctx.phase, detail=detail, value=float(value))
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.event_pub.publish(msg)
        self.log.info(f'[event] {name} {detail}')

    def publish_status(self, active: str):
        s = MissionStatus()
        s.header.stamp = self.node.get_clock().now().to_msg()
        s.phase = self.ctx.phase
        s.active_behavior = active
        s.elapsed = time.monotonic() - self.ctx.start_time if self.ctx.start_time else 0.0
        s.pick_attempts = self.ctx.pick_attempts
        s.nav_failures = self.ctx.nav_failures
        s.recoveries = self.ctx.recoveries
        s.object_attached = self.ctx.holding
        self.status_pub.publish(s)

    def forget_near(self, x: float, y: float):
        p = PointStamped()
        p.header.frame_id = self.ctx.map_frame
        p.point.x, p.point.y = float(x), float(y)
        self.forget_pub.publish(p)

    def clear_costmaps(self):
        for cli in (self.clear_global, self.clear_local):
            if cli.service_is_ready():
                cli.call_async(ClearEntireCostmap.Request())
