"""Autonomous mapping: drive to frontiers with Nav2 until none are left, then save the map.

ros2 run fetch_deliver_mission frontier_explorer --ros-args -p map_path:=/path/to/maps/apartment
"""

import math
import subprocess
import time

import numpy as np
import rclpy
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .frontiers import best_frontier, find_frontiers
from .geometry import quat_to_yaw, yaw_to_quat


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')
        self.declare_parameter('map_path', '')
        self.declare_parameter('min_frontier_cells', 10)
        self.declare_parameter('goal_timeout', 150.0)
        self.declare_parameter('max_duration', 1200.0)
        self.map = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.blacklist = []

    def on_map(self, msg):
        self.map = msg

    def robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
            return tf.transform.translation.x, tf.transform.translation.y, quat_to_yaw(
                tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z,
                tf.transform.rotation.w)
        except Exception:  # noqa: BLE001
            return None

    def spin_wait(self, fut, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return fut.done()

    def run(self):
        self.get_logger().info('waiting for Nav2 and the map...')
        self.nav.wait_for_server()
        while rclpy.ok() and (self.map is None or self.robot_xy() is None):
            rclpy.spin_once(self, timeout_sec=0.2)
        t0 = time.monotonic()
        empty_rounds = 0
        while rclpy.ok() and time.monotonic() - t0 < self.get_parameter('max_duration').value:
            rclpy.spin_once(self, timeout_sec=0.1)
            m = self.map
            grid = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
            robot = self.robot_xy()
            fr = find_frontiers(grid, m.info.resolution,
                                (m.info.origin.position.x, m.info.origin.position.y), robot[:2],
                                min_cells=self.get_parameter('min_frontier_cells').value,
                                blacklist=self.blacklist)
            f = best_frontier(fr)
            if f is None:
                empty_rounds += 1
                if empty_rounds >= 3:
                    break
                time.sleep(1.0)
                continue
            empty_rounds = 0
            yaw = math.atan2(f.centroid[1] - f.goal[1], f.centroid[0] - f.goal[0])
            # never send the same goal twice: reached-but-still-a-frontier goals are blacklisted
            self.blacklist.append(f.target)
            self.get_logger().info(f'frontier: {len(fr)} left, going to ({f.goal[0]:.2f}, {f.goal[1]:.2f}) '
                                   f'[{f.cells} cells, {f.distance:.1f} m]')
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.pose.position.x, goal.pose.pose.position.y = f.goal
            _, _, qz, qw = yaw_to_quat(yaw)
            goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = qz, qw
            gf = self.nav.send_goal_async(goal)
            if not self.spin_wait(gf, 10.0) or not gf.result().accepted:
                continue
            rf = gf.result().get_result_async()
            if not self.spin_wait(rf, self.get_parameter('goal_timeout').value):
                gf.result().cancel_goal_async()
                continue
        self.get_logger().info(f'exploration finished after {time.monotonic() - t0:.0f} s')
        path = self.get_parameter('map_path').value
        if path:
            subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path,
                            '--ros-args', '-p', 'use_sim_time:=true'], check=False)
            self.get_logger().info(f'map saved to {path}.yaml')


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
