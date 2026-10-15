"""Mission node: builds and ticks the fetch & deliver behaviour tree."""

import json
import math
import time

import py_trees
import rclpy
from fetch_deliver_gazebo.layouts import load_layout, world_to_map
from py_trees.common import Status
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .robot_io import MissionContext, RobotIO
from .tree import build_tree


class MissionNode(Node):

    def __init__(self):
        super().__init__('fetch_deliver_mission')
        p = self.declare_parameter
        p('layout', 'apartment')
        p('target_label', 'red_cube')
        p('search_strategy', 'waypoints')      # waypoints | frontier
        p('approach_standoff', 0.45)
        p('grasp_standoff', 0.20)
        p('place_reach', 0.19)
        p('max_pick_attempts', 4)
        p('mission_timeout', 900.0)
        p('tick_rate', 10.0)
        p('shutdown_when_done', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        layout = load_layout(g('layout'))
        to_map = lambda pose: world_to_map(layout, pose)  # noqa: E731
        marker = layout.marker_pose
        mx, my, myaw = to_map((marker[0], marker[1], marker[3]))
        place = to_map(layout.dropoff_pose)
        marker_offset = math.hypot(layout.dropoff_pose[0] - marker[0], layout.dropoff_pose[1] - marker[1])
        view_d = 1.3
        self.cfg = {
            'search_waypoints': [to_map(w) for w in layout.search_waypoints],
            'search_strategy': g('search_strategy'),
            'approach_standoff': g('approach_standoff'),
            'grasp_standoff': g('grasp_standoff'),
            'place_reach': g('place_reach'),
            'max_pick_attempts': int(g('max_pick_attempts')),
            'marker_offset': marker_offset,
            'marker_viewpoint': (mx + view_d * math.cos(myaw), my + view_d * math.sin(myaw),
                                 math.atan2(math.sin(myaw + math.pi), math.cos(myaw + math.pi))),
            'dropoff_place': place[:2],
        }
        self.ctx = MissionContext(target_label=g('target_label'))
        cb = ReentrantCallbackGroup()
        self.io = RobotIO(self, self.ctx, cb)
        self.tree = py_trees.trees.BehaviourTree(build_tree(self.io, self.cfg))
        self.tree.setup(timeout=5.0)
        self.tree_pub = self.create_publisher(String, '/fetch_deliver/tree', 2)
        self.timeout = g('mission_timeout')
        self.done = False
        tick_group = MutuallyExclusiveCallbackGroup()   # never tick the tree concurrently
        self.create_timer(1.0 / g('tick_rate'), self.tick, callback_group=tick_group)
        self.create_timer(1.0, self.publish_tree, callback_group=tick_group)
        self.get_logger().info(
            f'mission ready: target={self.ctx.target_label}, layout={layout.name}, '
            f'strategy={self.cfg["search_strategy"]}')

    def tick(self):
        if self.done:
            return
        self.tree.tick()
        status = self.tree.root.status
        tip = self.tree.root.tip()
        self.io.publish_status(tip.name if tip else '')
        elapsed = time.monotonic() - self.ctx.start_time if self.ctx.start_time else 0.0
        if self.ctx.start_time and elapsed > self.timeout and status == Status.RUNNING:
            self.tree.root.stop(Status.INVALID)
            self.finish(False, 'timeout', elapsed)
        elif status == Status.SUCCESS:
            self.finish(True, '', elapsed)
        elif status == Status.FAILURE:
            self.finish(False, f'failed in {self.ctx.phase}', elapsed)

    def finish(self, success: bool, reason: str, elapsed: float):
        self.done = True
        self.ctx.phase = 'DONE' if success else 'FAILED'
        detail = {
            'success': success, 'reason': reason, 'elapsed': elapsed,
            'pick_attempts': self.ctx.pick_attempts, 'nav_failures': self.ctx.nav_failures,
            'recoveries': self.ctx.recoveries,
            'time_to_detect': (self.ctx.first_detection_time - self.ctx.start_time)
            if self.ctx.first_detection_time else None,
        }
        self.io.event('mission_complete', json.dumps(detail), 1.0 if success else 0.0)
        self.io.publish_status('')
        self.get_logger().info(f'MISSION {"SUCCEEDED" if success else "FAILED"} '
                               f'in {elapsed:.1f} s {reason}')

    def publish_tree(self):
        self.tree_pub.publish(String(data=py_trees.display.unicode_tree(
            self.tree.root, show_status=True)))


def main():
    rclpy.init()
    node = MissionNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        while rclpy.ok():
            ex.spin_once(timeout_sec=0.1)
            if node.done and node.get_parameter('shutdown_when_done').value:
                time.sleep(2.0)
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
