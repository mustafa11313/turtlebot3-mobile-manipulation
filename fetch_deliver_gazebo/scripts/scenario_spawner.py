#!/usr/bin/env python3
"""Spawn a randomized scenario (target, distractors, dynamic obstacles) into Gazebo.

Parameters:
  layout (str), seed (int), difficulty (str), target_type (str),
  scenario_file (str, optional: load instead of generating),
  output_file (str, optional: where to write the ground-truth scenario JSON)

Dynamic obstacles shuttle between their waypoints via /gazebo/set_entity_state.
They are kinematic, so like a polite pedestrian they wait while the robot is in
their way: otherwise they would shove the robot, which wheel odometry cannot see.
"""

import math

import rclpy
from fetch_deliver_gazebo.layouts import load_layout
from fetch_deliver_gazebo.scenario import Scenario, generate
from fetch_deliver_gazebo.sdf_gen import OBJECT_TYPES, TABLE_HEIGHT, object_sdf, obstacle_sdf, table_sdf
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, QoSProfile


def yaw_pose(x, y, z, yaw) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


class ScenarioSpawner(Node):

    def __init__(self):
        super().__init__('scenario_spawner')
        self.declare_parameter('layout', 'apartment')
        self.declare_parameter('seed', 0)
        self.declare_parameter('difficulty', 'easy')
        self.declare_parameter('target_type', 'red_cylinder')
        self.declare_parameter('scenario_file', '')
        self.declare_parameter('output_file', '')

        layout = load_layout(self.get_parameter('layout').value)
        scenario_file = self.get_parameter('scenario_file').value
        if scenario_file:
            with open(scenario_file) as f:
                self.scenario = Scenario.from_json(f.read())
        else:
            self.scenario = generate(layout, int(self.get_parameter('seed').value),
                                     self.get_parameter('difficulty').value,
                                     self.get_parameter('target_type').value)
        out = self.get_parameter('output_file').value
        if out:
            with open(out, 'w') as f:
                f.write(self.scenario.to_json())

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, '/fetch_deliver/scenario', latched)
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.obstacle_state = {}  # name -> (segment index, progress along segment)
        self.robot_xy = None
        self.create_subscription(ModelStates, '/gazebo/model_states', self._on_models, 5)

    ROBOT = 'turtlebot3_manipulation_system'
    # obstacle half-diagonal (0.21) + robot circumradius (0.27) + margin
    YIELD_DIST = 0.60

    def _on_models(self, msg):
        if self.ROBOT in msg.name:
            p = msg.pose[msg.name.index(self.ROBOT)].position
            self.robot_xy = (p.x, p.y)

    def spawn_all(self) -> bool:
        if not self.spawn_cli.wait_for_service(timeout_sec=60.0):
            self.get_logger().error('/spawn_entity not available')
            return False
        for obj in self.scenario.objects:
            self._spawn(f'{obj.model_name}_table', table_sdf(f'{obj.model_name}_table'),
                        yaw_pose(obj.x, obj.y, 0.0, obj.yaw))
            z = TABLE_HEIGHT + OBJECT_TYPES[obj.object_type]['height'] / 2 + 0.001
            self._spawn(obj.model_name, object_sdf(obj.model_name, obj.object_type),
                        yaw_pose(obj.x, obj.y, z, obj.yaw))
        for obs in self.scenario.obstacles:
            x, y = obs.waypoints[0]
            self._spawn(obs.model_name, obstacle_sdf(obs.model_name), yaw_pose(x, y, 0.0, 0.0))
            self.obstacle_state[obs.model_name] = [0, 0.0]
        self.pub.publish(String(data=self.scenario.to_json()))
        tgt = self.scenario.target
        self.get_logger().info(
            f'Scenario {self.scenario.layout}/{self.scenario.difficulty}/seed={self.scenario.seed}: '
            f'target {tgt.object_type} at ({tgt.x:.2f}, {tgt.y:.2f}), '
            f'{len(self.scenario.objects) - 1} distractors, {len(self.scenario.obstacles)} obstacles')
        if self.scenario.obstacles:
            self.dt = 0.1
            self.create_timer(self.dt, self._move_obstacles)
        return True

    def _spawn(self, name, xml, pose):
        req = SpawnEntity.Request(name=name, xml=xml, initial_pose=pose)
        fut = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().error(f'failed to spawn {name}: {res.status_message if res else "timeout"}')

    def _move_obstacles(self):
        if not self.state_cli.service_is_ready():
            return
        for obs in self.scenario.obstacles:
            st = self.obstacle_state[obs.model_name]
            a = obs.waypoints[st[0] % len(obs.waypoints)]
            b = obs.waypoints[(st[0] + 1) % len(obs.waypoints)]
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            t = st[1] + obs.speed * self.dt / max(seg, 1e-6)
            if t >= 1.0:
                st[0], st[1] = st[0] + 1, 0.0
                continue
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if self.robot_xy is not None:
                cx = a[0] + (b[0] - a[0]) * st[1]
                cy = a[1] + (b[1] - a[1]) * st[1]
                d_next = math.hypot(x - self.robot_xy[0], y - self.robot_xy[1])
                d_now = math.hypot(cx - self.robot_xy[0], cy - self.robot_xy[1])
                if d_next < self.YIELD_DIST and d_next < d_now:
                    continue  # wait for the robot to pass
            st[1] = t
            state = EntityState(name=obs.model_name, pose=yaw_pose(x, y, 0.0, 0.0),
                                reference_frame='world')
            self.state_cli.call_async(SetEntityState.Request(state=state))


def main():
    rclpy.init()
    node = ScenarioSpawner()
    try:
        if node.spawn_all():
            rclpy.spin(node)  # keep latched topic alive and drive obstacles
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
