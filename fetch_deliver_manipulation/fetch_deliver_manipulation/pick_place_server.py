"""Pick and Place action servers for the OpenMANIPULATOR-X.

Grasps are planned with the closed-form solver in ``kinematics`` (most
top-down reachable pitch), moved collision-aware through MoveIt 2, and
"closed" through the honest link-attach service: the attach only succeeds if
Gazebo confirms the arm reached the planned pose, the fingers are closed and
the object is between them. Every attempt is reported as a MissionEvent so
the benchmark can compute grasp accuracy and the attach pass rate.
"""

import json
import math
import time

import rclpy
from fetch_deliver_interfaces.action import Pick, Place
from fetch_deliver_interfaces.msg import MissionEvent
from fetch_deliver_interfaces.srv import AttachObject, DetachObject
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from . import kinematics as kin
from .arm_interface import ArmInterface, wait_future

STOW = (0.0, -1.5, 1.2, 0.6)
CARRY = (0.0, -1.2, 0.6, 1.1)


def transform_point(tf_buffer, pose: PoseStamped, target: str, timeout: float = 1.0):
    """Transform a PoseStamped position into ``target``; returns (x, y, z)."""
    t = tf_buffer.lookup_transform(target, pose.header.frame_id, Time(),
                                   timeout=Duration(seconds=timeout))
    q = t.transform.rotation
    p = pose.pose.position
    rx, ry, rz = rotate_vector((q.x, q.y, q.z, q.w), (p.x, p.y, p.z))
    tr = t.transform.translation
    return (rx + tr.x, ry + tr.y, rz + tr.z)


def rotate_vector(q, v):
    """Rotate vector v by unit quaternion q = (x, y, z, w)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


class PickPlaceServer(Node):

    def __init__(self):
        super().__init__('pick_place_server')
        self.declare_parameter('use_moveit', True)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('pregrasp_back', 0.06)
        self.declare_parameter('pregrasp_up', 0.04)
        self.declare_parameter('lift_height', 0.08)
        self.declare_parameter('grasp_z_offset', 0.0)
        self.declare_parameter('velocity_scaling', 0.4)
        self.declare_parameter('gripper_close_effort', 1.0)

        self.base = self.get_parameter('base_frame').value
        cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm = ArmInterface(self, self.get_parameter('use_moveit').value,
                                self.get_parameter('velocity_scaling').value)
        self.attach_cli = self.create_client(AttachObject, '/fetch_deliver/attach', callback_group=cb)
        self.detach_cli = self.create_client(DetachObject, '/fetch_deliver/detach', callback_group=cb)
        self.events = self.create_publisher(MissionEvent, '/fetch_deliver/events', 20)
        self.holding = ''
        self.busy = False

        self.pick_server = ActionServer(
            self, Pick, '/manipulation/pick', self.execute_pick, callback_group=cb,
            goal_callback=self._goal_cb, cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.place_server = ActionServer(
            self, Place, '/manipulation/place', self.execute_place, callback_group=cb,
            goal_callback=self._goal_cb, cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.create_timer(0.5, self._startup, callback_group=cb)
        self._started = False

    def _startup(self):
        if self._started:
            return
        self._started = True
        if self.arm.wait_ready():
            self.arm.gripper(kin.GRIPPER_OPEN)
            self.arm.move_joints(STOW, plan=False, duration=3.0)
            self.get_logger().info('pick/place server ready (arm stowed)')

    def _goal_cb(self, _goal):
        return GoalResponse.REJECT if self.busy else GoalResponse.ACCEPT

    def _event(self, event, detail='', value=0.0):
        msg = MissionEvent(event=event, phase='MANIPULATION', detail=detail, value=float(value))
        msg.header.stamp = self.get_clock().now().to_msg()
        self.events.publish(msg)

    def _pose_msg(self, xyz, pitch, yaw) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        qx, qy, qz, qw = kin.yaw_to_quaternion_rpy(0.0, pitch, yaw)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
        return p

    # ------------------------------------------------------------------ pick
    def execute_pick(self, handle):
        self.busy = True
        try:
            return self._pick(handle)
        finally:
            self.busy = False

    def _pick(self, handle):
        goal = handle.request
        result = Pick.Result()
        fb = Pick.Feedback()

        def stage(name):
            fb.stage = name
            handle.publish_feedback(fb)
            self.get_logger().info(f'[pick] {name}')

        def fail(code, message):
            result.success, result.failure_code, result.message = False, code, message
            self._event('pick_failed', json.dumps({'code': code, 'message': message}))
            self.get_logger().warn(f'[pick] failed: {code} {message}')
            handle.abort()
            return result

        if self.holding:
            return fail('ALREADY_HOLDING', self.holding)
        try:
            obj = transform_point(self.tf_buffer, goal.object_pose, self.base)
        except Exception as e:  # noqa: BLE001
            return fail('LOST_OBJECT', f'cannot transform object pose: {e}')
        obj = (obj[0], obj[1], obj[2] + self.get_parameter('grasp_z_offset').value)

        sol = kin.solve_grasp(obj)
        if sol is None:
            return fail('NO_IK', f'object at ({obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}) not reachable')
        ee_xyz, pitch, yaw = kin.forward(sol.joints)
        back = self.get_parameter('pregrasp_back').value
        up = self.get_parameter('pregrasp_up').value
        pre = None
        for b, u in ((back, up), (back * 0.6, up), (0.0, up), (0.0, up * 0.5)):
            pre = kin.inverse(kin.approach_point(obj, pitch, yaw, b, u), pitch) or \
                kin.solve_grasp(kin.approach_point(obj, pitch, yaw, b, u))
            if pre is not None:
                break
        if pre is None:
            return fail('NO_IK', 'no reachable pre-grasp')

        stage('OPEN')
        self.arm.gripper(kin.GRIPPER_OPEN)
        stage('PRE_GRASP')
        if not self.arm.move_joints(pre.joints):
            return fail('PLAN_FAILED', 'could not reach pre-grasp')
        if handle.is_cancel_requested:
            handle.canceled()
            return result
        stage('GRASP')
        if not self.arm.move_joints(sol.joints, plan=False, duration=1.5):
            self.get_logger().warn('grasp pose tracking error above tolerance, attempting anyway')
        stage('CLOSE')
        self.arm.gripper(kin.GRIPPER_CLOSED, self.get_parameter('gripper_close_effort').value)

        stage('ATTACH')
        req = AttachObject.Request(object_model=goal.object_model, object_link='',
                                   planned_grasp_pose=self._pose_msg(ee_xyz, pitch, yaw))
        if not self.attach_cli.wait_for_service(timeout_sec=5.0):
            return fail('ATTACH_REJECTED', 'attach service unavailable')
        fut = self.attach_cli.call_async(req)
        if not wait_future(fut, 10.0):
            return fail('ATTACH_REJECTED', 'attach service timeout')
        res = fut.result()
        detail = json.dumps({
            'object': goal.object_model, 'success': res.success, 'message': res.message,
            'cond_pose_reached': res.cond_pose_reached,
            'cond_gripper_closed': res.cond_gripper_closed,
            'cond_within_tolerance': res.cond_within_tolerance,
            'ee_to_plan_error': res.ee_to_plan_error,
            'ee_to_plan_angle_error': res.ee_to_plan_angle_error,
            'gripper_to_object_error': res.gripper_to_object_error,
            'gripper_opening': res.gripper_opening,
            'planned_pitch_deg': math.degrees(pitch),
            'estimated_object_base': obj,
        })
        self._event('attach' if res.success else 'attach_rejected', detail,
                    res.gripper_to_object_error)
        result.gripper_to_object_error = res.gripper_to_object_error
        result.ee_to_plan_error = res.ee_to_plan_error
        if not res.success:
            self.arm.gripper(kin.GRIPPER_OPEN)
            self.arm.move_joints(pre.joints, plan=False, duration=1.5)
            self.arm.move_joints(STOW)
            return fail('ATTACH_REJECTED', res.message)
        self.holding = goal.object_model

        stage('LIFT')
        lift = kin.solve_grasp(kin.approach_point(obj, pitch, yaw, 0.0,
                                                  self.get_parameter('lift_height').value))
        if lift is not None:
            self.arm.move_joints(lift.joints, plan=False, duration=1.5)
        stage('STOW')
        if not self.arm.move_joints(CARRY):
            self.arm.move_joints(CARRY, plan=False, duration=2.5)
        result.success = True
        result.message = res.message
        handle.succeed()
        self.get_logger().info(
            f'[pick] success: grasp error {res.gripper_to_object_error * 1000:.1f} mm, '
            f'EE tracking error {res.ee_to_plan_error * 1000:.1f} mm')
        return result

    # ----------------------------------------------------------------- place
    def execute_place(self, handle):
        self.busy = True
        try:
            return self._place(handle)
        finally:
            self.busy = False

    def _place(self, handle):
        goal = handle.request
        result = Place.Result()
        fb = Place.Feedback()

        def stage(name):
            fb.stage = name
            handle.publish_feedback(fb)
            self.get_logger().info(f'[place] {name}')

        def fail(code, message):
            result.success, result.failure_code, result.message = False, code, message
            self._event('place_failed', json.dumps({'code': code, 'message': message}))
            handle.abort()
            return result

        if not self.holding:
            return fail('NOTHING_HELD', '')
        target = None
        if goal.place_pose.header.frame_id:
            try:
                target = transform_point(self.tf_buffer, goal.place_pose, self.base)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'place pose not transformable ({e}), using default')
        default = (0.19, 0.0, 0.0125 + 0.004)
        sol = kin.solve_grasp(target) if target else None
        if sol is None:
            if target:
                self.get_logger().warn(
                    f'place target ({target[0]:.2f}, {target[1]:.2f}) out of reach, placing in front')
            target = default
            sol = kin.solve_grasp(target)
        _, pitch, yaw = kin.forward(sol.joints)
        pre = kin.solve_grasp(kin.approach_point(target, pitch, yaw, 0.0,
                                                 self.get_parameter('pregrasp_up').value))
        stage('PRE_PLACE')
        if pre is None or not self.arm.move_joints(pre.joints):
            if pre is None or not self.arm.move_joints(pre.joints, plan=False, duration=2.5):
                return fail('PLAN_FAILED', 'could not reach pre-place')
        stage('PLACE')
        self.arm.move_joints(sol.joints, plan=False, duration=1.5)
        stage('RELEASE')
        self.arm.gripper(kin.GRIPPER_OPEN)
        fut = self.detach_cli.call_async(DetachObject.Request(object_model=self.holding))
        if not wait_future(fut, 10.0) or not fut.result().success:
            return fail('DETACH_FAILED', fut.result().message if fut.done() else 'timeout')
        time.sleep(0.5)
        result.released_pose = fut.result().object_pose
        self._event('released', json.dumps({
            'object': self.holding,
            'x': result.released_pose.position.x, 'y': result.released_pose.position.y,
            'z': result.released_pose.position.z}))
        self.holding = ''
        stage('RETREAT')
        self.arm.move_joints(pre.joints, plan=False, duration=1.5)
        if not self.arm.move_joints(STOW):
            self.arm.move_joints(STOW, plan=False, duration=2.5)
        result.success = True
        handle.succeed()
        return result


def main():
    rclpy.init()
    node = PickPlaceServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
