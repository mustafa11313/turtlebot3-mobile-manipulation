"""Arm and gripper motion: MoveIt 2 (collision-aware) with a direct-trajectory fallback."""

import time
from typing import Optional, Sequence

from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint

from .kinematics import JOINT_NAMES


def wait_future(fut, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while not fut.done():
        if time.monotonic() > end:
            return False
        time.sleep(0.005)
    return True


class ArmInterface:

    def __init__(self, node, use_moveit: bool = True, velocity_scaling: float = 0.3):
        self.node = node
        self.log = node.get_logger()
        cb = ReentrantCallbackGroup()
        self.velocity_scaling = velocity_scaling
        self.traj_client = ActionClient(node, FollowJointTrajectory,
                                        '/arm_controller/follow_joint_trajectory', callback_group=cb)
        self.gripper_client = ActionClient(node, GripperCommand,
                                           '/gripper_controller/gripper_cmd', callback_group=cb)
        self.move_group = ActionClient(node, MoveGroup, '/move_action', callback_group=cb) \
            if use_moveit else None
        self.current = None
        from sensor_msgs.msg import JointState
        node.create_subscription(JointState, '/joint_states', self._on_js, 10, callback_group=cb)

    def _on_js(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in JOINT_NAMES):
            self.current = [pos[j] for j in JOINT_NAMES]
        self.gripper_position = pos.get('gripper_left_joint', None)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        ok = self.traj_client.wait_for_server(timeout_sec=timeout) and \
            self.gripper_client.wait_for_server(timeout_sec=timeout)
        if self.move_group is not None and not self.move_group.wait_for_server(timeout_sec=timeout):
            self.log.warn('move_group not available, falling back to direct joint trajectories')
            self.move_group = None
        return ok

    # ------------------------------------------------------------------ arm
    def move_joints(self, q: Sequence[float], duration: Optional[float] = None,
                    plan: bool = True) -> bool:
        """Move to a joint configuration. ``plan`` uses MoveIt for collision checking."""
        if plan and self.move_group is not None:
            ok = self._moveit_joint_goal(q)
            if ok:
                return True
            self.log.warn('MoveIt plan/execute failed, not falling back (collision-aware motion required)')
            return False
        return self._direct(q, duration)

    def _direct(self, q, duration=None) -> bool:
        if duration is None:
            cur = self.current or [0.0] * 4
            dist = max(abs(a - b) for a, b in zip(q, cur))
            duration = max(1.0, dist / (1.5 * self.velocity_scaling))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint(positions=[float(v) for v in q], velocities=[0.0] * 4)
        pt.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points = [pt]
        fut = self.traj_client.send_goal_async(goal)
        if not wait_future(fut, 10.0) or not fut.result().accepted:
            self.log.error('arm trajectory goal rejected')
            return False
        res_fut = fut.result().get_result_async()
        if not wait_future(res_fut, duration + 10.0):
            self.log.error('arm trajectory timed out')
            return False
        code = res_fut.result().result.error_code
        if code != 0:
            self.log.warn(f'arm trajectory finished with error code {code}')
        return self.reached(q, tol=0.05)

    def _moveit_joint_goal(self, q) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'arm'
        req.num_planning_attempts = 3
        req.allowed_planning_time = 3.0
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.velocity_scaling
        c = Constraints()
        for name, v in zip(JOINT_NAMES, q):
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(v), tolerance_above=0.005,
                tolerance_below=0.005, weight=1.0))
        req.goal_constraints = [c]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        fut = self.move_group.send_goal_async(goal)
        if not wait_future(fut, 10.0) or not fut.result().accepted:
            return False
        res_fut = fut.result().get_result_async()
        if not wait_future(res_fut, 60.0):
            return False
        code = res_fut.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.log.warn(f'MoveIt error code {code}')
            return False
        return True

    def reached(self, q, tol=0.03) -> bool:
        time.sleep(0.3)
        if self.current is None:
            return False
        return max(abs(a - b) for a, b in zip(q, self.current)) < tol

    # -------------------------------------------------------------- gripper
    def gripper(self, position: float, max_effort: float = 1.0, timeout: float = 5.0) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        fut = self.gripper_client.send_goal_async(goal)
        if not wait_future(fut, 5.0) or not fut.result().accepted:
            return False
        res_fut = fut.result().get_result_async()
        wait_future(res_fut, timeout)  # a stalled gripper (holding an object) may not "reach"
        time.sleep(0.4)
        return True
