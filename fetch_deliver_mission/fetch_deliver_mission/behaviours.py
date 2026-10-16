"""py_trees behaviours for the fetch & deliver mission.

Every action-backed behaviour follows the same life cycle: send the goal in
``initialise``, poll the goal/result futures in ``update`` (never blocking the
tree), cancel in ``terminate`` if it was interrupted.
"""

import json
import math
import time
from typing import Callable, List, Optional

import numpy as np
import py_trees
from fetch_deliver_interfaces.action import Align, Pick, Place
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from py_trees.common import Status
from std_msgs.msg import Empty

from .frontiers import best_frontier, find_frontiers
from .geometry import Pose2D, pick_standoff, wrap
from .robot_io import RobotIO


class ActionBehaviour(py_trees.behaviour.Behaviour):
    """Base class: one ROS 2 action goal per activation."""

    def __init__(self, name: str, io: RobotIO, client, server_timeout: float = 5.0):
        super().__init__(name)
        self.io = io
        self.client = client
        self.server_timeout = server_timeout
        self.goal_future = None
        self.result_future = None
        self.goal_handle = None
        self.sent_at = 0.0

    def make_goal(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def on_result(self, status, result) -> Status:  # pragma: no cover - abstract
        raise NotImplementedError

    def initialise(self):
        self.goal_future = self.result_future = self.goal_handle = None
        goal = self.make_goal()
        if goal is None:
            return
        if not self.client.wait_for_server(timeout_sec=self.server_timeout):
            self.io.log.error(f'{self.name}: action server not available')
            return
        self.goal_future = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
        self.sent_at = time.monotonic()

    def on_feedback(self, _fb):
        pass

    def update(self) -> Status:
        if self.goal_future is None:
            return self.on_result(None, None)
        if self.result_future is None:
            if not self.goal_future.done():
                return Status.RUNNING
            self.goal_handle = self.goal_future.result()
            if not self.goal_handle.accepted:
                self.io.log.warn(f'{self.name}: goal rejected')
                return self.on_result(None, None)
            self.result_future = self.goal_handle.get_result_async()
        if not self.result_future.done():
            return self.running_update()
        res = self.result_future.result()
        return self.on_result(res.status, res.result)

    def running_update(self) -> Status:
        return Status.RUNNING

    def terminate(self, new_status: Status):
        if new_status == Status.INVALID and self.goal_handle is not None and \
                self.result_future is not None and not self.result_future.done():
            self.goal_handle.cancel_goal_async()


# ---------------------------------------------------------------- utilities
class SetPhase(py_trees.behaviour.Behaviour):

    def __init__(self, io: RobotIO, phase: str):
        super().__init__(f'Phase:{phase}')
        self.io, self.phase = io, phase

    def update(self):
        if self.io.ctx.phase != self.phase:
            self.io.ctx.phase = self.phase
            self.io.event('phase', self.phase)
        return Status.SUCCESS


class MarkPickedTable(py_trees.behaviour.Behaviour):
    """Tell perception where the picked object stood, so its table goes on the
    costmaps for the rest of the mission."""

    def __init__(self, io: RobotIO):
        super().__init__('MarkPickedTable')
        self.io = io

    def update(self):
        if self.io.ctx.target_map is not None:
            self.io.mark_picked_table(*self.io.ctx.target_map[:2])
        return Status.SUCCESS


class MarkDropoff(py_trees.behaviour.Behaviour):
    """Put the delivered object on the costmaps (it is below the LiDAR plane too),
    so the way home goes around it."""

    def __init__(self, io: RobotIO):
        super().__init__('MarkDropoff')
        self.io = io

    def update(self):
        if self.io.ctx.dropoff_place is not None:
            self.io.mark_picked_table(*self.io.ctx.dropoff_place)
        return Status.SUCCESS


class TurnTowards(py_trees.behaviour.Behaviour):
    """Rotate in place, closed-loop on yaw, until the robot faces a point."""

    SPEED = 0.5
    TOLERANCE = 0.1

    def __init__(self, name, io: RobotIO, point_fn: Callable, timeout: float = 20.0):
        super().__init__(name)
        self.io, self.point_fn, self.timeout = io, point_fn, timeout
        self.t0 = 0.0

    def initialise(self):
        self.t0 = time.monotonic()

    def update(self):
        robot, point = self.io.robot_pose(), self.point_fn()
        if robot is None or point is None:
            return Status.SUCCESS
        err = wrap(math.atan2(point[1] - robot[1], point[0] - robot[0]) - robot[2])
        if abs(err) < self.TOLERANCE or time.monotonic() - self.t0 > self.timeout:
            self.io.drive()
            return Status.SUCCESS
        self.io.drive(0.0, math.copysign(min(self.SPEED, max(0.15, abs(err))), err))
        return Status.RUNNING

    def terminate(self, new_status):
        self.io.drive()


class WaitForSystem(py_trees.behaviour.Behaviour):
    """RUNNING until localization (map->base TF) and every action server is up."""

    def __init__(self, io: RobotIO):
        super().__init__('WaitForSystem')
        self.io = io
        self.last_log = 0.0

    def update(self):
        missing = [n for n, c in (('navigate_to_pose', self.io.nav), ('align', self.io.align),
                                  ('pick', self.io.pick), ('place', self.io.place))
                   if not c.server_is_ready()]
        pose = self.io.robot_pose()
        if pose is None:
            missing.append('map->base_footprint TF')
        if missing:
            if time.monotonic() - self.last_log > 5.0:
                self.io.log.info(f'waiting for: {", ".join(missing)}')
                self.last_log = time.monotonic()
            return Status.RUNNING
        if self.io.ctx.home is None:
            self.io.ctx.home = pose
        if not self.io.ctx.start_time:
            self.io.ctx.start_time = time.monotonic()
            self.io.event('mission_start', json.dumps({'home': pose}))
        return Status.SUCCESS


class TargetKnown(py_trees.behaviour.Behaviour):

    def __init__(self, io: RobotIO):
        super().__init__('TargetKnown?')
        self.io = io

    def update(self):
        return Status.SUCCESS if self.io.ctx.target_map is not None else Status.FAILURE


class DropoffKnown(py_trees.behaviour.Behaviour):

    def __init__(self, io: RobotIO, marker_offset: float):
        super().__init__('DropoffKnown?')
        self.io, self.offset = io, marker_offset

    def update(self):
        ctx = self.io.ctx
        if ctx.marker_map is not None:
            mx, my, myaw = ctx.marker_map
            ctx.dropoff_place = (mx + self.offset * math.cos(myaw), my + self.offset * math.sin(myaw))
            return Status.SUCCESS
        return Status.FAILURE


def current_dropoff(io: RobotIO, marker_offset: float):
    """Drop-off point, refreshed from the marker whenever it has been seen (it is often
    first seen only on the way to a configured drop-off)."""
    ctx = io.ctx
    if ctx.marker_map is not None:
        mx, my, myaw = ctx.marker_map
        ctx.dropoff_place = (mx + marker_offset * math.cos(myaw), my + marker_offset * math.sin(myaw))
    return ctx.dropoff_place


class UseConfiguredDropoff(py_trees.behaviour.Behaviour):
    """Fallback: the marker was never seen, use the drop-off pose from the config."""

    def __init__(self, io: RobotIO, place_xy):
        super().__init__('UseConfiguredDropoff')
        self.io, self.place_xy = io, place_xy

    def update(self):
        self.io.ctx.dropoff_place = tuple(self.place_xy)
        self.io.event('dropoff_from_config', json.dumps({'x': self.place_xy[0], 'y': self.place_xy[1]}))
        return Status.SUCCESS


# --------------------------------------------------------------- navigation
class NavigateTo(ActionBehaviour):
    """NavigateToPose with recovery for blocked paths (clear costmaps, back up, retry)."""

    def __init__(self, name, io: RobotIO, pose_fn: Callable[[], Optional[Pose2D]],
                 retries: int = 2, abort_fn: Optional[Callable[[], bool]] = None):
        super().__init__(name, io, io.nav)
        self.pose_fn = pose_fn
        self.retries = retries
        self.abort_fn = abort_fn
        self.attempt = 0
        self.recovering = None
        self.goal_pose = None

    def initialise(self):
        self.attempt = 0
        self.recovering = None
        super().initialise()

    def make_goal(self):
        # a retry after a recovery re-sends the same goal: pose_fn may have side
        # effects (the search hands out its next viewpoint on every call)
        pose = self.goal_pose if self.attempt > 0 and self.goal_pose is not None else self.pose_fn()
        return self._goal_for(pose)

    def _goal_for(self, pose):
        if pose is None:
            return None
        self.goal_pose = pose
        g = NavigateToPose.Goal()
        g.pose = self.io.pose_stamped(pose)
        return g

    def running_update(self):
        if self.abort_fn is not None and self.abort_fn():
            self.goal_handle.cancel_goal_async()
            return Status.SUCCESS
        return Status.RUNNING

    def update(self):
        if self.recovering is not None:
            if not self.recovering.done():
                return Status.RUNNING
            self.recovering = None
            super().initialise()
            return Status.RUNNING
        return super().update()

    def on_result(self, status, result):
        if status == 4:  # SUCCEEDED
            return Status.SUCCESS
        if self.abort_fn is not None and self.abort_fn():
            return Status.SUCCESS
        if self.goal_future is None:
            return Status.FAILURE
        self.io.ctx.nav_failures += 1
        if self.attempt >= self.retries:
            self.io.event('nav_failed', json.dumps({'behaviour': self.name}))
            return Status.FAILURE
        # blocked path recovery: clear costmaps and back up before retrying
        self.attempt += 1
        self.io.ctx.recoveries += 1
        self.io.event('recovery', json.dumps({'type': 'blocked_path', 'behaviour': self.name,
                                              'attempt': self.attempt}))
        self.io.clear_costmaps()
        self.recovering = _OdomBackup(self.io, 0.15)
        return Status.RUNNING


class _OdomBackup:
    """Reverse a short distance with cmd_vel, closed-loop on the robot pose, stopping early
    if the LiDAR sees anything behind. Polled through ``done()`` from a behaviour's tick.
    (Nav2's BackUp aborts for the same footprint-stamp reason as its Spin.)"""

    SPEED = 0.05
    REAR_MARGIN = 0.21     # LiDAR to rear footprint edge (0.16 m) plus 5 cm

    def __init__(self, io: RobotIO, distance: float, timeout: float = 12.0):
        self.io, self.distance, self.timeout = io, distance, timeout
        self.start = None
        self.t0 = time.monotonic()
        self.finished = False

    def done(self) -> bool:
        if self.finished:
            return True
        robot = self.io.robot_pose()
        if robot is not None and self.start is None:
            self.start = robot
        moved = 0.0 if robot is None or self.start is None else \
            math.hypot(robot[0] - self.start[0], robot[1] - self.start[1])
        blocked = self.io.rear_clearance < self.REAR_MARGIN + 0.02
        if moved >= self.distance or blocked or time.monotonic() - self.t0 > self.timeout:
            self.io.drive()
            self.finished = True
            return True
        self.io.drive(-self.SPEED, 0.0)
        return False


class SpinInPlace(py_trees.behaviour.Behaviour):
    """Look around by rotating in place, closed-loop on the robot's yaw.

    Nav2's Spin behaviour was not used: its collision checker looks up the
    transform at the published footprint's stamp, which runs slightly ahead of
    odometry when the simulator is below real time, so most spins aborted at
    once. Rotating in place sweeps a circle of the footprint's circumradius, so
    the turn only starts when the LiDAR sees nothing inside that circle.
    """

    SPEED = 0.5            # rad/s: keeps a small object in view for several camera frames
    CLEARANCE = 0.33       # m from the LiDAR; footprint circumradius 0.27 m plus margin

    def __init__(self, name, io, angle=2 * math.pi, abort_fn=None):
        super().__init__(name)
        self.io, self.angle, self.abort_fn = io, angle, abort_fn
        self.turned = 0.0
        self.last_yaw = None
        self.t0 = 0.0

    def initialise(self):
        self.turned, self.last_yaw, self.t0 = 0.0, None, time.monotonic()

    def update(self):
        if self.abort_fn is not None and self.abort_fn():
            self.io.drive()
            return Status.SUCCESS
        if self.last_yaw is None and self.io.min_scan_range < self.CLEARANCE:
            self.io.event('spin_skipped', json.dumps({'behaviour': self.name,
                                                      'clearance': self.io.min_scan_range}))
            return Status.SUCCESS
        robot = self.io.robot_pose()
        if robot is None:
            return Status.RUNNING
        if self.last_yaw is not None:
            self.turned += abs(wrap(robot[2] - self.last_yaw))
        self.last_yaw = robot[2]
        if self.turned >= self.angle or time.monotonic() - self.t0 > 3.0 * self.angle / self.SPEED + 10.0:
            self.io.drive()
            return Status.SUCCESS
        self.io.drive(0.0, self.SPEED)
        return Status.RUNNING

    def terminate(self, new_status):
        self.io.drive()


class BackUpBehaviour(py_trees.behaviour.Behaviour):

    def __init__(self, name, io, distance=0.15):
        super().__init__(name)
        self.io, self.distance = io, distance
        self.motion = None

    def initialise(self):
        self.motion = _OdomBackup(self.io, self.distance)

    def update(self):
        return Status.SUCCESS if self.motion.done() else Status.RUNNING

    def terminate(self, new_status):
        self.io.drive()


# ------------------------------------------------------------------- search
class SearchForTarget(py_trees.behaviour.Behaviour):
    """Visit search viewpoints (configured waypoints or frontiers), looking around at
    each one, until the target is detected. FAILURE once every viewpoint is used up."""

    def __init__(self, io: RobotIO, waypoints: List[Pose2D], strategy: str = 'waypoints',
                 spin_at_waypoints: bool = True):
        super().__init__('SearchForTarget')
        self.io = io
        self.waypoints = list(waypoints)
        self.strategy = strategy
        self.spin_at_waypoints = spin_at_waypoints
        self.index = 0
        self.frontier_blacklist = []
        found = lambda: self.io.ctx.target_map is not None  # noqa: E731
        self.nav = NavigateTo('SearchNav', io, self._next_viewpoint, retries=1, abort_fn=found)
        self.spin = SpinInPlace('LookAround', io, abort_fn=found)
        self.child = None
        self.current_viewpoint = None

    def _next_viewpoint(self) -> Optional[Pose2D]:
        robot = self.io.robot_pose()
        if self.strategy == 'frontier' and self.io.map_msg is not None and robot is not None:
            m = self.io.map_msg
            grid = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
            fr = find_frontiers(grid, m.info.resolution,
                                (m.info.origin.position.x, m.info.origin.position.y),
                                robot[:2], blacklist=self.frontier_blacklist)
            f = best_frontier(fr)
            if f is not None:
                yaw = math.atan2(f.centroid[1] - f.goal[1], f.centroid[0] - f.goal[0])
                self.current_viewpoint = (f.goal[0], f.goal[1], yaw)
                return self.current_viewpoint
        # waypoints: nearest unvisited first (greedy tour)
        if self.index >= len(self.waypoints):
            return None
        if robot is not None:
            rest = self.waypoints[self.index:]
            j = min(range(len(rest)), key=lambda k: math.hypot(rest[k][0] - robot[0],
                                                               rest[k][1] - robot[1]))
            rest[0], rest[j] = rest[j], rest[0]
            self.waypoints[self.index:] = rest
        self.current_viewpoint = self.waypoints[self.index]
        self.index += 1
        return self.current_viewpoint

    def initialise(self):
        self.child = None
        self.started = False

    def update(self):
        found = self.io.ctx.target_map is not None
        if found:
            return Status.SUCCESS
        if self.child is None:
            self.child = self.spin if (self.spin_at_waypoints and not self.started) else self.nav
            self.started = True
        self.child.tick_once()
        status = self.child.status
        if status == Status.RUNNING:
            return Status.RUNNING
        if self.io.ctx.target_map is not None:
            return Status.SUCCESS
        if self.child is self.nav:
            if status == Status.FAILURE:
                if self.nav.goal_future is None:          # no viewpoints left
                    self.io.event('search_exhausted')
                    return Status.FAILURE
                if self.current_viewpoint is not None:  # unreachable: skip it
                    self.frontier_blacklist.append(self.current_viewpoint[:2])
                self.child = self.nav
            else:
                self.child = self.spin if self.spin_at_waypoints else self.nav
        else:
            self.child = self.nav
        return Status.RUNNING

    def terminate(self, new_status):
        if self.child is not None and self.child.status == Status.RUNNING:
            self.child.stop(Status.INVALID)


# --------------------------------------------------------------- manipulate
class ApproachTarget(NavigateTo):
    """Drive to a standoff pose facing the target. The first sighting is often from
    far away, where a pixel of error is tens of centimetres, so the goal is re-sent
    whenever the tracked estimate moves noticeably as the robot gets closer."""

    RETARGET_DIST = 0.20
    MAX_RETARGETS = 6

    def __init__(self, io: RobotIO, standoff: float, grasp_standoff: Optional[float] = None):
        super().__init__('ApproachTarget', io, self._standoff, retries=1)
        self.standoff = standoff
        self.grasp_standoff = grasp_standoff
        self.goal_target = None
        self.retargets = 0

    def initialise(self):
        self.retargets = 0
        super().initialise()

    def running_update(self):
        target = self.io.ctx.target_map
        if target is not None and self.goal_target is not None and \
                self.retargets < self.MAX_RETARGETS and time.monotonic() - self.sent_at > 2.0 and \
                math.hypot(target[0] - self.goal_target[0], target[1] - self.goal_target[1]) > self.RETARGET_DIST:
            goal = self._goal_for(self._standoff())
            if goal is not None:
                self.retargets += 1
                self.io.event('retarget', json.dumps({'x': target[0], 'y': target[1]}))
                # Nav2 preempts the running goal with the new one
                self.goal_future = self.client.send_goal_async(goal)
                self.result_future = self.goal_handle = None
                self.sent_at = time.monotonic()
        return Status.RUNNING

    def _standoff(self):
        ctx = self.io.ctx
        robot = self.io.robot_pose()
        if ctx.target_map is None or robot is None:
            return None
        self.goal_target = tuple(ctx.target_map[:2])
        if not ctx.approach_excludes and \
                math.hypot(ctx.target_map[0] - robot[0], ctx.target_map[1] - robot[1]) < self.standoff + 0.15:
            # already close (and no failed attempt from here): just face it
            yaw = math.atan2(ctx.target_map[1] - robot[1], ctx.target_map[0] - robot[0])
            ctx.last_approach_pose = (robot[0], robot[1], yaw)
            return ctx.last_approach_pose
        pose = pick_standoff(ctx.target_map[:2], robot[:2], self.standoff, self.io.costmap,
                             ctx.approach_excludes, grasp_distance=self.grasp_standoff)
        ctx.last_approach_pose = pose
        return pose

    def on_result(self, status, result):
        out = super().on_result(status, result)
        if out == Status.FAILURE and getattr(self, 'goal_pose', None) is not None:
            self.io.ctx.approach_excludes.append(self.goal_pose)
        return out


class AlignWithTarget(ActionBehaviour):

    def __init__(self, io: RobotIO, standoff: float):
        super().__init__('AlignWithTarget', io, io.align)
        self.standoff = standoff

    def make_goal(self):
        g = Align.Goal()
        g.target_label = self.io.ctx.target_label
        g.standoff = float(self.standoff)
        g.timeout = 40.0
        return g

    def on_result(self, status, result):
        if result is not None and result.success:
            self.io.ctx.aligned_pose = result.object_pose
            return Status.SUCCESS
        code = result.failure_code if result is not None else 'NO_SERVER'
        self.io.event('align_failed', code)
        if code in ('BLOCKED', 'LOST_OBJECT') and self.io.ctx.last_approach_pose is not None:
            # come from another side next time
            self.io.ctx.approach_excludes.append(self.io.ctx.last_approach_pose)
        if code == 'WRONG_OBJECT':
            self.io.reject_pub.publish(Empty())
            self.io.ctx.target_map = None          # forces a new search
            self.io.event('recovery', json.dumps({'type': 'wrong_object'}))
        elif code == 'LOST_OBJECT':
            self.io.event('recovery', json.dumps({'type': 'object_lost'}))
            self.io.ctx.target_map = None if time.monotonic() - self.io.ctx.target_stamp > 5.0 \
                else self.io.ctx.target_map
        self.io.ctx.recoveries += 1
        return Status.FAILURE


class PickTarget(ActionBehaviour):

    def __init__(self, io: RobotIO):
        super().__init__('PickTarget', io, io.pick)

    def make_goal(self):
        if self.io.ctx.aligned_pose is None:
            return None
        self.io.ctx.pick_attempts += 1
        g = Pick.Goal()
        g.object_model = ''
        g.object_pose = self.io.ctx.aligned_pose
        return g

    def on_result(self, status, result):
        if result is not None and result.success:
            self.io.ctx.holding = True
            self.io.event('picked', json.dumps({'grasp_error': result.gripper_to_object_error}),
                          result.gripper_to_object_error)
            return Status.SUCCESS
        code = result.failure_code if result is not None else 'NO_SERVER'
        self.io.event('recovery', json.dumps({'type': 'failed_grasp', 'code': code}))
        self.io.ctx.recoveries += 1
        self.io.ctx.aligned_pose = None
        return Status.FAILURE


class PlaceAtDropoff(ActionBehaviour):

    def __init__(self, io: RobotIO):
        super().__init__('PlaceAtDropoff', io, io.place)

    def make_goal(self):
        g = Place.Goal()
        x, y = self.io.ctx.dropoff_place
        ps = PoseStamped()
        ps.header.frame_id = self.io.ctx.map_frame
        ps.header.stamp = self.io.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), 0.0165
        ps.pose.orientation.w = 1.0
        g.place_pose = ps
        return g

    def on_result(self, status, result):
        if result is not None and result.success:
            self.io.ctx.holding = False
            p = result.released_pose.position
            self.io.forget_near(*self.io.ctx.dropoff_place)
            self.io.event('delivered', json.dumps({'released_world': [p.x, p.y, p.z],
                                                   'place_target_map': self.io.ctx.dropoff_place}))
            return Status.SUCCESS
        self.io.event('place_failed', result.failure_code if result else 'NO_SERVER')
        return Status.FAILURE


def fine_approach_command(robot: Pose2D, point, reach: float, tol_d: float = 0.015,
                          tol_b: float = 0.03):
    """Velocity command that brings ``point`` to ``reach`` metres straight ahead.

    Returns (v, w, done). Turns first, then drives straight, so the base never
    swings its footprint far from where Nav2 left it.
    """
    dx, dy = point[0] - robot[0], point[1] - robot[1]
    xb = math.cos(robot[2]) * dx + math.sin(robot[2]) * dy
    yb = -math.sin(robot[2]) * dx + math.cos(robot[2]) * dy
    d_err = math.hypot(xb, yb) - reach
    b_err = math.atan2(yb, xb)
    if abs(d_err) < tol_d and abs(b_err) < tol_b:
        return 0.0, 0.0, True
    w = max(-0.4, min(0.4, 1.5 * b_err))
    if abs(w) < 0.08 and abs(b_err) >= tol_b:
        w = math.copysign(0.08, b_err)
    v = 0.0
    if abs(b_err) < 0.15 and abs(d_err) >= tol_d:
        v = max(-0.05, min(0.05, 0.8 * d_err))
        if abs(v) < 0.012:
            v = math.copysign(0.012, d_err)
    return v, w, False


class FineApproachPoint(py_trees.behaviour.Behaviour):
    """Closed-loop base adjustment after Nav2 (whose goal tolerance is several cm) so
    a map point ends up exactly at the arm's reach. Fails only on timeout; the
    place stage still clamps an out-of-reach target, so this is best effort."""

    def __init__(self, name: str, io: RobotIO, point_fn: Callable, reach: float, timeout: float = 20.0):
        super().__init__(name)
        self.io, self.point_fn, self.reach, self.timeout = io, point_fn, reach, timeout
        self.t0 = 0.0
        self.settled = 0

    def initialise(self):
        self.t0 = time.monotonic()
        self.settled = 0

    def update(self):
        point, robot = self.point_fn(), self.io.robot_pose()
        if point is None:
            return Status.FAILURE
        if robot is None:
            return Status.RUNNING
        if time.monotonic() - self.t0 > self.timeout:
            self.io.drive()
            self.io.event('fine_approach_timeout')
            return Status.SUCCESS
        v, w, done = fine_approach_command(robot, point, self.reach)
        self.io.drive(v, w)
        self.settled = self.settled + 1 if done else 0
        return Status.SUCCESS if self.settled >= 3 else Status.RUNNING

    def terminate(self, new_status):
        self.io.drive()


def dropoff_standoff(io: RobotIO, reach: float) -> Optional[Pose2D]:
    """Robot pose that puts the drop-off point ``reach`` metres straight ahead."""
    ctx = io.ctx
    if ctx.dropoff_place is None:
        return None
    px, py = ctx.dropoff_place
    robot = io.robot_pose() or (0.0, 0.0, 0.0)
    # face the marker if known (approach from the open side), else come from the robot
    if ctx.marker_map is not None:
        heading = wrap(ctx.marker_map[2] + math.pi)
        cands = [heading + math.radians(o) for o in (0, 30, -30, 60, -60)]
    else:
        base = math.atan2(py - robot[1], px - robot[0])
        cands = [base + math.radians(o) for o in (0, 45, -45, 90, -90)]
    for h in cands:
        pose = (px - reach * math.cos(h), py - reach * math.sin(h), h)
        if io.costmap is None or io.costmap.footprint_free(pose):
            return pose
    h = cands[0]
    return (px - reach * math.cos(h), py - reach * math.sin(h), h)
