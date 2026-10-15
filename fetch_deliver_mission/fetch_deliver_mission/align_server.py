"""Align action server: fine base alignment in front of an object by visual servoing.

Drives the base (cmd_vel) until the object sits at ``standoff`` metres straight
ahead of base_footprint, then averages a burst of stationary observations to
produce the final object estimate used for grasping. While close, it also
checks the object's shape so a same-colored distractor is caught before the
grasp (failure_code WRONG_OBJECT).
"""

import math
import time
from collections import Counter

import numpy as np
import rclpy
from fetch_deliver_interfaces.action import Align
from fetch_deliver_interfaces.msg import ObjectObservation
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AlignServer(Node):

    def __init__(self):
        super().__init__('align_server')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('distance_tolerance', 0.012)
        self.declare_parameter('bearing_tolerance', 0.04)
        self.declare_parameter('lost_timeout', 2.5)
        self.declare_parameter('max_linear', 0.10)
        self.declare_parameter('max_angular', 0.5)
        self.declare_parameter('settle_samples', 12)
        self.cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(ObjectObservation, '/perception/observations', self.on_obs, 50,
                                 callback_group=self.cb)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 5, callback_group=self.cb)
        self.obs = []          # recent observations: (t, label, x_odom, y_odom, x_base, y_base, z)
        self.front_clearance = float('inf')
        self.server = ActionServer(self, Align, '/mission/align', self.execute,
                                   callback_group=self.cb,
                                   cancel_callback=lambda _: CancelResponse.ACCEPT)

    # ------------------------------------------------------------------ io
    def on_scan(self, msg: LaserScan):
        r = np.array(msg.ranges, float)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        front = r[(np.abs(ang) < 0.5) & np.isfinite(r) & (r > msg.range_min)]
        self.front_clearance = float(front.min()) if len(front) else float('inf')

    def on_obs(self, msg: ObjectObservation):
        try:
            tf = self.tf_buffer.lookup_transform(self.get_parameter('odom_frame').value,
                                                 msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                                 timeout=Duration(seconds=0.05))
        except Exception:  # noqa: BLE001
            return
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y = msg.pose.position.x, msg.pose.position.y
        xo = tf.transform.translation.x + math.cos(yaw) * x - math.sin(yaw) * y
        yo = tf.transform.translation.y + math.sin(yaw) * x + math.cos(yaw) * y
        now = time.monotonic()
        self.obs.append((now, msg.label, xo, yo, msg.pose.position.z))
        self.obs = [o for o in self.obs if now - o[0] < 3.0]

    def robot_pose(self):
        tf = self.tf_buffer.lookup_transform(self.get_parameter('odom_frame').value,
                                             self.get_parameter('base_frame').value, Time())
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def stop(self):
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------- execute
    def execute(self, handle):
        goal = handle.request
        color = goal.target_label.partition('_')[0]
        shape = goal.target_label.partition('_')[2]
        standoff = goal.standoff or 0.20
        timeout = goal.timeout or 40.0
        result = Align.Result()
        fb = Align.Feedback()
        start = time.monotonic()
        last_seen = time.monotonic()
        est = None
        votes = Counter()
        tol_d = self.get_parameter('distance_tolerance').value
        tol_b = self.get_parameter('bearing_tolerance').value
        settled = 0

        def finish(code):
            self.stop()
            result.success = code == ''
            result.failure_code = code
            if result.success:
                handle.succeed()
            else:
                handle.abort()
            self.get_logger().info(f'align finished: {code or "OK"} '
                                   f'(d_err {result.final_distance_error:.3f}, '
                                   f'b_err {result.final_bearing_error:.3f})')
            return result

        while rclpy.ok():
            if handle.is_cancel_requested:
                self.stop()
                handle.canceled()
                return result
            if time.monotonic() - start > timeout:
                return finish('TIMEOUT')
            try:
                rx, ry, ryaw = self.robot_pose()
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
                continue
            now = time.monotonic()
            cands = [o for o in self.obs if now - o[0] < 0.4 and o[1].startswith(color + '_')]
            if est is not None:
                cands = [o for o in cands if math.hypot(o[2] - est[0], o[3] - est[1]) < 0.15]
            if cands:
                if est is None:
                    # first lock: closest observation to the robot
                    o = min(cands, key=lambda o: math.hypot(o[2] - rx, o[3] - ry))
                    est = [o[2], o[3], o[4]]
                else:
                    o = cands[-1]
                    est = [0.6 * est[0] + 0.4 * o[2], 0.6 * est[1] + 0.4 * o[3], o[4]]
                last_seen = now
                for c in cands:
                    dist = math.hypot(c[2] - rx, c[3] - ry)
                    if dist < 0.6:
                        votes[c[1].partition('_')[2]] += 1
            elif est is None or now - last_seen > self.get_parameter('lost_timeout').value:
                if est is None and now - start < self.get_parameter('lost_timeout').value:
                    time.sleep(0.05)
                    continue
                return finish('LOST_OBJECT')

            if votes['cylinder'] + votes['cube'] >= 6 and shape:
                other = 'cylinder' if shape == 'cube' else 'cube'
                if votes[other] > 2 * votes[shape]:
                    return finish('WRONG_OBJECT')

            # error in the robot frame
            dx, dy = est[0] - rx, est[1] - ry
            xb = math.cos(ryaw) * dx + math.sin(ryaw) * dy
            yb = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
            d_err = xb - standoff
            b_err = math.atan2(yb, xb)
            fb.distance_error, fb.bearing_error = float(d_err), float(b_err)
            handle.publish_feedback(fb)
            result.final_distance_error, result.final_bearing_error = float(d_err), float(b_err)

            if abs(d_err) < tol_d and abs(b_err) < tol_b:
                self.stop()
                settled += 1
                if settled >= 5:
                    break
            else:
                settled = 0
                cmd = Twist()
                cmd.angular.z = clamp(1.8 * b_err, -self.get_parameter('max_angular').value,
                                      self.get_parameter('max_angular').value)
                if abs(b_err) < 0.25:
                    v = clamp(0.8 * d_err, -0.06, self.get_parameter('max_linear').value)
                    if v > 0 and self.front_clearance < 0.20:
                        return finish('BLOCKED')
                    cmd.linear.x = v if abs(v) > 0.01 else math.copysign(0.01, v)
                self.cmd_pub.publish(cmd)
            time.sleep(0.05)

        # stationary burst for the final estimate
        samples = []
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end and len(samples) < self.get_parameter('settle_samples').value:
            now = time.monotonic()
            for o in self.obs:
                if now - o[0] < 0.2 and o[1].startswith(color + '_') and \
                        math.hypot(o[2] - est[0], o[3] - est[1]) < 0.06:
                    samples.append(o)
                    votes[o[1].partition('_')[2]] += 1
            time.sleep(0.1)
        if samples:
            est = [float(np.median([s[2] for s in samples])), float(np.median([s[3] for s in samples])),
                   float(np.median([s[4] for s in samples]))]
        if shape and votes[shape] == 0 and (votes['cylinder'] + votes['cube']) > 0:
            return finish('WRONG_OBJECT')
        ps = PoseStamped()
        ps.header.frame_id = self.get_parameter('odom_frame').value
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = est
        ps.pose.orientation.w = 1.0
        result.object_pose = ps
        return finish('')


def main():
    rclpy.init()
    node = AlignServer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
