"""Records one mission trial and writes its metrics as JSON.

Inputs: mission events, the ground-truth scenario, Gazebo model states and
perception observations. Output (``output_file``): task success broken down
into found / picked / delivered, time, grasp accuracy, attach-condition pass
rate, placement accuracy, detection precision/recall, localization error and
collisions.
"""

import json
import math
import time

import rclpy
from fetch_deliver_gazebo.layouts import load_layout, map_to_world
from fetch_deliver_gazebo.scenario import Scenario
from fetch_deliver_interfaces.msg import MissionEvent, ObjectObservation
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .metrics import (CollisionCounter, DetectionScore, clearance, in_camera_view, line_of_sight,
                      placement_error)

ROBOT = 'turtlebot3_manipulation_system'


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class MetricsRecorder(Node):

    def __init__(self):
        super().__init__('metrics_recorder')
        self.declare_parameter('layout', 'arena')
        self.declare_parameter('output_file', 'trial.json')
        self.declare_parameter('trial_name', '')
        self.declare_parameter('detection_rate', 3.0)
        self.layout = load_layout(self.get_parameter('layout').value)
        self.output = self.get_parameter('output_file').value
        self.scenario = None
        self.robot = None
        self.models = {}
        self.events = []
        self.attach_results = []
        self.obs_window = []
        self.det = DetectionScore()
        self.target_det = DetectionScore()
        self.collisions = CollisionCounter()
        self.distance = 0.0
        self.last_robot = None
        self.result = None
        self.wall_start = time.time()
        self.loc_errors = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/fetch_deliver/scenario', self.on_scenario, latched)
        self.create_subscription(MissionEvent, '/fetch_deliver/events', self.on_event, 50)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_models, 5)
        self.create_subscription(ObjectObservation, '/perception/observations', self.on_obs, 50)
        self.create_timer(1.0 / self.get_parameter('detection_rate').value, self.score_detections)
        self.create_timer(0.5, self.sample_localization)

    def on_scenario(self, msg):
        self.scenario = Scenario.from_json(msg.data)

    def on_models(self, msg: ModelStates):
        self.models = {n: p for n, p in zip(msg.name, msg.pose)}
        if ROBOT not in self.models:
            return
        p = self.models[ROBOT]
        self.robot = (p.position.x, p.position.y, yaw_of(p.orientation))
        if self.last_robot is not None:
            self.distance += math.hypot(self.robot[0] - self.last_robot[0],
                                        self.robot[1] - self.last_robot[1])
        self.last_robot = self.robot
        boxes = []
        if self.scenario:
            for o in self.scenario.obstacles:
                if o.model_name in self.models:
                    q = self.models[o.model_name]
                    boxes.append((q.position.x, q.position.y, 0.0, 0.3, 0.3))
        self.collisions.update(clearance(self.layout, self.robot, boxes))

    def on_obs(self, msg: ObjectObservation):
        if self.robot is None:
            return
        x, y, yaw = self.robot
        bx, by = msg.pose.position.x, msg.pose.position.y
        wx = x + math.cos(yaw) * bx - math.sin(yaw) * by
        wy = y + math.sin(yaw) * bx + math.cos(yaw) * by
        self.obs_window.append((time.monotonic(), msg.label, wx, wy))

    def score_detections(self):
        now = time.monotonic()
        window = [o for o in self.obs_window if now - o[0] < 0.25]
        self.obs_window = window
        if self.robot is None or self.scenario is None:
            return
        visible = []
        for o in self.scenario.objects:
            p = self.models.get(o.model_name)
            if p is None or p.position.z > 0.08:      # lifted = held by the robot
                continue
            xy = (p.position.x, p.position.y)
            if in_camera_view(self.robot, xy) and line_of_sight(self.layout, self.robot[:2], xy):
                visible.append((o.object_type, xy[0], xy[1]))
        # de-duplicate detections of the same blob within the window (keep latest)
        dets = {}
        for _, label, x, y in window:
            dets[(label.partition('_')[0], round(x, 1), round(y, 1))] = (label, x, y)
        self.det.score_frame(list(dets.values()), visible)
        tgt = self.scenario.target_type
        self.target_det.score_frame([d for d in dets.values() if d[0].split('_')[0] == tgt.split('_')[0]],
                                    [v for v in visible if v[0] == tgt])

    def sample_localization(self):
        """Ground truth vs the robot's own map-frame pose estimate (AMCL or SLAM)."""
        if self.robot is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except Exception:  # noqa: BLE001
            return
        t = tf.transform
        est = map_to_world(self.layout, (t.translation.x, t.translation.y, yaw_of(t.rotation)))
        self.loc_errors.append(math.hypot(est[0] - self.robot[0], est[1] - self.robot[1]))

    def on_event(self, msg: MissionEvent):
        try:
            detail = json.loads(msg.detail) if msg.detail.startswith('{') else msg.detail
        except ValueError:
            detail = msg.detail
        self.events.append({'t': time.time() - self.wall_start, 'event': msg.event, 'phase': msg.phase,
                            'detail': detail, 'value': msg.value})
        if msg.event in ('attach', 'attach_rejected'):
            self.attach_results.append(detail)
        if msg.event == 'mission_complete':
            self.write(detail)

    def write(self, mission_detail):
        attaches = [a for a in self.attach_results if isinstance(a, dict)]
        ok = [a for a in attaches if a.get('success')]
        held = [a.get('message', '').replace('attached ', '') for a in ok]
        target_name = self.scenario.target.model_name if self.scenario else 'target_object'
        released = [e for e in self.events if e['event'] == 'released']
        delivered_ev = [e for e in self.events if e['event'] == 'delivered']
        place_err = None
        delivered_target = False
        if self.scenario and target_name in self.models:
            p = self.models[target_name].position
            place_err = placement_error((p.x, p.y), self.layout.dropoff_pose[:2])
            delivered_target = bool(delivered_ev) and place_err < 0.25
        found = any(e['event'] == 'target_detected' for e in self.events)
        picked_target = target_name in held
        md = mission_detail if isinstance(mission_detail, dict) else {}
        result = {
            'trial': self.get_parameter('trial_name').value,
            'layout': self.layout.name,
            'scenario': json.loads(self.scenario.to_json()) if self.scenario else None,
            'mission_success': bool(md.get('success')),
            'task_success': bool(md.get('success')) and picked_target and delivered_target,
            'found': found,
            'picked_target': picked_target,
            'picked_wrong_object': any(h != target_name for h in held),
            'delivered_target': delivered_target,
            'time_to_complete': md.get('elapsed'),
            'time_to_detect': md.get('time_to_detect'),
            'pick_attempts': md.get('pick_attempts'),
            'recoveries': md.get('recoveries'),
            'nav_failures': md.get('nav_failures'),
            'failure_reason': md.get('reason'),
            'attach_attempts': len(attaches),
            'attach_passed': len(ok),
            'attach_pass_rate': len(ok) / len(attaches) if attaches else None,
            'grasp_error_m': ok[-1].get('gripper_to_object_error') if ok else None,
            'ee_tracking_error_m': ok[-1].get('ee_to_plan_error') if ok else None,
            'attach_conditions': attaches,
            'placement_error_m': place_err if delivered_ev else None,
            'released_events': len(released),
            'detection_precision': self.det.precision,
            'detection_recall': self.det.recall,
            'detection_counts': {'tp': self.det.tp, 'fp': self.det.fp, 'fn': self.det.fn},
            'target_detection_precision': self.target_det.precision,
            'target_detection_recall': self.target_det.recall,
            'collisions': self.collisions.count,
            'min_clearance_m': self.collisions.min_clearance,
            'distance_travelled_m': self.distance,
            'localization_error_mean_m': sum(self.loc_errors) / len(self.loc_errors) if self.loc_errors else None,
            'localization_error_max_m': max(self.loc_errors) if self.loc_errors else None,
            'events': self.events,
        }
        with open(self.output, 'w') as f:
            json.dump(result, f, indent=2, default=float)
        self.result = result
        self.get_logger().info(f'trial written to {self.output}: task_success={result["task_success"]}')


def main():
    rclpy.init()
    node = MetricsRecorder()
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
