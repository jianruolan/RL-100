#!/usr/bin/env python3
"""Local 100 Hz safety/trajectory node for remote DP3 action chunks."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from piper_msgs.srv import Enable

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.local import piper_remote_runtime as runtime


class ActionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("remote_dp3_action_planner")
        self.declare_parameter("feedback_topic", "/joint_states_feedback")
        self.declare_parameter("action_topic", "/remote_dp3/action_chunk")
        self.declare_parameter("command_topic", "/remote_dp3/joint_cmd")
        self.declare_parameter("rate", 100.0)
        self.declare_parameter("max_action_age_ms", 250.0)
        self.declare_parameter("state_stale_seconds", 0.5)
        self.declare_parameter("max_joint_speed_rad_s", 0.20)
        self.declare_parameter("max_joint_accel_rad_s2", 0.50)
        self.declare_parameter("max_gripper_speed_m_s", 0.02)
        self.declare_parameter("tracking_error_rad", 0.10)
        self.state: np.ndarray | None = None
        self.planner: runtime.JointTrajectoryPlanner | None = None
        self.active_target: np.ndarray | None = None
        self.last_action_monotonic = 0.0
        self.last_state_monotonic = 0.0
        self.stop_requested = False
        self.action_min = np.r_[runtime.PIPER_HARD_LOWER_RAD, 0.0]
        self.action_max = np.r_[runtime.PIPER_HARD_UPPER_RAD, runtime.GRIPPER_UPPER_M]
        self.pub = self.create_publisher(JointState, str(self.get_parameter("command_topic").value), 10)
        self.status_pub = self.create_publisher(String, "/remote_dp3/planner_status", 10)
        self.enable_client = self.create_client(Enable, "/enable_srv")
        self.create_subscription(JointState, str(self.get_parameter("feedback_topic").value), self.on_state, 10)
        self.create_subscription(String, str(self.get_parameter("action_topic").value), self.on_action, 10)
        self.create_subscription(Bool, "/remote_dp3/stop", self.on_stop, 10)
        self.create_subscription(Bool, "/remote_dp3/estop", self.on_estop, 10)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate").value), self.control_tick)

    def on_state(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        gripper_name = "gripper" if "gripper" in values else "gripper_width"
        if all(name in values for name in names) and gripper_name in values:
            self.state = np.asarray([*(values[name] for name in names), values[gripper_name]], dtype=np.float32)
            self.last_state_monotonic = time.monotonic()
            if self.planner is None:
                self.planner = runtime.JointTrajectoryPlanner(
                    self.state,
                    float(self.get_parameter("max_joint_speed_rad_s").value),
                    float(self.get_parameter("max_joint_accel_rad_s2").value),
                    float(self.get_parameter("max_gripper_speed_m_s").value),
                    float(self.get_parameter("tracking_error_rad").value),
                )

    def on_action(self, message: String) -> None:
        if self.stop_requested or self.state is None or self.planner is None:
            return
        try:
            value = json.loads(message.data)
            action = np.asarray(value["action_chunk"], dtype=np.float32)
            if action.shape != (4, 7) or not np.all(np.isfinite(action)):
                raise ValueError(f"action_chunk shape={action.shape}, expected=(4,7)")
            if value.get("protocol_version") != runtime.PROTOCOL_VERSION:
                raise ValueError("protocol version mismatch")
            self.action_min = np.maximum(self.action_min, np.asarray(value.get("action_min", self.action_min)))
            self.action_max = np.minimum(self.action_max, np.asarray(value.get("action_max", self.action_max)))
            self.active_target = np.clip(action[0], self.action_min, self.action_max)
            self.last_action_monotonic = time.monotonic()
            self.status_pub.publish(String(data=json.dumps({"phase": "action_accepted", "sequence_id": value.get("sequence_id")})))
        except Exception as exc:
            self.active_target = None
            self.status_pub.publish(String(data=json.dumps({"phase": "action_rejected", "error": repr(exc)})))

    def on_stop(self, message: Bool) -> None:
        if message.data:
            self.stop_requested = True
            self.active_target = None

    def on_estop(self, message: Bool) -> None:
        if not message.data:
            return
        self.stop_requested = True
        self.active_target = None
        if self.enable_client.service_is_ready():
            request = Enable.Request()
            request.enable_request = False
            self.enable_client.call_async(request)
        self.status_pub.publish(String(data=json.dumps({"phase": "estop_disable_requested"})))

    def control_tick(self) -> None:
        if self.state is None or self.planner is None:
            return
        now = time.monotonic()
        age = now - self.last_action_monotonic
        state_stale = now - self.last_state_monotonic > float(self.get_parameter("state_stale_seconds").value)
        expired = self.last_action_monotonic <= 0 or age * 1000.0 > float(self.get_parameter("max_action_age_ms").value)
        target = None if self.stop_requested or expired or state_stale else self.active_target
        planned = self.planner.hold(self.state, 0.01) if target is None else self.planner.step(target, self.state, 0.01)
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        message.position = planned.tolist()
        self.pub.publish(message)


def main() -> None:
    rclpy.init()
    node = ActionPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
