"""ROS 2 telemetry and operator-control boundary for the local Piper bridge."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String


JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_width",
)


class RemotePolicyRosNode(Node):
    """Expose bridge state to ROS while keeping one owner of the CAN device."""

    def __init__(self, node_name: str = "remote_dp3_piper_bridge") -> None:
        super().__init__(node_name)
        self.stop_requested = False
        self.estop_requested = False
        self.feedback_publisher = self.create_publisher(
            JointState, "/remote_dp3/joint_states_feedback", 10
        )
        self.target_publisher = self.create_publisher(
            JointState, "/remote_dp3/joint_target", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/remote_dp3/status", 10
        )
        self.executing_publisher = self.create_publisher(
            Bool, "/remote_dp3/executing", 10
        )
        self.create_subscription(
            Bool, "/remote_dp3/stop", self._on_stop, 10
        )
        self.create_subscription(
            Bool, "/remote_dp3/estop", self._on_estop, 10
        )

    def _on_stop(self, message: Bool) -> None:
        if message.data:
            self.stop_requested = True
            self.get_logger().warning("收到 /remote_dp3/stop，进入保持并退出")

    def _on_estop(self, message: Bool) -> None:
        if message.data:
            self.estop_requested = True
            self.get_logger().error("收到 /remote_dp3/estop，请求 Piper 急停")

    def publish_cycle(
        self,
        state: np.ndarray,
        target: np.ndarray,
        *,
        executing: bool,
        status: dict[str, Any],
    ) -> None:
        self.feedback_publisher.publish(self._joint_state(state))
        self.target_publisher.publish(self._joint_state(target))
        self.executing_publisher.publish(Bool(data=bool(executing)))
        self.status_publisher.publish(
            String(data=json.dumps(status, ensure_ascii=False, separators=(",", ":")))
        )

    def publish_final(self, *, executing: bool, stop_reason: str) -> None:
        self.executing_publisher.publish(Bool(data=bool(executing)))
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {"phase": "stopped", "stop_reason": stop_reason},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )

    def _joint_state(self, values: np.ndarray) -> JointState:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"ROS JointState需要7维有限数值，实际{vector.shape}")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = vector.tolist()
        return message


def init_ros_node(node_name: str) -> RemotePolicyRosNode:
    if not rclpy.ok():
        rclpy.init(args=None)
    return RemotePolicyRosNode(node_name)


def spin_ros_once(node: RemotePolicyRosNode) -> None:
    rclpy.spin_once(node, timeout_sec=0.0)


def shutdown_ros(node: RemotePolicyRosNode | None) -> None:
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
