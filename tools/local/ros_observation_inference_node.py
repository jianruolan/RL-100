#!/usr/bin/env python3
"""ROS observation builder and asynchronous remote DP3 client.

This node never opens CAN and never sends Piper commands. A separate planner
node owns the command topic; the Piper ROS driver remains the only CAN owner.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local import piper_remote_runtime as runtime


class ObservationInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("remote_dp3_observation_inference")
        self.declare_parameter("server", "127.0.0.1:50051")
        self.declare_parameter("joint_topic", "/joint_states_feedback")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("action_topic", "/remote_dp3/action_chunk")
        self.declare_parameter("status_topic", "/remote_dp3/inference_status")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("max_sensor_gap_ms", 150.0)
        self.declare_parameter("expected_output_name", runtime_name())
        self.declare_parameter("expected_policy_subdir", "bc")

        self.bridge = CvBridge()
        self.joint: np.ndarray | None = None
        self.color: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.color_stamp_ns = 0
        self.depth_stamp_ns = 0
        self.depth_intrinsics: tuple[float, float, float, float] | None = None
        self.sequence_id = 0
        self.episode_id = f"ros-{uuid.uuid4().hex}"
        self.last_submit_monotonic = 0.0
        self.last_result_sequence = -1

        self.client = runtime.PolicyClient(
            str(self.get_parameter("server").value), 5.0
        )
        self.contract = self.client.get_contract(5.0)
        self.contract.validate_model_identity(
            str(self.get_parameter("expected_output_name").value),
            str(self.get_parameter("expected_policy_subdir").value),
        )
        self.client.reset_episode(self.episode_id, 5.0)
        self.worker = runtime.AsyncPolicyClient(
            self.client, 0.2, self.contract.n_action_steps
        )
        self.worker.start()

        self.action_pub = self.create_publisher(
            String, str(self.get_parameter("action_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.create_subscription(
            JointState, str(self.get_parameter("joint_topic").value), self.on_joint, 10
        )
        self.create_subscription(
            Image, str(self.get_parameter("color_topic").value), self.on_color, 10
        )
        self.create_subscription(
            Image, str(self.get_parameter("depth_topic").value), self.on_depth, 10
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self.on_camera_info,
            10,
        )
        period = 1.0 / float(self.get_parameter("fps").value)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            f"remote inference ready: server={self.get_parameter('server').value}, "
            f"episode={self.episode_id}"
        )

    def on_joint(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        gripper_name = "gripper" if "gripper" in values else "gripper_width"
        if not all(name in values for name in names) or gripper_name not in values:
            return
        self.joint = np.asarray([*(values[name] for name in names), values[gripper_name]], dtype=np.float32)

    def on_color(self, message: Image) -> None:
        try:
            self.color = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            self.color_stamp_ns = stamp_ns(message)
        except Exception as exc:
            self.get_logger().warning(f"color conversion failed: {exc}")

    def on_depth(self, message: Image) -> None:
        try:
            self.depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            self.depth_stamp_ns = stamp_ns(message)
        except Exception as exc:
            self.get_logger().warning(f"depth conversion failed: {exc}")

    def on_camera_info(self, message: CameraInfo) -> None:
        if len(message.k) >= 9 and message.k[0] > 0 and message.k[4] > 0:
            self.depth_intrinsics = (
                float(message.k[0]), float(message.k[4]),
                float(message.k[2]), float(message.k[5]),
            )

    def tick(self) -> None:
        now = time.monotonic()
        result, error = self.worker.snapshot()
        if error is not None:
            self.publish_status({"phase": "rpc_error", "error": repr(error)})
        if result is not None and result.sequence_id > self.last_result_sequence:
            self.last_result_sequence = result.sequence_id
            if result.action_chunk is not None:
                self.action_pub.publish(String(data=json.dumps({
                    "protocol_version": runtime.PROTOCOL_VERSION,
                    "episode_id": self.episode_id,
                    "sequence_id": result.sequence_id,
                    "capture_timestamp_ns": result.capture_timestamp_ns,
                    "received_monotonic_ns": time.monotonic_ns(),
                    "model_version": self.contract.model_version,
                    "action_chunk": result.action_chunk.tolist(),
                    "action_min": self.contract.stats.action_min.tolist(),
                    "action_max": self.contract.stats.action_max.tolist(),
                    "state_min": self.contract.stats.state_min.tolist(),
                    "state_max": self.contract.stats.state_max.tolist(),
                    "gripper_action_mode": self.contract.stats.gripper_action_mode,
                }, ensure_ascii=False, separators=(",", ":"))))

        max_gap = float(self.get_parameter("max_sensor_gap_ms").value) / 1000.0
        if self.joint is None or self.color is None or self.depth is None or self.depth_intrinsics is None:
            return
        if abs(self.color_stamp_ns - self.depth_stamp_ns) / 1e9 > max_gap:
            self.publish_status({"phase": "sensor_wait", "reason": "rgb_depth_gap"})
            return
        if now - self.last_submit_monotonic < 1.0 / float(self.get_parameter("fps").value):
            return
        try:
            frame = {
                "color": self.color,
                "depth": self.depth,
                "depth_intrinsics": self.depth_intrinsics,
                "depth_aligned_to_color": False,
            }
            image, point_cloud = runtime.preprocess_camera_frame(frame)
            self.sequence_id += 1
            request = runtime.make_request(
                self.episode_id, self.sequence_id, time.time_ns(),
                self.joint, image, point_cloud,
            )
            self.worker.submit(request)
            self.last_submit_monotonic = now
            self.publish_status({"phase": "submitted", "sequence_id": self.sequence_id})
        except Exception as exc:
            self.publish_status({"phase": "preprocess_error", "error": repr(exc)})

    def publish_status(self, value: dict) -> None:
        self.status_pub.publish(String(data=json.dumps(value, ensure_ascii=False)))

    def close(self) -> None:
        self.worker.stop()
        try:
            self.client.reset_episode(self.episode_id, 1.0)
        except Exception:
            pass
        self.client.close()


def runtime_name() -> str:
    return "piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42"


def stamp_ns(message: Image | JointState) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = ObservationInferenceNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
