"""Hardware, safety, preprocessing, and gRPC helpers for the local bridge."""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import grpc
import numpy as np

from server import policy_pb2, policy_pb2_grpc


PROTOCOL_VERSION = "rl100-dp3-v1"
IMAGE_SHAPE = (3, 84, 84)
POINT_CLOUD_SHAPE = (512, 3)
AGENT_POS_SHAPE = (7,)
ACTION_SHAPE = (7,)
RAW_TO_RAD = math.pi / 180.0 / 1000.0
RAD_TO_RAW = 180.0 / math.pi * 1000.0
GRIPPER_RAW_PER_M = 1_000_000.0
GRIPPER_LOWER_M = 0.0
GRIPPER_UPPER_M = 0.07
GRIPPER_EFFORT = 1000
PIPER_HARD_LOWER_RAD = np.deg2rad(
    np.array([-150.0, -1.0, -170.0, -100.0, -70.0, -120.0], dtype=np.float32)
)
PIPER_HARD_UPPER_RAD = np.deg2rad(
    np.array([150.0, 180.0, 1.0, 100.0, 70.0, 120.0], dtype=np.float32)
)


def _f32_bytes(array: np.ndarray, shape: tuple[int, ...], name: str) -> bytes:
    value = np.asarray(array, dtype="<f4")
    if value.shape != shape:
        raise ValueError(f"{name} shape应为{shape}，实际为{value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}包含NaN/Inf")
    return np.ascontiguousarray(value).tobytes()


def decode_f32(payload: bytes, shape: tuple[int, ...], name: str) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{name}字节数应为{expected_bytes}，实际为{len(payload)}"
        )
    value = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}包含NaN/Inf")
    return value


def make_request(
    episode_id: str,
    sequence_id: int,
    capture_timestamp_ns: int,
    agent_pos: np.ndarray,
    image: np.ndarray,
    point_cloud: np.ndarray,
) -> policy_pb2.InferenceRequest:
    return policy_pb2.InferenceRequest(
        protocol_version=PROTOCOL_VERSION,
        episode_id=episode_id,
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        agent_pos_f32=_f32_bytes(agent_pos, AGENT_POS_SHAPE, "agent_pos"),
        image_f32=_f32_bytes(image, IMAGE_SHAPE, "image"),
        point_cloud_f32=_f32_bytes(
            point_cloud, POINT_CLOUD_SHAPE, "point_cloud"
        ),
    )


@dataclass(frozen=True)
class SafetyStats:
    state_min: np.ndarray
    state_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    point_cloud_low: np.ndarray
    point_cloud_high: np.ndarray
    gripper_action_mode: str
    action_key: str

    @classmethod
    def from_server_info(cls, info: policy_pb2.ServerInfoResponse) -> "SafetyStats":
        def array(values, shape, name):
            result = np.asarray(values, dtype=np.float32)
            if result.shape != shape or not np.all(np.isfinite(result)):
                raise RuntimeError(
                    f"服务器{name}无效，应为{shape}有限数值，实际{result.shape}"
                )
            return result

        if info.gripper_action_mode not in ("command", "width"):
            raise RuntimeError(
                f"不支持的gripper_action_mode={info.gripper_action_mode!r}"
            )
        if info.action_key != "policy_action":
            raise RuntimeError(
                f"服务器action_key={info.action_key!r}，期望'policy_action'"
            )
        return cls(
            state_min=array(info.state_min, (7,), "state_min"),
            state_max=array(info.state_max, (7,), "state_max"),
            action_min=array(info.action_min, (7,), "action_min"),
            action_max=array(info.action_max, (7,), "action_max"),
            point_cloud_low=array(info.point_cloud_low, (3,), "point_cloud_low"),
            point_cloud_high=array(info.point_cloud_high, (3,), "point_cloud_high"),
            gripper_action_mode=info.gripper_action_mode,
            action_key=info.action_key,
        )


@dataclass(frozen=True)
class ServerContract:
    model_version: str
    output_dir: str
    policy_subdir: str
    n_obs_steps: int
    n_action_steps: int
    stats: SafetyStats

    @classmethod
    def from_info(cls, info: policy_pb2.ServerInfoResponse) -> "ServerContract":
        if info.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"协议不一致: server={info.protocol_version!r}, "
                f"local={PROTOCOL_VERSION!r}"
            )
        expected_shapes = {
            "image": (tuple(info.image_shape), IMAGE_SHAPE),
            "point_cloud": (tuple(info.point_cloud_shape), POINT_CLOUD_SHAPE),
            "agent_pos": (tuple(info.agent_pos_shape), AGENT_POS_SHAPE),
            "action": (tuple(info.action_shape), ACTION_SHAPE),
        }
        for name, (actual, expected) in expected_shapes.items():
            if actual != expected:
                raise RuntimeError(
                    f"服务器{name} shape={actual}，本地期望{expected}"
                )
        if int(info.n_obs_steps) != 3 or int(info.n_action_steps) != 4:
            raise RuntimeError(
                "服务器模型时序不匹配: "
                f"n_obs_steps={info.n_obs_steps}, "
                f"n_action_steps={info.n_action_steps}"
            )
        return cls(
            model_version=info.model_version,
            output_dir=info.output_dir,
            policy_subdir=info.policy_subdir,
            n_obs_steps=int(info.n_obs_steps),
            n_action_steps=int(info.n_action_steps),
            stats=SafetyStats.from_server_info(info),
        )

    def validate_model_identity(
        self, expected_output_name: str, expected_policy_subdir: str
    ) -> None:
        if Path(self.output_dir).name != expected_output_name:
            raise RuntimeError(
                "服务器权重目录不匹配: "
                f"server={self.output_dir!r}, expected={expected_output_name!r}"
            )
        if self.policy_subdir != expected_policy_subdir:
            raise RuntimeError(
                "服务器权重子目录不匹配: "
                f"server={self.policy_subdir!r}, "
                f"expected={expected_policy_subdir!r}"
            )


class PolicyClient:
    def __init__(
        self,
        address: str,
        connect_timeout: float,
        max_message_mb: int = 8,
    ):
        limit = int(max_message_mb) * 1024 * 1024
        self.channel = grpc.insecure_channel(
            address,
            options=(
                ("grpc.max_send_message_length", limit),
                ("grpc.max_receive_message_length", limit),
            ),
        )
        grpc.channel_ready_future(self.channel).result(timeout=connect_timeout)
        self.stub = policy_pb2_grpc.PolicyServiceStub(self.channel)

    def get_contract(self, timeout: float) -> ServerContract:
        info = self.stub.GetServerInfo(
            policy_pb2.ServerInfoRequest(protocol_version=PROTOCOL_VERSION),
            timeout=timeout,
        )
        return ServerContract.from_info(info)

    def reset_episode(self, episode_id: str, timeout: float) -> None:
        response = self.stub.ResetEpisode(
            policy_pb2.ResetEpisodeRequest(
                protocol_version=PROTOCOL_VERSION,
                episode_id=episode_id,
            ),
            timeout=timeout,
        )
        if response.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("ResetEpisode响应协议不一致")
        if response.episode_id != episode_id or not response.reset:
            raise RuntimeError("服务器未确认episode reset")

    def infer(
        self, request: policy_pb2.InferenceRequest, timeout: float
    ) -> policy_pb2.InferenceResponse:
        return self.stub.Infer(request, timeout=timeout)

    def close(self) -> None:
        self.channel.close()


@dataclass(frozen=True)
class PolicyResult:
    sequence_id: int
    capture_timestamp_ns: int
    submitted_monotonic: float
    completed_monotonic: float
    action_chunk: np.ndarray | None
    inference_time_ms: float
    status_message: str


class AsyncPolicyClient:
    """Send only the newest pending frame without blocking the control loop."""

    def __init__(self, client: PolicyClient, rpc_timeout: float, action_steps: int):
        self.client = client
        self.rpc_timeout = float(rpc_timeout)
        self.action_steps = int(action_steps)
        self.condition = threading.Condition()
        self.pending: tuple[policy_pb2.InferenceRequest, float] | None = None
        self.latest: PolicyResult | None = None
        self.error: BaseException | None = None
        self.stop_requested = False
        self.thread = threading.Thread(
            target=self._loop, name="remote-policy-rpc", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, request: policy_pb2.InferenceRequest) -> None:
        with self.condition:
            self.pending = (request, time.monotonic())
            self.condition.notify_all()

    def snapshot(self) -> tuple[PolicyResult | None, BaseException | None]:
        with self.condition:
            return self.latest, self.error

    def _loop(self) -> None:
        while True:
            with self.condition:
                while self.pending is None and not self.stop_requested:
                    self.condition.wait()
                if self.stop_requested:
                    return
                request, submitted = self.pending
                self.pending = None
            try:
                response = self.client.infer(request, self.rpc_timeout)
                if response.protocol_version != PROTOCOL_VERSION:
                    raise RuntimeError("Infer响应协议版本不一致")
                if response.episode_id != request.episode_id:
                    raise RuntimeError("Infer响应episode_id不一致")
                if int(response.sequence_id) != int(request.sequence_id):
                    raise RuntimeError("Infer响应sequence_id不一致")
                if int(response.capture_timestamp_ns) != int(
                    request.capture_timestamp_ns
                ):
                    raise RuntimeError("Infer响应capture_timestamp不一致")
                if response.ready:
                    shape = (
                        int(response.action_steps),
                        int(response.action_dims),
                    )
                    expected = (self.action_steps, 7)
                    if shape != expected:
                        raise RuntimeError(
                            f"服务器action chunk shape={shape}，期望{expected}"
                        )
                    chunk = decode_f32(
                        response.action_chunk_f32, expected, "action_chunk"
                    )
                else:
                    chunk = None
                result = PolicyResult(
                    sequence_id=int(response.sequence_id),
                    capture_timestamp_ns=int(response.capture_timestamp_ns),
                    submitted_monotonic=submitted,
                    completed_monotonic=time.monotonic(),
                    action_chunk=chunk,
                    inference_time_ms=float(response.inference_time_ms),
                    status_message=response.status_message,
                )
                with self.condition:
                    self.latest = result
                    self.error = None
            except BaseException as exc:
                with self.condition:
                    self.error = exc

    def stop(self) -> None:
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


def resize_rgb_to_chw(rgb: np.ndarray, size: int = 84) -> np.ndarray:
    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"D435i RGB shape异常: {rgb.shape}")
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)


def depth_to_point_cloud(
    depth_raw: np.ndarray,
    intrinsics: tuple[float, ...],
    num_points: int = 512,
) -> np.ndarray:
    if depth_raw.shape != (480, 640):
        raise RuntimeError(f"D435i depth shape异常: {depth_raw.shape}")
    depth_m = depth_raw.astype(np.float32) * 0.001
    valid = (
        (depth_raw > 0)
        & (depth_raw < 65535)
        & (depth_m >= 0.1)
        & (depth_m <= 2.0)
    )
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise RuntimeError("实时深度没有0.1~2.0m内的有效点")
    fx, fy, cx, cy = map(float, intrinsics)
    z = depth_m[v, u]
    points = np.stack(
        (
            (u.astype(np.float32) - cx) * z / fx,
            (v.astype(np.float32) - cy) * z / fy,
            z,
        ),
        axis=1,
    )
    if len(points) >= num_points:
        indices = np.linspace(0, len(points) - 1, num_points, dtype=np.int64)
        return points[indices].astype(np.float32)
    repeats = math.ceil(num_points / len(points))
    return np.concatenate([points] * repeats, axis=0)[:num_points].astype(
        np.float32
    )


def preprocess_camera_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    color_bgr = np.asarray(frame["color"])
    rgb = cv2.cvtColor(color_bgr[..., :3], cv2.COLOR_BGR2RGB)
    image = resize_rgb_to_chw(rgb)
    if frame.get("depth_aligned_to_color", True):
        raise RuntimeError("必须使用未对齐depth: align_depth_to_color=False")
    point_cloud = depth_to_point_cloud(
        np.asarray(frame["depth"]), tuple(frame["depth_intrinsics"]), 512
    )
    return image, point_cloud


def point_cloud_outlier_fraction(
    point_cloud: np.ndarray, stats: SafetyStats
) -> float:
    outside = np.any(
        (point_cloud < stats.point_cloud_low[None])
        | (point_cloud > stats.point_cloud_high[None]),
        axis=1,
    )
    return float(outside.mean())


def safe_policy_target(
    predicted: np.ndarray,
    current: np.ndarray,
    stats: SafetyStats,
    args: Any,
) -> tuple[np.ndarray, list[str]]:
    predicted = np.asarray(predicted, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if predicted.shape != (7,) or current.shape != (7,):
        raise RuntimeError("策略目标和反馈都必须是7维")
    lower = np.maximum(
        PIPER_HARD_LOWER_RAD,
        stats.action_min[:6] - float(args.dataset_margin_rad),
    )
    upper = np.minimum(
        PIPER_HARD_UPPER_RAD,
        stats.action_max[:6] + float(args.dataset_margin_rad),
    )
    warnings: list[str] = []
    if not args.skip_current_joint_range_check and (
        np.any(current[:6] < lower) or np.any(current[:6] > upper)
    ):
        raise RuntimeError("当前关节姿态超出训练分布安全范围")
    if args.skip_current_joint_range_check:
        warnings.append("已跳过当前关节训练范围检查")
    target = predicted.copy()
    if stats.gripper_action_mode == "command":
        command = float(target[6])
        target[6] = (
            GRIPPER_UPPER_M
            if command >= float(args.gripper_command_threshold)
            else GRIPPER_LOWER_M
        )
        warnings.append(f"夹爪命令{command:.3f}已解码")
    if np.any(target[:6] < lower) or np.any(target[:6] > upper):
        if not args.clip_actions:
            raise RuntimeError("策略关节目标超出训练范围")
        target[:6] = np.clip(target[:6], lower, upper)
        warnings.append("关节目标已裁剪到训练范围")
    if not args.skip_gripper_safety_check:
        if stats.gripper_action_mode == "command":
            grip_lower = max(
                GRIPPER_LOWER_M,
                float(stats.state_min[6]) - float(args.gripper_margin_m),
            )
            grip_upper = min(
                GRIPPER_UPPER_M,
                float(stats.state_max[6]) + float(args.gripper_margin_m),
            )
        else:
            grip_lower = max(
                GRIPPER_LOWER_M,
                float(stats.action_min[6]) - float(args.gripper_margin_m),
            )
            grip_upper = min(
                GRIPPER_UPPER_M,
                float(stats.action_max[6]) + float(args.gripper_margin_m),
            )
        if current[6] < grip_lower - 1e-6 or current[6] > grip_upper + 1e-6:
            raise RuntimeError("当前夹爪宽度超出训练分布")
        if target[6] < grip_lower - 1e-6 or target[6] > grip_upper + 1e-6:
            if not args.clip_actions:
                raise RuntimeError("策略夹爪目标超出训练范围")
            target[6] = np.clip(target[6], grip_lower, grip_upper)
            warnings.append("夹爪目标已裁剪到训练范围")
    else:
        warnings.append("已跳过夹爪范围检查")
    if not np.all(np.isfinite(target)):
        raise RuntimeError("安全过滤后的动作包含NaN/Inf")
    return target.astype(np.float32), warnings


class JointTrajectoryPlanner:
    def __init__(
        self,
        initial: np.ndarray,
        joint_speed: float,
        joint_accel: float,
        gripper_speed: float,
        tracking_error: float,
    ):
        self.position = np.asarray(initial, dtype=np.float64).copy()
        self.velocity = np.zeros(6, dtype=np.float64)
        self.joint_speed = float(joint_speed)
        self.joint_accel = float(joint_accel)
        self.gripper_speed = float(gripper_speed)
        self.tracking_error = float(tracking_error)

    def hold(self, measured: np.ndarray, dt: float) -> np.ndarray:
        return self.step(np.asarray(measured), measured, dt)

    def step(
        self, target: np.ndarray, measured: np.ndarray, dt: float
    ) -> np.ndarray:
        dt = float(np.clip(dt, 1e-4, 0.1))
        target = np.asarray(target, dtype=np.float64)
        measured = np.asarray(measured, dtype=np.float64)
        tracking_delta = self.position[:6] - measured[:6]
        bad = np.abs(tracking_delta) > self.tracking_error
        if np.any(bad):
            self.position[:6][bad] = measured[:6][bad] + np.clip(
                tracking_delta[bad], -self.tracking_error, self.tracking_error
            )
            self.velocity[bad] = 0.0
        error = target[:6] - self.position[:6]
        braking_speed = np.sqrt(2.0 * self.joint_accel * np.abs(error))
        desired_velocity = np.sign(error) * np.minimum(
            self.joint_speed, braking_speed
        )
        max_dv = self.joint_accel * dt
        self.velocity += np.clip(
            desired_velocity - self.velocity, -max_dv, max_dv
        )
        self.velocity = np.clip(
            self.velocity, -self.joint_speed, self.joint_speed
        )
        delta = self.velocity * dt
        overshoot = (np.sign(delta) == np.sign(error)) & (
            np.abs(delta) >= np.abs(error)
        )
        delta[overshoot] = error[overshoot]
        self.velocity[overshoot] = 0.0
        self.position[:6] += delta
        grip_delta = float(target[6] - self.position[6])
        self.position[6] += float(
            np.clip(
                grip_delta,
                -self.gripper_speed * dt,
                self.gripper_speed * dt,
            )
        )
        return self.position.astype(np.float32).copy()


class LatestFrameCamera:
    def __init__(self, camera: Any):
        self.camera = camera
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.camera.start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.get_frame(timeout=5.0)

    def _loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                frame = self.camera.get_frame(require_pc=False)
                with self.condition:
                    self.latest = frame
                    self.condition.notify_all()
        except BaseException as exc:
            if not self.stop_event.is_set():
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()

    def get_frame(self, timeout: float = 1.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest is None and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("等待RealSense帧超时")
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError("RealSense后台采集失败") from self.error
            assert self.latest is not None
            return self.latest

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.camera.stop()
        finally:
            if self.thread is not None:
                self.thread.join(timeout=2.0)


def import_piper_sdk(sdk_root: Path | None):
    if sdk_root is not None:
        resolved = sdk_root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Piper SDK目录不存在: {resolved}")
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    from piper_sdk import C_PiperInterface_V2

    return C_PiperInterface_V2


class PiperStateReader:
    def __init__(self, piper: Any):
        self.piper = piper
        self.last_gripper_status: dict[str, bool] = {}

    def read(self) -> tuple[np.ndarray, float, float]:
        arm_msg = self.piper.GetArmJointMsgs()
        joints = arm_msg.joint_state
        raw = np.array(
            [
                joints.joint_1,
                joints.joint_2,
                joints.joint_3,
                joints.joint_4,
                joints.joint_5,
                joints.joint_6,
            ],
            dtype=np.float64,
        )
        joint_rad = (raw * RAW_TO_RAD).astype(np.float32)
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper = gripper_msg.gripper_state
        width = float(gripper.grippers_angle) / GRIPPER_RAW_PER_M
        foc = gripper.foc_status
        self.last_gripper_status = {
            name: bool(getattr(foc, name))
            for name in (
                "voltage_too_low",
                "motor_overheating",
                "driver_overcurrent",
                "driver_overheating",
                "sensor_status",
                "driver_error_status",
                "driver_enable_status",
                "homing_status",
            )
        }
        faults = [
            name
            for name in (
                "voltage_too_low",
                "motor_overheating",
                "driver_overcurrent",
                "driver_overheating",
                "sensor_status",
                "driver_error_status",
            )
            if self.last_gripper_status[name]
        ]
        state = np.r_[joint_rad, np.float32(width)].astype(np.float32)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Piper反馈包含NaN/Inf")
        if faults:
            raise RuntimeError(f"Piper夹爪反馈故障: {faults}")
        return (
            state,
            float(getattr(arm_msg, "time_stamp", 0.0)),
            float(getattr(gripper_msg, "time_stamp", 0.0)),
        )


def _enum_value(value: Any) -> int:
    return int(getattr(value, "value", value))


def enable_robot(piper: Any, speed_percent: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    feedback = piper.GetArmStatus()
    status = feedback.arm_status
    if _enum_value(status.arm_status) == 0x01:
        raise RuntimeError("Piper处于EMERGENCY_STOP，拒绝使能")
    if _enum_value(status.arm_status) not in (0x00, 0x0B):
        raise RuntimeError("Piper状态异常，拒绝使能")
    if _enum_value(status.ctrl_mode) == 0x02 or _enum_value(
        status.teach_status
    ) == 0x01:
        piper.MotionCtrl_1(0x00, 0x00, 0x02)
        time.sleep(0.1)
    while time.monotonic() < deadline:
        enabled = bool(piper.EnablePiper())
        piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
        time.sleep(0.02)
        status = piper.GetArmStatus().arm_status
        if _enum_value(status.arm_status) == 0x01:
            raise RuntimeError("Piper使能期间进入EMERGENCY_STOP")
        if (
            enabled
            and _enum_value(status.ctrl_mode) == 0x01
            and _enum_value(status.mode_feed) == 0x01
        ):
            return
    raise RuntimeError("Piper未在超时时间内进入CAN/MOVE J控制模式")


def enable_gripper(
    piper: Any,
    reader: PiperStateReader,
    hold_width_m: float,
    require_homing: bool,
    timeout: float = 3.0,
) -> None:
    hold_raw = int(round(float(hold_width_m) * GRIPPER_RAW_PER_M))
    reader.read()
    status = reader.last_gripper_status
    if status.get("driver_enable_status") and (
        status.get("homing_status") or not require_homing
    ):
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x01, 0)
        return
    if not status.get("driver_enable_status"):
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x02, 0)
        time.sleep(0.1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        piper.GripperCtrl(hold_raw, GRIPPER_EFFORT, 0x01, 0)
        time.sleep(0.02)
        reader.read()
        status = reader.last_gripper_status
        if status.get("driver_enable_status") and (
            status.get("homing_status") or not require_homing
        ):
            return
    raise RuntimeError("Piper夹爪未在超时时间内使能")


def send_action(piper: Any, action: np.ndarray, speed_percent: int) -> None:
    joints_raw = np.rint(action[:6] * RAD_TO_RAW).astype(np.int64)
    gripper_raw = int(round(float(action[6]) * GRIPPER_RAW_PER_M))
    piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
    piper.JointCtrl(*[int(value) for value in joints_raw])
    piper.GripperCtrl(gripper_raw, GRIPPER_EFFORT, 0x01, 0)


class OperatorConsole:
    def __init__(self):
        self.commands: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            self.commands.put(line.strip().lower() or "stop")

    def poll(self) -> str | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None


def offline_smoke() -> None:
    request = make_request(
        "offline",
        1,
        123,
        np.zeros(AGENT_POS_SHAPE, dtype=np.float32),
        np.zeros(IMAGE_SHAPE, dtype=np.float32),
        np.zeros(POINT_CLOUD_SHAPE, dtype=np.float32),
    )
    assert len(request.agent_pos_f32) == 7 * 4
    chunk = np.zeros((4, 7), dtype=np.float32)
    decoded = decode_f32(chunk.tobytes(), (4, 7), "action_chunk")
    planner = JointTrajectoryPlanner(
        np.zeros(7, dtype=np.float32), 0.2, 0.5, 0.02, 0.1
    )
    planned = planner.step(np.ones(7, dtype=np.float32) * 0.01, np.zeros(7), 0.01)
    if decoded.shape != (4, 7) or planned.shape != (7,):
        raise RuntimeError("offline smoke shape失败")
    print("[offline-smoke] 协议序列化、动作解码和轨迹规划通过", flush=True)
