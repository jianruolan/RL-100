#!/usr/bin/env python3
"""将 Piper pick-and-place ROS2 bag 转成 RL-100 ReplayBuffer Zarr。

本脚本不依赖 ROS2；它直接读取 rosbag2 的 SQLite3 数据库，并只实现本数据集
使用到的标准 CDR 消息解码。转换定义如下：

* state  = [六个关节反馈(rad), 夹爪总开口宽度(m)]
* action = [六个实际下发关节目标(rad), 夹爪 position(m)]
* policy_action = [六个实际下发关节目标(rad), 夹爪期望开合状态]
* gripper_command_state = Pico 开合命令锁存后的期望状态（1=张开，0=闭合）
* gripper_open/close_event = 相邻保留 RGB 帧之间是否出现对应命令
* 采样时间轴采用 RGB bag timestamp，其他 topic 做最近邻/线性插值。
* 可选地仅保留 controlAccepted=true 的任务控制帧；被删除区间和采样时间跳变
  会写入 sequence_break，供训练采样器禁止窗口跨越。
* 点云由深度图和深度相机内参反投影，坐标系是深度相机 optical frame。
* 仅转换元数据明确标记为成功且 rosbag 完整的 episode。
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import math
import shutil
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import zarr


REQUIRED_TOPICS = {
    "/joint_states",
    "/piper_teleop/status",
    "/piper_teleop/gripper/position",
    "/piper_teleop/pico_frame",
    "/piper_camera/d435i/color/image_raw",
    "/piper_camera/d435i/depth/image_rect_raw",
    "/piper_camera/d435i/depth/camera_info",
}
IMAGENET_MEAN_UINT8 = np.array([123, 116, 104], dtype=np.uint8)
MDEG_TO_RAD = 0.001 * math.pi / 180.0


class CdrReader:
    """ROS2 little-endian CDR 的最小读取器。

    CDR 对齐从 4 字节 encapsulation header 后开始计算，而不是从 blob 的第 0
    字节开始。当前数据的 header 为 ``00 01 00 00``（little endian）。
    """

    def __init__(self, blob: bytes):
        if len(blob) < 4 or blob[:2] != b"\x00\x01":
            raise ValueError("only little-endian CDR is supported")
        self.data = memoryview(blob)
        self.base = 4
        self.pos = 4

    def align(self, size: int) -> None:
        rel = self.pos - self.base
        self.pos += (-rel) % size

    def unpack(self, fmt: str, align: int):
        self.align(align)
        value = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return value[0] if len(value) == 1 else value

    def u8(self) -> int:
        return self.unpack("B", 1)

    def u32(self) -> int:
        return self.unpack("I", 4)

    def f64(self) -> float:
        return self.unpack("d", 8)

    def string(self) -> str:
        n = self.u32()
        if n == 0:
            return ""
        raw = bytes(self.data[self.pos : self.pos + n])
        self.pos += n
        return raw[:-1].decode("utf-8") if raw.endswith(b"\0") else raw.decode("utf-8")

    def bytes(self) -> memoryview:
        n = self.u32()
        value = self.data[self.pos : self.pos + n]
        self.pos += n
        return value

    def float64_sequence(self) -> np.ndarray:
        n = self.u32()
        self.align(8)
        out = np.frombuffer(self.data, dtype="<f8", count=n, offset=self.pos).copy()
        self.pos += n * 8
        return out


def read_header(r: CdrReader) -> tuple[int, str]:
    sec = r.u32()
    nanosec = r.u32()
    frame_id = r.string()
    return sec * 1_000_000_000 + nanosec, frame_id


def decode_std_string(blob: bytes) -> str:
    return CdrReader(blob).string()


def decode_float64(blob: bytes) -> float:
    return CdrReader(blob).f64()


def decode_joint_state(blob: bytes) -> np.ndarray:
    r = CdrReader(blob)
    read_header(r)
    n_names = r.u32()
    names = [r.string() for _ in range(n_names)]
    positions = r.float64_sequence()
    # 数据文档保证顺序固定；仍检查名称，避免错误数据静默进入训练。
    expected = [f"joint{i}" for i in range(1, 9)]
    if names[:8] != expected or len(positions) < 8:
        raise ValueError(f"unexpected JointState: names={names}, positions={positions.shape}")
    return positions[:8]


def decode_image(blob: bytes) -> tuple[np.ndarray, str, str, int]:
    r = CdrReader(blob)
    header_ns, frame_id = read_header(r)
    height, width = r.u32(), r.u32()
    encoding = r.string()
    is_bigendian = bool(r.u8())
    step = r.u32()
    raw = r.bytes()
    if encoding == "rgb8":
        row_bytes = width * 3
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
        image = arr.reshape(height, width, 3).copy()
    elif encoding in ("16UC1", "mono16"):
        dtype = ">u2" if is_bigendian else "<u2"
        values_per_row = step // 2
        arr = np.frombuffer(raw, dtype=dtype).reshape(height, values_per_row)[:, :width]
        image = arr.astype(np.uint16, copy=True)
    else:
        raise ValueError(f"unsupported image encoding: {encoding}")
    return image, encoding, frame_id, header_ns


def decode_camera_info(blob: bytes) -> dict[str, Any]:
    r = CdrReader(blob)
    header_ns, frame_id = read_header(r)
    height, width = r.u32(), r.u32()
    distortion_model = r.string()
    distortion = r.float64_sequence()
    k = np.array([r.f64() for _ in range(9)], dtype=np.float64).reshape(3, 3)
    rotation = np.array([r.f64() for _ in range(9)], dtype=np.float64).reshape(3, 3)
    projection = np.array([r.f64() for _ in range(12)], dtype=np.float64).reshape(3, 4)
    return {
        "header_ns": header_ns,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "distortion_model": distortion_model,
        "distortion": distortion,
        "k": k,
        "rotation": rotation,
        "projection": projection,
    }


def resize_rgb_to_chw(image_rgb: np.ndarray, size: int, mode: str) -> np.ndarray:
    """输入已经是 RGB；禁止像旧 HDF5 脚本那样再次交换 R/B 通道。"""
    if mode == "stretch":
        out = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    elif mode == "letterbox":
        height, width = image_rgb.shape[:2]
        scale = min(size / width, size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=interpolation)
        out = np.empty((size, size, 3), dtype=np.uint8)
        out[...] = IMAGENET_MEAN_UINT8
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        out[top : top + resized_height, left : left + resized_width] = resized
    else:
        raise ValueError(mode)
    return np.ascontiguousarray(np.transpose(out, (2, 0, 1)))


def resize_depth_to_chw_uint8(
    depth_raw: np.ndarray,
    size: int,
    mode: str,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """把16UC1深度按固定物理范围量化为RGB-D的第4个uint8通道。"""
    depth_m = depth_raw.astype(np.float32) * 0.001
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    normalized = np.zeros_like(depth_m, dtype=np.float32)
    normalized[valid] = (depth_m[valid] - min_depth_m) / (max_depth_m - min_depth_m)
    depth_u8 = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mode == "stretch":
        out = cv2.resize(depth_u8, (size, size), interpolation=cv2.INTER_NEAREST)
    elif mode == "letterbox":
        height, width = depth_u8.shape
        scale = min(size / width, size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(depth_u8, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        out = np.zeros((size, size), dtype=np.uint8)
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        out[top : top + resized_height, left : left + resized_width] = resized
    else:
        raise ValueError(mode)
    return out[None]


def depth_to_point_cloud(
    depth_raw: np.ndarray,
    k: np.ndarray,
    num_points: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """在深度相机 optical frame 下反投影，并作确定性均匀下采样。"""
    depth_m = depth_raw.astype(np.float32) * 0.001
    valid = (
        (depth_raw > 0)
        & (depth_raw < 65535)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    z = depth_m[v, u]
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    points = np.stack((x, y, z), axis=1).astype(np.float32)
    if len(points) >= num_points:
        indices = np.linspace(0, len(points) - 1, num_points, dtype=np.int64)
        return points[indices]
    repeats = math.ceil(num_points / len(points))
    return np.concatenate([points] * repeats, axis=0)[:num_points]


def nearest_index(times: np.ndarray, target: int) -> int:
    i = bisect.bisect_left(times, target)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i - 1 if target - int(times[i - 1]) <= int(times[i]) - target else i


def interpolate(times: np.ndarray, values: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    """线性插值连续状态，并返回到最近原始样本的绝对时间差。"""
    i = bisect.bisect_left(times, target)
    nearest = nearest_index(times, target)
    error_ns = abs(int(times[nearest]) - target)
    if i <= 0:
        return values[0].copy(), error_ns
    if i >= len(times):
        return values[-1].copy(), error_ns
    t0, t1 = int(times[i - 1]), int(times[i])
    alpha = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
    return ((1.0 - alpha) * values[i - 1] + alpha * values[i]).astype(np.float64), error_ns


def episode_is_success(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if metadata.get("status") != "complete":
        reasons.append(f"status={metadata.get('status')!r}")
    if metadata.get("outcome") != "success":
        reasons.append(f"outcome={metadata.get('outcome')!r}")
    if metadata.get("rosbag_return_code") != 0:
        reasons.append(f"rosbag_return_code={metadata.get('rosbag_return_code')!r}")
    if metadata.get("missing_topics_at_start"):
        reasons.append("missing_topics_at_start is not empty")
    return not reasons, reasons


@dataclass
class BagIndex:
    connection: sqlite3.Connection
    topics: dict[str, tuple[int, str]]

    @classmethod
    def open(cls, path: Path) -> "BagIndex":
        connection = sqlite3.connect(str(path))
        topics = {
            name: (topic_id, msg_type)
            for topic_id, name, msg_type in connection.execute("SELECT id,name,type FROM topics")
        }
        missing = sorted(REQUIRED_TOPICS - topics.keys())
        if missing:
            connection.close()
            raise RuntimeError(f"{path}: missing required topics: {missing}")
        return cls(connection, topics)

    def rows(self, topic: str) -> list[tuple[int, int, bytes]]:
        topic_id = self.topics[topic][0]
        return list(
            self.connection.execute(
                "SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)
            )
        )

    def message_index(self, topic: str) -> tuple[np.ndarray, np.ndarray]:
        topic_id = self.topics[topic][0]
        rows = list(
            self.connection.execute(
                "SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)
            )
        )
        return (
            np.asarray([row[1] for row in rows], dtype=np.int64),
            np.asarray([row[0] for row in rows], dtype=np.int64),
        )

    def image_message_index(self, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 bag 时间、相机 Header 时间和 message id，不读取整张图像。"""
        topic_id = self.topics[topic][0]
        rows = list(
            self.connection.execute(
                "SELECT id,timestamp,substr(data,1,128) FROM messages "
                "WHERE topic_id=? ORDER BY timestamp",
                (topic_id,),
            )
        )
        header_times = []
        for _, _, prefix in rows:
            r = CdrReader(prefix)
            header_ns, _ = read_header(r)
            header_times.append(header_ns)
        return (
            np.asarray([row[1] for row in rows], dtype=np.int64),
            np.asarray(header_times, dtype=np.int64),
            np.asarray([row[0] for row in rows], dtype=np.int64),
        )

    def blob(self, message_id: int) -> bytes:
        row = self.connection.execute("SELECT data FROM messages WHERE id=?", (int(message_id),)).fetchone()
        if row is None:
            raise KeyError(message_id)
        return row[0]


def load_numeric_stream(bag: BagIndex, topic: str, decoder) -> tuple[np.ndarray, np.ndarray]:
    rows = bag.rows(topic)
    times = np.asarray([r[1] for r in rows], dtype=np.int64)
    values = np.stack([decoder(r[2]) for r in rows], axis=0)
    return times, values


def load_status_stream(bag: BagIndex) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = bag.rows("/piper_teleop/status")
    times = np.asarray([r[1] for r in rows], dtype=np.int64)
    values = [json.loads(decode_std_string(r[2])) for r in rows]
    return times, values


def load_pico_command_stream(bag: BagIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 Pico 帧里的夹爪开合命令。

    该 topic 是 ``std_msgs/String`` 包裹的 JSON。命令字段位于 JSON 顶层，不能
    从 ``controllerInput`` 中猜测按键映射。
    """
    rows = bag.rows("/piper_teleop/pico_frame")
    times = np.asarray([row[1] for row in rows], dtype=np.int64)
    open_commands, close_commands = [], []
    for _, _, blob in rows:
        value = json.loads(decode_std_string(blob))
        if "gripperOpen" not in value or "gripperClose" not in value:
            raise RuntimeError("Pico JSON missing gripperOpen/gripperClose")
        open_commands.append(bool(value["gripperOpen"]))
        close_commands.append(bool(value["gripperClose"]))
    return (
        times,
        np.asarray(open_commands, dtype=bool),
        np.asarray(close_commands, dtype=bool),
    )


def align_gripper_commands(
    pico_times: np.ndarray,
    open_commands: np.ndarray,
    close_commands: np.ndarray,
    frame_times: np.ndarray,
    initial_open: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """把异步 Pico 短脉冲转换到 RGB 主时间轴。

    对每个 RGB 帧处理上一帧之后、当前帧之前（含当前时间）的全部 Pico 消息，
    避免用最近邻采样时漏掉短按键脉冲。第一帧之前的消息也会在首个区间处理；
    首帧测量宽度只负责提供没有命令时的初始锁存状态。
    """
    if len(frame_times) == 0:
        raise ValueError("frame_times must not be empty")
    command_state = np.empty(len(frame_times), dtype=np.uint8)
    open_event = np.zeros(len(frame_times), dtype=np.uint8)
    close_event = np.zeros(len(frame_times), dtype=np.uint8)
    desired_open = bool(initial_open)
    cursor = 0
    conflicts = 0
    for frame_i, frame_time in enumerate(frame_times):
        while cursor < len(pico_times) and int(pico_times[cursor]) <= int(frame_time):
            open_now = bool(open_commands[cursor])
            close_now = bool(close_commands[cursor])
            if open_now:
                open_event[frame_i] = 1
            if close_now:
                close_event[frame_i] = 1
            if open_now and close_now:
                # 异常冲突时闭合优先，避免把含糊命令解释成意外松爪。
                conflicts += 1
                desired_open = False
            elif close_now:
                desired_open = False
            elif open_now:
                desired_open = True
            cursor += 1
        command_state[frame_i] = int(desired_open)
    return command_state, open_event, close_event, conflicts


def construct_next(array: np.ndarray) -> np.ndarray:
    out = np.empty_like(array)
    out[:-1] = array[1:]
    out[-1] = array[-1]
    return out


def convert_episode(episode_dir: Path, args) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    db_files = sorted((episode_dir / "rosbag").glob("*.db3"))
    if len(db_files) != 1:
        raise RuntimeError(f"{episode_dir}: expected exactly one db3, got {db_files}")
    bag = BagIndex.open(db_files[0])
    try:
        joint_times, joints = load_numeric_stream(bag, "/joint_states", decode_joint_state)
        grip_times, grip = load_numeric_stream(bag, "/piper_teleop/gripper/position", decode_float64)
        grip = grip.reshape(-1, 1)
        status_times, statuses = load_status_stream(bag)
        pico_times, pico_open, pico_close = load_pico_command_stream(bag)
        rgb_times, rgb_header_times, rgb_ids = bag.image_message_index(
            "/piper_camera/d435i/color/image_raw"
        )
        depth_times, depth_header_times, depth_ids = bag.image_message_index(
            "/piper_camera/d435i/depth/image_rect_raw"
        )

        camera_info_rows = bag.rows("/piper_camera/d435i/depth/camera_info")
        if not camera_info_rows:
            raise RuntimeError("no depth CameraInfo")
        camera_info = decode_camera_info(camera_info_rows[0][2])

        states, actions, images, point_clouds = [], [], [], []
        joint_errors, grip_errors, status_errors = [], [], []
        depth_bag_errors, depth_header_errors = [], []
        control_accepted = []
        trigger_home_active = []
        retained_rgb_times = []
        retained_rgb_indices = []
        dropped_unpaired_depth = 0
        dropped_unpaired_robot = 0
        for frame_i, (timestamp, rgb_id) in enumerate(zip(rgb_times, rgb_ids)):
            t = int(timestamp)
            depth_i = nearest_index(depth_header_times, int(rgb_header_times[frame_i]))
            indexed_header_error = abs(
                int(depth_header_times[depth_i]) - int(rgb_header_times[frame_i])
            )
            if indexed_header_error > int(args.max_depth_sync_ms * 1e6):
                # 对缺失深度的 RGB 帧直接丢弃，禁止用前后相邻时刻的点云冒充。
                dropped_unpaired_depth += 1
                continue
            joint_nearest = nearest_index(joint_times, t)
            grip_nearest = nearest_index(grip_times, t)
            status_i = nearest_index(status_times, t)
            raw_sync_errors = (
                abs(int(joint_times[joint_nearest]) - t),
                abs(int(grip_times[grip_nearest]) - t),
                abs(int(status_times[status_i]) - t),
            )
            if max(raw_sync_errors) > int(args.max_robot_sync_ms * 1e6):
                # 常见于 rosbag 刚启动时相机已出帧、机器人流尚未开始的边界。
                dropped_unpaired_robot += 1
                continue
            state_joint8, joint_error = interpolate(joint_times, joints, t)
            state_grip, grip_error = interpolate(grip_times, grip, t)
            status = statuses[status_i]
            status_error = abs(int(status_times[status_i]) - t)
            command = status.get("ikCommandJointMdeg")
            if not isinstance(command, list) or len(command) != 6:
                raise RuntimeError(f"invalid ikCommandJointMdeg at RGB frame {frame_i}: {command!r}")

            # rosbag 写入时间包含两个图像 callback/序列化的调度差，不能代表曝光
            # 时刻。先用它找候选帧，严格同步检查使用相机 header.stamp。
            depth_bag_error = abs(int(depth_times[depth_i]) - t)
            if joint_error > int(args.max_robot_sync_ms * 1e6):
                raise RuntimeError(f"joint sync error {joint_error / 1e6:.3f} ms exceeds limit")
            if grip_error > int(args.max_robot_sync_ms * 1e6):
                raise RuntimeError(f"gripper sync error {grip_error / 1e6:.3f} ms exceeds limit")
            if status_error > int(args.max_robot_sync_ms * 1e6):
                raise RuntimeError(f"status sync error {status_error / 1e6:.3f} ms exceeds limit")

            rgb, rgb_encoding, _, rgb_header_ns = decode_image(bag.blob(int(rgb_id)))
            depth, depth_encoding, _, depth_header_ns = decode_image(bag.blob(int(depth_ids[depth_i])))
            depth_header_error = abs(depth_header_ns - rgb_header_ns)
            if depth_header_error > int(args.max_depth_sync_ms * 1e6):
                raise RuntimeError(
                    f"RGB/depth camera timestamp error {depth_header_error / 1e6:.3f} ms exceeds "
                    f"--max-depth-sync-ms={args.max_depth_sync_ms}"
                )
            if rgb_encoding != "rgb8" or depth_encoding != "16UC1":
                raise RuntimeError(f"unexpected encodings: RGB={rgb_encoding}, depth={depth_encoding}")

            states.append(np.r_[state_joint8[:6], float(state_grip[0])])
            actions.append(np.r_[np.asarray(command, dtype=np.float64) * MDEG_TO_RAD, float(state_grip[0])])
            rgb_chw = resize_rgb_to_chw(rgb, args.image_size, args.image_resize_mode)
            if args.rgbd:
                depth_chw = resize_depth_to_chw_uint8(
                    depth, args.image_size, args.image_resize_mode,
                    args.min_depth_m, args.max_depth_m,
                )
                images.append(np.concatenate([rgb_chw, depth_chw], axis=0))
            else:
                images.append(rgb_chw)
            point_clouds.append(
                depth_to_point_cloud(
                    depth,
                    camera_info["k"],
                    args.num_points,
                    args.min_depth_m,
                    args.max_depth_m,
                )
            )
            joint_errors.append(joint_error / 1e6)
            grip_errors.append(grip_error / 1e6)
            status_errors.append(status_error / 1e6)
            depth_bag_errors.append(depth_bag_error / 1e6)
            depth_header_errors.append(depth_header_error / 1e6)
            control_accepted.append(bool(status.get("controlAccepted", False)))
            trigger_home_active.append(bool(status.get("triggerHomeActive", False)))
            retained_rgb_times.append(t)
            retained_rgb_indices.append(frame_i)

        state = np.asarray(states, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        image = np.asarray(images, dtype=np.uint8)
        point_cloud = np.asarray(point_clouds, dtype=np.float32)
        retained_rgb_times = np.asarray(retained_rgb_times, dtype=np.int64)
        retained_rgb_indices = np.asarray(retained_rgb_indices, dtype=np.int64)
        control_accepted = np.asarray(control_accepted, dtype=bool)
        trigger_home_active = np.asarray(trigger_home_active, dtype=bool)
        synchronized_frames = len(state)
        if synchronized_frames < args.min_episode_len:
            raise RuntimeError(
                f"episode too short: {synchronized_frames} < {args.min_episode_len}"
            )
        for name, value in {"state": state, "action": action, "img": image, "point_cloud": point_cloud}.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"non-finite value in {name}")

        # controlAccepted=false 在这批数据中表示 Grip/deadman 未按下，其中还包括
        # triggerHomeActive 的采集后自动回零。自主 pick-and-place BC 不应学习这些
        # 等待/复位动作。删除它们后用 sequence_break 明确标记不连续处，避免
        # SequenceSampler 把断点两侧误当作相邻时刻。
        keep_mask = np.ones(synchronized_frames, dtype=bool)
        eligible_control = control_accepted & ~trigger_home_active
        eligible_indices = np.flatnonzero(eligible_control)
        if len(eligible_indices):
            first_control = int(eligible_indices[0])
            last_control = int(eligible_indices[-1])
            control_prefix_frames = first_control
            control_suffix_frames = synchronized_frames - 1 - last_control
            control_interior_rejected_frames = int(
                np.count_nonzero(~eligible_control[first_control : last_control + 1])
            )
        else:
            control_prefix_frames = synchronized_frames
            control_suffix_frames = 0
            control_interior_rejected_frames = 0
        if args.control_accepted_only:
            keep_mask &= eligible_control
        kept_original_indices = np.flatnonzero(keep_mask)
        if len(kept_original_indices) < args.min_episode_len:
            raise RuntimeError(
                f"episode has only {len(kept_original_indices)} accepted task frames; "
                f"minimum is {args.min_episode_len}"
            )

        # 新段开始的条件：中间删除了 RGB/控制帧，或真实时间间隔超过显式阈值。
        segment_start = np.zeros(len(kept_original_indices), dtype=bool)
        segment_start[0] = True
        if len(kept_original_indices) > 1:
            previous = kept_original_indices[:-1]
            current = kept_original_indices[1:]
            segment_start[1:] |= retained_rgb_indices[current] != (
                retained_rgb_indices[previous] + 1
            )
            if args.max_retained_frame_gap_ms is not None:
                gap_ns = retained_rgb_times[current] - retained_rgb_times[previous]
                segment_start[1:] |= gap_ns > int(
                    args.max_retained_frame_gap_ms * 1e6
                )

        state = state[keep_mask]
        action = action[keep_mask]
        image = image[keep_mask]
        point_cloud = point_cloud[keep_mask]
        selected_rgb_times = retained_rgb_times[keep_mask]
        selected_control_accepted = control_accepted[keep_mask]
        selected_trigger_home = trigger_home_active[keep_mask]
        n = len(state)

        gripper_command_state, gripper_open_event, gripper_close_event, pico_conflicts = (
            align_gripper_commands(
                pico_times,
                pico_open,
                pico_close,
                selected_rgb_times,
                initial_open=bool(state[0, 6] >= args.gripper_open_threshold_m),
            )
        )
        # 断点前发生的事件不应跨段成为监督标签；命令锁存状态仍保留。
        gripper_open_event[segment_start] = 0
        gripper_close_event[segment_start] = 0
        gripper_command_state = gripper_command_state[:, None]
        gripper_open_event = gripper_open_event[:, None]
        gripper_close_event = gripper_close_event[:, None]
        # 新训练应读取 policy_action；旧 action 保留用于兼容与逐项验证。
        policy_action = np.concatenate(
            [action[:, :6], gripper_command_state.astype(np.float32)], axis=-1
        )

        terminal_crop = None
        if args.truncate_after_gripper_open_frames is not None:
            # open_event可能包含操作者重复按键；锁存命令从close(0)切换到open(1)
            # 才是真实松爪时刻。保留松爪后的固定数量已筛选控制帧，删除其后所有
            # 撤离和回零准备动作。sequence_break仍原样保留，禁止跨越真实断点采样。
            open_transitions = (
                np.flatnonzero(
                    np.diff(gripper_command_state[:, 0].astype(np.int8)) == 1
                )
                + 1
            )
            if not len(open_transitions):
                raise RuntimeError(
                    "--truncate-after-gripper-open-frames需要每条轨迹包含至少一个"
                    "gripper_command_state 0->1松爪转换"
                )
            release_index = int(open_transitions[-1])
            crop_end = min(
                n,
                release_index + 1 + int(args.truncate_after_gripper_open_frames),
            )
            terminal_crop = {
                "release_frame_before_segment_filter": release_index,
                "requested_post_open_frames": int(
                    args.truncate_after_gripper_open_frames
                ),
                "retained_post_open_frames_before_segment_filter": int(
                    crop_end - release_index - 1
                ),
                "frames_before_crop": n,
                "frames_after_crop_before_segment_filter": crop_end,
                "suffix_frames_removed_by_terminal_crop": n - crop_end,
            }
            state = state[:crop_end]
            action = action[:crop_end]
            image = image[:crop_end]
            point_cloud = point_cloud[:crop_end]
            selected_rgb_times = selected_rgb_times[:crop_end]
            selected_control_accepted = selected_control_accepted[:crop_end]
            selected_trigger_home = selected_trigger_home[:crop_end]
            segment_start = segment_start[:crop_end]
            gripper_command_state = gripper_command_state[:crop_end]
            gripper_open_event = gripper_open_event[:crop_end]
            gripper_close_event = gripper_close_event[:crop_end]
            policy_action = policy_action[:crop_end]
            n = crop_end

        current_arrays = {
            "state": state,
            "action": action,
            "policy_action": policy_action,
            "point_cloud": point_cloud,
            "img": image,
            "gripper_command_state": gripper_command_state,
            "gripper_open_event": gripper_open_event,
            "gripper_close_event": gripper_close_event,
        }
        segment_starts = np.flatnonzero(segment_start)
        segment_ends = np.r_[segment_starts[1:], n]
        segment_lengths = segment_ends - segment_starts
        valid_segments = segment_lengths >= args.min_episode_len
        if terminal_crop is not None:
            # 固定的松爪后尾帧可能恰好单独形成长度1的segment。末端语义优先于
            # 常规短段清理：保留包含真实松爪以及其后的所有尾段，确保最终数据中
            # 每条轨迹确实有请求数量的松爪后帧。
            valid_segments |= segment_ends > release_index
        dropped_short_segment_frames = int(segment_lengths[~valid_segments].sum())
        segment_starts = segment_starts[valid_segments]
        segment_ends = segment_ends[valid_segments]
        if not len(segment_starts):
            raise RuntimeError("no accepted task segment survives --min-episode-len")

        pieces: dict[str, list[np.ndarray]] = {}
        for segment_i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            for key, value in current_arrays.items():
                part = value[start:end]
                pieces.setdefault(key, []).append(part)
                pieces.setdefault(f"next_{key}", []).append(construct_next(part))
            length = end - start
            reward = np.zeros((length, 1), dtype=np.float32)
            done = np.zeros((length, 1), dtype=bool)
            timeout = np.zeros((length, 1), dtype=bool)
            done[-1, 0] = True
            timeout[-1, 0] = True
            # 只有原任务的最后一个有效控制段获得成功奖励；较早断点只是采样边界。
            if segment_i == len(segment_starts) - 1:
                reward[-1, 0] = args.terminal_reward
            pieces.setdefault("reward", []).append(reward)
            pieces.setdefault("done", []).append(done)
            pieces.setdefault("timeout", []).append(timeout)

        arrays = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
        kept_times = np.concatenate(
            [selected_rgb_times[start:end] for start, end in zip(segment_starts, segment_ends)]
        )
        kept_control = np.concatenate(
            [selected_control_accepted[start:end] for start, end in zip(segment_starts, segment_ends)]
        )
        kept_trigger_home = np.concatenate(
            [selected_trigger_home[start:end] for start, end in zip(segment_starts, segment_ends)]
        )
        final_sequence_break = np.zeros(len(kept_times), dtype=bool)
        offset = 0
        for start, end in zip(segment_starts, segment_ends):
            final_sequence_break[offset] = True
            offset += end - start
        arrays.update({
            "timestamp_ns": kept_times,
            "control_accepted": kept_control[:, None],
            "trigger_home_active": kept_trigger_home[:, None],
            "sequence_break": final_sequence_break[:, None],
        })
        n = len(kept_times)

        def stats(values: Iterable[float]) -> dict[str, float]:
            a = np.asarray(list(values), dtype=np.float64)
            return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}

        report = {
            "episode_id": episode_dir.name,
            "frames": n,
            "synchronized_frames_before_control_filter": synchronized_frames,
            "input_rgb_frames": int(len(rgb_times)),
            "dropped_unpaired_depth_frames": dropped_unpaired_depth,
            "dropped_unpaired_robot_frames": dropped_unpaired_robot,
            "duration_sec": float((kept_times[-1] - kept_times[0]) / 1e9) if n > 1 else 0.0,
            "rgb_rate_hz": float((n - 1) / ((kept_times[-1] - kept_times[0]) / 1e9)) if n > 1 else 0.0,
            "control_accepted_ratio": float(np.mean(control_accepted)),
            "control_filter": {
                "enabled": bool(args.control_accepted_only),
                "prefix_frames_removed": (
                    control_prefix_frames if args.control_accepted_only else 0
                ),
                "suffix_frames_removed": (
                    control_suffix_frames if args.control_accepted_only else 0
                ),
                "interior_rejected_frames": (
                    control_interior_rejected_frames
                    if args.control_accepted_only else 0
                ),
                "trigger_home_frames": int(trigger_home_active.sum()),
                "segments": int(len(segment_starts)),
                "segment_lengths": [
                    int(end - start) for start, end in zip(segment_starts, segment_ends)
                ],
                "dropped_short_segment_frames": dropped_short_segment_frames,
            },
            "gripper_commands": {
                "initial_open_from_feedback": bool(
                    state[0, 6] >= args.gripper_open_threshold_m
                ),
                "state_transitions": int(
                    np.count_nonzero(np.diff(gripper_command_state[:, 0]))
                ),
                "open_event_frames": int(gripper_open_event.sum()),
                "close_event_frames": int(gripper_close_event.sum()),
                "conflicting_pico_messages": int(pico_conflicts),
            },
            "terminal_crop": terminal_crop,
            "sync": {
                "depth_camera_header": stats(depth_header_errors),
                "depth_bag_write_time": stats(depth_bag_errors),
                "joint": stats(joint_errors),
                "gripper": stats(grip_errors),
                "status": stats(status_errors),
            },
            "depth_camera": {
                "frame_id": camera_info["frame_id"],
                "width": camera_info["width"],
                "height": camera_info["height"],
                "k": camera_info["k"].reshape(-1).tolist(),
            },
        }
        return arrays, report
    finally:
        bag.connection.close()


def compute_return(reward: np.ndarray, done: np.ndarray, timeout: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(reward))):
        not_done = 1.0 - float(done[i, 0] or timeout[i, 0])
        running = float(reward[i, 0]) + gamma * running * not_done
        out[i, 0] = running
    return out


def create_array(group, name: str, array: np.ndarray) -> None:
    try:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:
        compressor = None
    lead = min(100, max(1, len(array)))
    chunks = (lead,) + array.shape[1:]
    kwargs = {"data": array, "chunks": chunks, "overwrite": True}
    if compressor is not None:
        kwargs["compressor"] = compressor
    group.create_dataset(name, **kwargs)


def write_zarr(episodes: list[dict[str, np.ndarray]], reports: list[dict[str, Any]], args) -> None:
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(args.output))
    data = root.create_group("data")
    meta = root.create_group("meta")
    keys = list(episodes[0])
    arrays = {key: np.concatenate([ep[key] for ep in episodes], axis=0) for key in keys}
    arrays["return"] = compute_return(arrays["reward"], arrays["done"], arrays["timeout"], args.gamma)
    episode_ends, total = [], 0
    for ep in episodes:
        total += len(ep["state"])
        episode_ends.append(total)
    for key, value in arrays.items():
        create_array(data, key, value)
        print(f"[zarr] data/{key}: {value.shape} {value.dtype}", flush=True)
        gc.collect()
    create_array(meta, "episode_ends", np.asarray(episode_ends, dtype=np.int64))
    root.attrs.update(
        {
            "schema": "rl100_replay_zarr_from_piper_rosbag_pick_place_v2_gripper_commands",
            "state_definition": "joint_feedback_rad[0:6] + gripper_total_width_m",
            "action_definition": (
                "ikCommandJointMdeg_to_rad[0:6] + gripper_position_m; action[6] is legacy "
                "feedback compatibility only, use data/gripper_command_state for gripper supervision"
            ),
            "policy_action_definition": (
                "ikCommandJointMdeg_to_rad[0:6] + latched gripper command state; "
                "use this tensor for new BC/CM/Q/dynamics/offline training"
            ),
            "gripper_command_definition": {
                "state": "latched Pico command, uint8: 1=open, 0=close",
                "open_event": "any gripperOpen=true Pico message since previous retained RGB frame",
                "close_event": "any gripperClose=true Pico message since previous retained RGB frame",
                "initial_feedback_threshold_m": args.gripper_open_threshold_m,
            },
            "point_cloud_frame": "d435i_depth_optical_frame",
            "point_cloud_depth_scale_m": 0.001,
            "image_preprocessing": {
                "input_color": "RGB",
                "output_layout": "CHW",
                "output_size": [args.image_size, args.image_size],
                "resize_mode": args.image_resize_mode,
                "channels": "RGBD" if args.rgbd else "RGB",
                "depth_encoding": (
                    "uint8: round(clip((depth_m-min_depth_m)/(max_depth_m-min_depth_m),0,1)*255), "
                    "invalid/out-of-range=0"
                    if args.rgbd else None
                ),
            },
            "source_manifest": reports,
            "sequence_break_definition": (
                "true on the first frame of each contiguous accepted task segment; "
                "training windows must not cross this boundary"
            ),
            "control_filter": {
                "control_accepted_only": bool(args.control_accepted_only),
                "max_retained_frame_gap_ms": args.max_retained_frame_gap_ms,
            },
            "terminal_crop": {
                "truncate_after_gripper_open_frames": (
                    args.truncate_after_gripper_open_frames
                ),
                "release_definition": (
                    "last gripper_command_state close-to-open (0->1) transition"
                ),
            },
        }
    )


def validate_zarr(path: Path, expected_episodes: int) -> dict[str, Any]:
    root = zarr.open(str(path), mode="r")
    required = {
        "state", "next_state", "action", "next_action", "policy_action", "next_policy_action",
        "point_cloud", "next_point_cloud",
        "img", "next_img", "reward", "return", "done", "timeout",
        "gripper_command_state", "next_gripper_command_state",
        "gripper_open_event", "next_gripper_open_event",
        "gripper_close_event", "next_gripper_close_event",
    }
    missing = required - set(root["data"].keys())
    if missing:
        raise RuntimeError(f"missing Zarr arrays: {sorted(missing)}")
    ends = root["meta/episode_ends"][:]
    if len(ends) != expected_episodes or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise RuntimeError(f"invalid episode_ends: {ends}")
    n = int(ends[-1])
    for key in required:
        if root[f"data/{key}"].shape[0] != n:
            raise RuntimeError(f"data/{key} length mismatch")
    if root["data/state"].shape[1] != 7 or root["data/action"].shape[1] != 7:
        raise RuntimeError("state/action must both be 7-D")
    state = root["data/state"][:]
    action = root["data/action"][:]
    policy_action = root["data/policy_action"][:]
    command_state = root["data/gripper_command_state"][:]
    open_event = root["data/gripper_open_event"][:]
    close_event = root["data/gripper_close_event"][:]
    for name, value in {
        "gripper_command_state": command_state,
        "gripper_open_event": open_event,
        "gripper_close_event": close_event,
    }.items():
        if value.shape != (n, 1) or not np.all((value == 0) | (value == 1)):
            raise RuntimeError(f"{name} must have shape ({n}, 1) and contain only 0/1")
    if policy_action.shape != action.shape:
        raise RuntimeError(
            f"policy_action shape {policy_action.shape} != action shape {action.shape}"
        )
    if not np.array_equal(policy_action[:, :6], action[:, :6]):
        raise RuntimeError("policy_action joint targets differ from action joint targets")
    if not np.array_equal(policy_action[:, 6], command_state[:, 0].astype(np.float32)):
        raise RuntimeError("policy_action gripper dimension differs from command state")
    starts = np.r_[0, ends[:-1]]
    transitions_per_episode = [
        int(np.count_nonzero(np.diff(command_state[start:end, 0])))
        for start, end in zip(starts, ends)
    ]
    terminal_crop = root.attrs.get("terminal_crop", {})
    requested_post_open_frames = terminal_crop.get(
        "truncate_after_gripper_open_frames"
    )
    post_open_frames_per_episode = []
    if requested_post_open_frames is not None:
        for start, end in zip(starts, ends):
            open_transitions = (
                np.flatnonzero(
                    np.diff(command_state[start:end, 0].astype(np.int8)) == 1
                )
                + 1
            )
            if not len(open_transitions):
                raise RuntimeError(
                    f"episode [{start}, {end})缺少gripper_command_state 0->1松爪转换"
                )
            post_frames = int(end - start - int(open_transitions[-1]) - 1)
            post_open_frames_per_episode.append(post_frames)
            if post_frames != int(requested_post_open_frames):
                raise RuntimeError(
                    f"episode [{start}, {end})松爪后帧数={post_frames}，"
                    f"期望={requested_post_open_frames}"
                )
    return {
        "episodes": len(ends),
        "transitions": n,
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "policy_action_min": policy_action.min(axis=0).tolist(),
        "policy_action_max": policy_action.max(axis=0).tolist(),
        "gripper_command_open_ratio": float(command_state.mean()),
        "gripper_open_event_frames": int(open_event.sum()),
        "gripper_close_event_frames": int(close_event.sum()),
        "gripper_state_transitions_per_episode": transitions_per_episode,
        "post_open_frames_per_episode": post_open_frames_per_episode,
    }


def discover_episodes(input_dir: Path, limit: int | None) -> tuple[list[Path], list[dict[str, Any]]]:
    accepted, skipped = [], []
    for episode_dir in sorted(input_dir.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            skipped.append({"episode_id": episode_dir.name, "reasons": ["missing metadata.json"]})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ok, reasons = episode_is_success(metadata)
        if ok:
            accepted.append(episode_dir)
        else:
            skipped.append({"episode_id": episode_dir.name, "reasons": reasons})
    if limit is not None:
        accepted = accepted[:limit]
    return accepted, skipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="包含 episode_* 目录的 piper_data")
    parser.add_argument("--output", type=Path, required=True, help="输出 .zarr 路径")
    parser.add_argument("--report", type=Path, default=None, help="JSON 转换报告；默认在 Zarr 同级生成")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只发现和过滤 episode，不读取图像")
    parser.add_argument("--limit", type=int, default=None, help="仅转换前 N 条成功轨迹，用于 smoke test")
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--image-resize-mode", choices=["stretch", "letterbox"], default="stretch")
    parser.add_argument("--rgbd", action="store_true", help="将深度量化后作为img的第4通道")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--max-depth-sync-ms", type=float, default=10.0)
    parser.add_argument("--max-robot-sync-ms", type=float, default=20.0)
    parser.add_argument(
        "--control-accepted-only",
        action="store_true",
        help=(
            "仅保留controlAccepted=true且非triggerHomeActive的任务控制帧；"
            "删除区间会生成sequence_break，防止训练窗口跨越"
        ),
    )
    parser.add_argument(
        "--max-retained-frame-gap-ms",
        type=float,
        default=None,
        help="相邻保留帧超过该时间间隔时写入sequence_break；默认不额外按时间切断",
    )
    parser.add_argument("--min-episode-len", type=int, default=2)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--gripper-open-threshold-m",
        type=float,
        default=0.061,
        help="仅用于每条轨迹首帧的命令状态初始化；后续标签完全由 Pico 命令锁存",
    )
    parser.add_argument(
        "--truncate-after-gripper-open-frames",
        type=int,
        default=None,
        help=(
            "在最后一次gripper_command_state 0->1真实松爪后仅保留N个已筛选控制帧，"
            "删除后续撤离/回零准备动作；默认不截断"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.num_points <= 0:
        raise ValueError("--image-size and --num-points must be positive")
    if (
        args.truncate_after_gripper_open_frames is not None
        and args.truncate_after_gripper_open_frames < 0
    ):
        raise ValueError("--truncate-after-gripper-open-frames must be non-negative")
    accepted, skipped = discover_episodes(args.input, args.limit)
    print(f"[discover] accepted={len(accepted)} skipped={len(skipped)}", flush=True)
    for item in skipped:
        print(f"[skip] {item['episode_id']}: {', '.join(item['reasons'])}", flush=True)
    if not accepted:
        raise RuntimeError("no successful episodes found")
    if args.dry_run:
        for path in accepted:
            print(f"[accept] {path.name}")
        return

    episodes, episode_reports = [], []
    for i, path in enumerate(accepted, 1):
        print(f"[convert] {i}/{len(accepted)} {path.name}", flush=True)
        arrays, report = convert_episode(path, args)
        episodes.append(arrays)
        episode_reports.append(report)
        print(
            f"[episode] frames={report['frames']} rate={report['rgb_rate_hz']:.2f}Hz "
            f"depth_header_sync_max={report['sync']['depth_camera_header']['max_ms']:.3f}ms "
            f"robot_sync_max={max(report['sync'][k]['max_ms'] for k in ('joint','gripper','status')):.3f}ms",
            flush=True,
        )

    write_zarr(episodes, episode_reports, args)
    validation = validate_zarr(args.output, len(accepted))
    report_path = args.report or args.output.with_suffix(".conversion_report.json")
    full_report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "accepted_episode_ids": [p.name for p in accepted],
        "skipped_episodes": skipped,
        "configuration": {
            "image_size": args.image_size,
            "image_resize_mode": args.image_resize_mode,
            "rgbd": args.rgbd,
            "num_points": args.num_points,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "terminal_reward": args.terminal_reward,
            "gamma": args.gamma,
            "gripper_open_threshold_m": args.gripper_open_threshold_m,
            "control_accepted_only": args.control_accepted_only,
            "max_retained_frame_gap_ms": args.max_retained_frame_gap_ms,
            "truncate_after_gripper_open_frames": (
                args.truncate_after_gripper_open_frames
            ),
        },
        "episodes": episode_reports,
        "validation": validation,
    }
    report_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validate] OK: {validation}", flush=True)
    print(f"[saved] zarr={args.output}", flush=True)
    print(f"[saved] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
