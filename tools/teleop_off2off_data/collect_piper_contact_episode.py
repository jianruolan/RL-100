import argparse
import json
import sys
import time
import threading
from pathlib import Path

import h5py
import numpy as np

from realsense import RealSense


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPER_SDK_ROOT = REPO_ROOT.parent / "01-Piper" / "piper_sdk"


def add_piper_sdk_to_path(piper_sdk_root: Path) -> None:
    piper_sdk_root = piper_sdk_root.expanduser().resolve()
    if not piper_sdk_root.exists():
        raise FileNotFoundError(f"piper sdk root does not exist: {piper_sdk_root}")
    sys.path.insert(0, str(piper_sdk_root))


class PiperStateReader:
    """Read-only Piper wrapper for demonstration data collection.

    This class intentionally does not send motion/gripper commands.  It only
    connects to the CAN feedback stream and converts the raw SDK units into
    training-friendly values.
    """

    def __init__(self, can_name: str, piper_sdk_root: Path, warmup_sec: float = 1.0):
        add_piper_sdk_to_path(piper_sdk_root)
        from piper_sdk import C_PiperInterface_V2

        self.can_name = can_name
        self.piper = C_PiperInterface_V2(can_name)
        print(f"[piper] connect CAN port: {can_name}", flush=True)
        self.piper.ConnectPort()
        if warmup_sec > 0:
            print(f"[piper] warmup {warmup_sec:.1f}s", flush=True)
            time.sleep(warmup_sec)

    def get_enable_status(self) -> list:
        return list(self.piper.GetArmEnableStatus())

    def print_enable_status(self, prefix: str) -> None:
        status = self.get_enable_status()
        print(f"[piper] {prefix} enable_status={status}", flush=True)

    def enable_until_ready(self, timeout_sec: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if self.piper.EnablePiper():
                self.print_enable_status("enabled")
                return True
            time.sleep(0.01)
        self.print_enable_status("enable timeout")
        return False

    def disable_until_ready(self, timeout_sec: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if not self.piper.DisablePiper():
                self.print_enable_status("disabled")
                return True
            time.sleep(0.01)
        self.print_enable_status("disable timeout")
        return False

    def start_drag_teach(self) -> None:
        # Official Piper SDK MotionCtrl_1 grag_teach_ctrl:
        # 0x01 = enter drag teaching record mode.
        print("[piper] enter drag-teach mode: MotionCtrl_1(0x00, 0x00, 0x01)", flush=True)
        self.piper.MotionCtrl_1(0x00, 0x00, 0x01)
        time.sleep(0.1)
        self.print_enable_status("after enter drag-teach")

    def stop_drag_teach(self) -> None:
        # Official Piper SDK MotionCtrl_1 grag_teach_ctrl:
        # 0x02 = exit drag teaching record mode.
        print("[piper] exit drag-teach mode: MotionCtrl_1(0x00, 0x00, 0x02)", flush=True)
        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)
        time.sleep(0.1)
        self.print_enable_status("after exit drag-teach")

    def set_master_passive_output(self) -> None:
        # This mirrors piper_sdk/demo/V2/master_passive.py exactly.
        print("[piper] set master/slave passive output: MasterSlaveConfig(0xFC, 0, 0, 0)", flush=True)
        self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.1)

    def apply_collection_state(self, policy: str, timeout_sec: float) -> None:
        self.print_enable_status("before collection state switch")
        if policy == "none":
            print("[piper] arm-state-policy=none; keep current Piper state", flush=True)
        elif policy == "drag_teach":
            self.start_drag_teach()
        elif policy == "disable":
            print("[piper] arm-state-policy=disable; disabling motors for manual dragging", flush=True)
            ok = self.disable_until_ready(timeout_sec=timeout_sec)
            if not ok:
                raise RuntimeError("failed to disable Piper before collection")
        elif policy == "master_passive":
            self.set_master_passive_output()
        else:
            raise ValueError(f"unknown arm state policy: {policy}")

    def restore_after_collection(self, policy: str, restore_enable: bool, timeout_sec: float) -> None:
        if policy == "drag_teach":
            self.stop_drag_teach()
        if restore_enable:
            print("[piper] restore enable after collection", flush=True)
            ok = self.enable_until_ready(timeout_sec=timeout_sec)
            if not ok:
                raise RuntimeError("failed to re-enable Piper after collection")
        else:
            self.print_enable_status("after collection")

    @staticmethod
    def _joint_raw_to_rad(joint_raw: np.ndarray) -> np.ndarray:
        # Piper feedback joint unit: 0.001 degree.
        return joint_raw.astype(np.float64) * 0.001 * np.pi / 180.0

    @staticmethod
    def _gripper_raw_to_m(gripper_raw: int) -> float:
        # Piper gripper stroke unit: 0.001 mm.
        return float(gripper_raw) * 1e-6

    def read(self) -> dict:
        joint_msg = self.piper.GetArmJointMsgs()
        gripper_msg = self.piper.GetArmGripperMsgs()

        joint_state = joint_msg.joint_state
        gripper_state = gripper_msg.gripper_state

        joint_raw = np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ],
            dtype=np.int64,
        )
        joint_rad = self._joint_raw_to_rad(joint_raw)

        gripper_raw = int(gripper_state.grippers_angle)
        gripper_m = self._gripper_raw_to_m(gripper_raw)
        state_7d = np.concatenate([joint_rad, np.array([gripper_m], dtype=np.float64)])

        return {
            "time_ns": time.time_ns(),
            "joint_sdk_time": float(joint_msg.time_stamp),
            "gripper_sdk_time": float(gripper_msg.time_stamp),
            "joint_hz": float(joint_msg.Hz),
            "gripper_hz": float(gripper_msg.Hz),
            "joint_raw": joint_raw,
            "joint_rad": joint_rad,
            "gripper_raw": gripper_raw,
            "gripper_m": gripper_m,
            "gripper_effort_raw": int(gripper_state.grippers_effort),
            "gripper_status_code": int(gripper_state.status_code),
            "state_7d": state_7d,
        }


def make_action_from_state(state_7d: np.ndarray) -> np.ndarray:
    """Use next observed state as action placeholder for hand-drag demos."""

    if len(state_7d) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    action = np.empty_like(state_7d)
    action[:-1] = state_7d[1:]
    action[-1] = state_7d[-1]
    return action


def format_joint_feedback(robot_state: dict) -> str:
    joint_raw = robot_state["joint_raw"]
    joint_deg = joint_raw.astype(np.float64) * 0.001
    joint_rad = robot_state["joint_rad"]
    raw_str = np.array2string(joint_raw, separator=", ")
    deg_str = np.array2string(joint_deg, precision=2, separator=", ")
    rad_str = np.array2string(joint_rad, precision=4, separator=", ")
    return f"joint_raw_0.001deg={raw_str} joint_deg={deg_str} joint_rad={rad_str}"


def collect_loop(args, robot, camera, stop_event, stop_clock, max_seconds: float | None, mode: str) -> dict:
    dt = 1.0 / args.rate
    samples = []
    loop_start_ns = time.time_ns()
    loop_start_wall = time.time()
    joint_print_every = max(1, int(round(args.rate / max(args.joint_print_hz, 1e-6))))

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if max_seconds is not None and (time.time() - loop_start_wall) >= max_seconds:
            break

        iter_t0 = time.time()
        robot_state = robot.read()
        frame = None
        if camera is not None:
            frame = camera.get_frame(require_pc=args.save_point_cloud)
        host_time_ns = time.time_ns()
        episode_time_s = (host_time_ns - loop_start_ns) * 1e-9
        stop_ns = None if stop_clock is None else stop_clock.get("ns")
        if stop_ns is not None and host_time_ns >= stop_ns:
            print(
                f"[collect:{mode}] stop requested at {stop_ns}, "
                f"discarding sample {len(samples)} captured at {host_time_ns}",
                flush=True,
            )
            break

        sample = {
            "idx": len(samples),
            "host_time_ns": host_time_ns,
            "episode_time_s": episode_time_s,
            "robot": robot_state,
            "camera_timestamp": np.nan if frame is None else float(frame["timestamp"]),
            "rgb": None if frame is None else frame["color"],
            "depth": None if frame is None else frame["depth"],
            "depth_scale": np.nan if frame is None else float(frame["depth_scale"]),
            "point_cloud": None if frame is None else frame["point_cloud"],
        }
        samples.append(sample)

        elapsed = time.time() - iter_t0
        should_print_progress = len(samples) == 1 or len(samples) % max(1, args.rate) == 0
        should_print_joint = args.print_joints and (
            len(samples) == 1 or len(samples) % joint_print_every == 0
        )
        if should_print_progress or should_print_joint:
            print(
                f"[collect:{mode}] {len(samples):04d} "
                f"loop={elapsed * 1000:.1f}ms "
                f"joint_hz={robot_state['joint_hz']:.1f} "
                f"gripper_hz={robot_state['gripper_hz']:.1f}"
                + (f" {format_joint_feedback(robot_state)}" if should_print_joint else ""),
                flush=True,
            )
        sleep_s = dt - (time.time() - iter_t0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return {
        "samples": samples,
        "loop_start_ns": loop_start_ns,
    }


def collect_static(args) -> dict:
    print("[collect] mode=static", flush=True)
    print(f"[collect] duration={args.duration}s rate={args.rate}Hz", flush=True)

    robot = PiperStateReader(
        can_name=args.can,
        piper_sdk_root=args.piper_sdk_root,
        warmup_sec=args.robot_warmup,
    )

    camera = None
    camera_meta = {}
    if not args.no_camera:
        print("[camera] start RealSense", flush=True)
        camera = RealSense(num_points=args.num_points)
        camera.start()
        camera_meta = {
            "depth_intrinsics": list(camera.depth_intrinsics),
            "color_intrinsics": list(camera.color_intrinsics),
        }

    try:
        n_target = int(round(args.duration * args.rate))
        print("[collect] mode=static start", flush=True)
        record = collect_loop(
            args,
            robot,
            camera,
            stop_event=None,
            stop_clock=None,
            max_seconds=args.duration,
            mode="static",
        )
    finally:
        if camera is not None:
            print("[camera] stop RealSense", flush=True)
            camera.stop()

    return {
        "samples": record["samples"],
        "meta": {
            "mode": "static",
            "task": "piper_soft_block_contact",
            "can": args.can,
            "rate": args.rate,
            "duration": args.duration,
            "num_points": args.num_points,
            "save_point_cloud": bool(args.save_point_cloud),
            "no_camera": bool(args.no_camera),
            "schema_version": "piper_contact_raw_v0",
            "gripper_control_available": False,
            "gripper_dimension_retained": True,
            "episode_from_home_pose_recommended": True,
            "notes": args.notes,
        },
        "episode": {
            "record_length_target": n_target,
            "record_length_actual": len(record["samples"]),
            "start_time_ns": record["loop_start_ns"],
        },
        "camera_meta": camera_meta,
    }


def collect_episode(args) -> dict:
    print("[collect] mode=episode", flush=True)
    print(f"[collect] rate={args.rate}Hz max_seconds={args.episode_max_seconds}s", flush=True)
    print("[collect] recommend: start each episode from the same home pose / zero pose if safe", flush=True)
    print("[collect] press Enter to start recording, then press Enter again to stop", flush=True)

    robot = PiperStateReader(
        can_name=args.can,
        piper_sdk_root=args.piper_sdk_root,
        warmup_sec=args.robot_warmup,
    )

    camera = None
    camera_meta = {}
    if not args.no_camera:
        print("[camera] start RealSense", flush=True)
        camera = RealSense(num_points=args.num_points)
        camera.start()
        camera_meta = {
            "depth_intrinsics": list(camera.depth_intrinsics),
            "color_intrinsics": list(camera.color_intrinsics),
        }

    record = None
    try:
        if sys.stdin.isatty():
            input(
                "[collect] keep the arm enabled and move it to the home/zero pose, then press Enter..."
            )
            print("[collect] switching Piper into collection state...", flush=True)
            robot.apply_collection_state(args.arm_state_policy, args.arm_state_timeout)
            input("[collect] collection state is ready, press Enter to start recording...")
            stop_event = threading.Event()
            stop_clock = {"ns": None}

            def wait_for_stop():
                try:
                    input()
                finally:
                    stop_clock["ns"] = time.time_ns()
                    stop_event.set()

            threading.Thread(target=wait_for_stop, daemon=True).start()
            print("[collect] recording... press Enter again to stop", flush=True)
            record = collect_loop(
                args,
                robot,
                camera,
                stop_event=stop_event,
                stop_clock=stop_clock,
                max_seconds=args.episode_max_seconds,
                mode="episode",
            )
        else:
            print("[collect] stdin is not a tty; starting immediately", flush=True)
            robot.apply_collection_state(args.arm_state_policy, args.arm_state_timeout)
            print("[collect] recording... move the arm manually now", flush=True)
            record = collect_loop(
                args,
                robot,
                camera,
                stop_event=None,
                stop_clock=None,
                max_seconds=args.episode_max_seconds,
                mode="episode",
            )
    finally:
        if camera is not None:
            print("[camera] stop RealSense", flush=True)
            camera.stop()
        robot.restore_after_collection(
            policy=args.arm_state_policy,
            restore_enable=args.restore_enable,
            timeout_sec=args.arm_state_timeout,
        )

    return {
        "samples": record["samples"],
        "meta": {
            "mode": "episode",
            "task": "piper_soft_block_contact",
            "can": args.can,
            "rate": args.rate,
            "episode_max_seconds": args.episode_max_seconds,
            "num_points": args.num_points,
            "save_point_cloud": bool(args.save_point_cloud),
            "no_camera": bool(args.no_camera),
            "schema_version": "piper_contact_raw_v0",
            "gripper_control_available": False,
            "gripper_dimension_retained": True,
            "episode_from_home_pose_recommended": True,
            "arm_state_policy": args.arm_state_policy,
            "restore_enable": bool(args.restore_enable),
            "notes": args.notes,
        },
        "episode": {
            "record_length_actual": len(record["samples"]),
            "start_time_ns": record["loop_start_ns"],
        },
        "camera_meta": camera_meta,
    }


def stack_optional(samples, key: str):
    values = [s[key] for s in samples]
    if values[0] is None:
        return None
    return np.stack(values, axis=0)


def save_hdf5(result: dict, out_path: Path) -> None:
    samples = result["samples"]
    meta = result["meta"]
    camera_meta = result.get("camera_meta", {})
    if not samples:
        raise RuntimeError("no samples collected")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    host_time_ns = np.array([s["host_time_ns"] for s in samples], dtype=np.int64)
    camera_time = np.array([s["camera_timestamp"] for s in samples], dtype=np.float64)
    depth_scale = np.array([s["depth_scale"] for s in samples], dtype=np.float64)

    joint_raw = np.stack([s["robot"]["joint_raw"] for s in samples], axis=0)
    joint_rad = np.stack([s["robot"]["joint_rad"] for s in samples], axis=0)
    gripper_raw = np.array([s["robot"]["gripper_raw"] for s in samples], dtype=np.int64)
    gripper_m = np.array([s["robot"]["gripper_m"] for s in samples], dtype=np.float64)
    gripper_effort_raw = np.array(
        [s["robot"]["gripper_effort_raw"] for s in samples],
        dtype=np.int64,
    )
    gripper_status_code = np.array(
        [s["robot"]["gripper_status_code"] for s in samples],
        dtype=np.uint8,
    )
    joint_sdk_time = np.array([s["robot"]["joint_sdk_time"] for s in samples], dtype=np.float64)
    gripper_sdk_time = np.array(
        [s["robot"]["gripper_sdk_time"] for s in samples],
        dtype=np.float64,
    )
    joint_hz = np.array([s["robot"]["joint_hz"] for s in samples], dtype=np.float64)
    gripper_hz = np.array([s["robot"]["gripper_hz"] for s in samples], dtype=np.float64)
    state_7d = np.stack([s["robot"]["state_7d"] for s in samples], axis=0)
    action_7d = make_action_from_state(state_7d)
    episode_time_s = np.array([s["episode_time_s"] for s in samples], dtype=np.float64)

    rgb = stack_optional(samples, "rgb")
    depth = stack_optional(samples, "depth")
    point_cloud = stack_optional(samples, "point_cloud")

    with h5py.File(out_path, "w") as f:
        f.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False)
        f.attrs["created_time_ns"] = time.time_ns()
        f.attrs["description"] = (
            "Raw Piper single-arm contact demonstration data. "
            "state/action keep 7D, with gripper dimension retained even when frozen."
        )
        if camera_meta:
            f.attrs["camera_meta_json"] = json.dumps(camera_meta, ensure_ascii=False)

        g_time = f.create_group("time")
        g_time.create_dataset("host_time_ns", data=host_time_ns)
        g_time.create_dataset("episode_time_s", data=episode_time_s)
        g_time.create_dataset("camera_time_s", data=camera_time)
        g_time.create_dataset("joint_sdk_time", data=joint_sdk_time)
        g_time.create_dataset("gripper_sdk_time", data=gripper_sdk_time)

        g_robot = f.create_group("robot")
        g_robot.create_dataset("joint_raw_0p001deg", data=joint_raw)
        g_robot.create_dataset("joint_rad", data=joint_rad)
        g_robot.create_dataset("gripper_raw_0p001mm", data=gripper_raw)
        g_robot.create_dataset("gripper_m", data=gripper_m)
        g_robot.create_dataset("gripper_effort_raw", data=gripper_effort_raw)
        g_robot.create_dataset("gripper_status_code", data=gripper_status_code)
        g_robot.create_dataset("joint_hz", data=joint_hz)
        g_robot.create_dataset("gripper_hz", data=gripper_hz)
        g_robot.create_dataset("state_7d", data=state_7d)
        g_robot.create_dataset("action_7d", data=action_7d)

        g_camera = f.create_group("camera")
        g_camera.create_dataset("depth_scale", data=depth_scale)
        if rgb is not None:
            g_camera.create_dataset(
                "rgb",
                data=rgb,
                compression="gzip",
                compression_opts=4,
                chunks=(1, *rgb.shape[1:]),
            )
        if depth is not None:
            g_camera.create_dataset(
                "depth",
                data=depth,
                compression="gzip",
                compression_opts=4,
                chunks=(1, *depth.shape[1:]),
            )
        if point_cloud is not None:
            g_camera.create_dataset("point_cloud", data=point_cloud)


def inspect_hdf5(out_path: Path) -> None:
    with h5py.File(out_path, "r") as f:
        print("[inspect] saved:", out_path, flush=True)
        print("[inspect] meta:", f.attrs["meta_json"], flush=True)
        for name in [
            "time/host_time_ns",
            "robot/state_7d",
            "robot/action_7d",
            "camera/rgb",
            "camera/depth",
            "camera/point_cloud",
        ]:
            if name in f:
                print(f"[inspect] {name}: {f[name].shape} {f[name].dtype}", flush=True)
        state = f["robot/state_7d"][:]
        print("[inspect] state_7d first:", state[0], flush=True)
        print("[inspect] state_7d last :", state[-1], flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "episode"], default="static")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--episode-max-seconds", type=float, default=30.0)
    parser.add_argument("--rate", type=int, default=20)
    parser.add_argument("--can", type=str, default="can0")
    parser.add_argument("--piper-sdk-root", type=Path, default=DEFAULT_PIPER_SDK_ROOT)
    parser.add_argument("--robot-warmup", type=float, default=1.0)
    parser.add_argument(
        "--arm-state-policy",
        choices=["none", "drag_teach", "disable", "master_passive"],
        default="drag_teach",
        help=(
            "Piper state switch before episode collection. "
            "drag_teach uses MotionCtrl_1(..., 0x01/0x02) from the official SDK; "
            "disable uses DisablePiper; master_passive mirrors demo/V2/master_passive.py."
        ),
    )
    parser.add_argument("--arm-state-timeout", type=float, default=5.0)
    parser.add_argument(
        "--restore-enable",
        action="store_true",
        help="re-enable Piper after collection; off by default for manual drag data collection",
    )
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--save-point-cloud", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--print-joints", action="store_true", default=True)
    parser.add_argument("--no-print-joints", action="store_false", dest="print_joints")
    parser.add_argument("--joint-print-hz", type=float, default=1.0)
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.episode_max_seconds <= 0:
        raise ValueError("--episode-max-seconds must be positive")

    if args.mode == "static":
        result = collect_static(args)
    elif args.mode == "episode":
        result = collect_episode(args)
    else:
        raise NotImplementedError(args.mode)

    save_hdf5(result, args.out)
    inspect_hdf5(args.out)


if __name__ == "__main__":
    main()
