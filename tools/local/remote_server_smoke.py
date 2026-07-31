#!/usr/bin/env python3
"""Network-only smoke test for the remote DP3 gRPC inference service.

This script never imports ROS, RealSense, or Piper SDK. It sends one synthetic
observation to the real server and validates the returned action chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local import piper_remote_runtime as runtime


EXPECTED_OUTPUT_NAME = (
    "piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_"
    "episode10_bs64_epoch4000_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--expected-output-name", default=EXPECTED_OUTPUT_NAME)
    parser.add_argument("--expected-policy-subdir", default="bc")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--rpc-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = runtime.PolicyClient(args.server, args.connect_timeout)
    episode_id = f"network-smoke-{uuid.uuid4().hex}"
    started = time.monotonic()
    try:
        contract = client.get_contract(args.rpc_timeout)
        contract.validate_model_identity(
            args.expected_output_name, args.expected_policy_subdir
        )
        client.reset_episode(episode_id, args.rpc_timeout)

        # Shape-valid, finite tensors only. No device, camera, ROS, or Piper access.
        agent_pos = np.zeros(runtime.AGENT_POS_SHAPE, dtype=np.float32)
        image = np.zeros(runtime.IMAGE_SHAPE, dtype=np.float32)
        point_cloud = np.zeros(runtime.POINT_CLOUD_SHAPE, dtype=np.float32)
        response = None
        for sequence_id in range(1, contract.n_obs_steps + 1):
            request = runtime.make_request(
                episode_id,
                sequence_id=sequence_id,
                capture_timestamp_ns=time.time_ns(),
                agent_pos=agent_pos,
                image=image,
                point_cloud=point_cloud,
            )
            response = client.infer(request, args.rpc_timeout)
            if response.protocol_version != runtime.PROTOCOL_VERSION:
                raise RuntimeError(
                    f"响应协议不一致: server={response.protocol_version!r}, "
                    f"local={runtime.PROTOCOL_VERSION!r}"
                )
            if (
                response.episode_id != episode_id
                or response.sequence_id != sequence_id
            ):
                raise RuntimeError(
                    "响应 episode_id/sequence_id 不匹配: "
                    f"episode={response.episode_id!r}, "
                    f"seq={response.sequence_id}, expected={sequence_id}"
                )
            if not response.ready and sequence_id < contract.n_obs_steps:
                print(
                    f"[network-smoke] 预热观测历史 {sequence_id}/"
                    f"{contract.n_obs_steps}: {response.status_message}",
                    flush=True,
                )
                continue

        if response is None:
            raise RuntimeError("未收到服务器响应")
        if not response.ready:
            raise RuntimeError(f"服务器未就绪: {response.status_message}")
        action = runtime.decode_f32(
            response.action_chunk_f32,
            (contract.n_action_steps, runtime.ACTION_SHAPE[0]),
            "action_chunk",
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        result = {
            "server": args.server,
            "protocol_version": response.protocol_version,
            "model_version": response.model_version,
            "output_dir": contract.output_dir,
            "policy_subdir": contract.policy_subdir,
            "episode_id": episode_id,
            "sequence_id": response.sequence_id,
            "action_shape": list(action.shape),
            "inference_time_ms": response.inference_time_ms,
            "round_trip_time_ms": elapsed_ms,
            "status": "PASS",
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(
                "[network-smoke] PASS: "
                f"server={args.server}, model={response.model_version}, "
                f"action_shape={tuple(action.shape)}, "
                f"inference={response.inference_time_ms:.2f}ms, "
                f"round_trip={elapsed_ms:.2f}ms",
                flush=True,
            )
        return 0
    finally:
        try:
            client.reset_episode(episode_id, args.rpc_timeout)
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
