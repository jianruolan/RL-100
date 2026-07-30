#!/usr/bin/env python3
"""Measure the laptop <-> inference-server transport path.

The protocol keeps one TCP connection open, uploads one encoded image per
request, and returns a 7-float robot action.  It deliberately uses only the
Python standard library so that the same file can be copied to a clean server.
OpenCV is optional and is used only when ``--decode-image`` is requested.

This is a transport probe, not a robot controller: the returned action is all
zeros and must never be forwarded to a robot as a policy command.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 1024 * 1024
ACTION_BYTES = struct.pack("!7f", *([0.0] * 7))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, header: dict[str, Any], payload: bytes) -> None:
    header = dict(header)
    header["protocol_version"] = PROTOCOL_VERSION
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("message header is too large")
    sock.sendall(struct.pack("!I", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def _recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if header_size > MAX_HEADER_BYTES:
        raise ValueError(f"refusing {header_size}-byte header")
    header = json.loads(_recv_exact(sock, header_size))
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol version mismatch")
    payload_size = int(header.get("payload_bytes", -1))
    if payload_size < 0:
        raise ValueError("invalid payload size")
    return header, _recv_exact(sock, payload_size)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def _decode_image(payload: bytes) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--decode-image requires numpy and opencv-python") from exc
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("uploaded payload is not an image OpenCV can decode")


def run_server(args: argparse.Namespace) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen(args.backlog)
    print(f"listening on {args.bind}:{args.port}", flush=True)
    while True:
        conn, address = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"client connected: {address[0]}:{address[1]}", flush=True)
        try:
            with conn:
                while True:
                    receive_start_ns = time.perf_counter_ns()
                    header, payload = _recv_message(conn)
                    receive_done_ns = time.perf_counter_ns()
                    receive_done_wall_ns = time.time_ns()
                    request_type = header.get("type")
                    if request_type == "close":
                        break
                    process_start_ns = time.perf_counter_ns()
                    if request_type == "inference":
                        if args.decode_image:
                            _decode_image(payload)
                        if args.inference_ms:
                            time.sleep(args.inference_ms / 1000.0)
                    elif request_type != "clock":
                        raise ValueError(f"unknown request type: {request_type!r}")
                    process_done_ns = time.perf_counter_ns()
                    response_header = {
                        "type": request_type,
                        "sequence": int(header["sequence"]),
                        "server_receive_done_wall_ns": receive_done_wall_ns,
                        "server_receive_ms": (receive_done_ns - receive_start_ns) / 1e6,
                        "server_process_ms": (process_done_ns - process_start_ns) / 1e6,
                        "server_send_wall_ns": time.time_ns(),
                    }
                    response_payload = ACTION_BYTES if request_type == "inference" else b""
                    _send_message(conn, response_header, response_payload)
        except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"client {address[0]}:{address[1]} ended: {exc}", file=sys.stderr, flush=True)


def _one_request(
    sock: socket.socket, request_type: str, sequence: int, payload: bytes
) -> dict[str, float | int]:
    start_wall_ns = time.time_ns()
    start_ns = time.perf_counter_ns()
    _send_message(
        sock,
        {
            "type": request_type,
            "sequence": sequence,
            "client_send_wall_ns": start_wall_ns,
        },
        payload,
    )
    response, action = _recv_message(sock)
    done_ns = time.perf_counter_ns()
    done_wall_ns = time.time_ns()
    if int(response["sequence"]) != sequence:
        raise RuntimeError("response sequence mismatch")
    if request_type == "inference" and len(action) != len(ACTION_BYTES):
        raise RuntimeError("server returned an invalid action")
    server_receive_wall_ns = int(response["server_receive_done_wall_ns"])
    server_send_wall_ns = int(response["server_send_wall_ns"])
    # NTP four-timestamp offset estimate: server clock minus client clock.
    clock_offset_ns = (
        (server_receive_wall_ns - start_wall_ns)
        + (server_send_wall_ns - done_wall_ns)
    ) / 2.0
    return {
        "sequence": sequence,
        "payload_bytes": len(payload),
        "rtt_ms": (done_ns - start_ns) / 1e6,
        "server_receive_ms": float(response["server_receive_ms"]),
        "server_process_ms": float(response["server_process_ms"]),
        "clock_offset_ms": clock_offset_ns / 1e6,
        "upload_raw_clock_delta_ms": (server_receive_wall_ns - start_wall_ns) / 1e6,
        "download_raw_clock_delta_ms": (done_wall_ns - server_send_wall_ns) / 1e6,
    }


def _format_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label:27s} mean={stats['mean']:8.3f}  p50={stats['p50']:8.3f}  "
        f"p95={stats['p95']:8.3f}  min={stats['min']:8.3f}  max={stats['max']:8.3f} ms"
    )


def run_client(args: argparse.Namespace) -> None:
    if args.image:
        payload = args.image.read_bytes()
    else:
        # Content does not affect TCP transport latency; zero bytes avoid an
        # expensive random-data generation step contaminating the measurement.
        payload = bytes(args.payload_bytes)
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        clock_trials = [
            _one_request(sock, "clock", -(index + 1), b"")
            for index in range(args.clock_trials)
        ]
        # Low-RTT clock samples are least distorted by queueing. This offset is
        # diagnostic only: per-request one-way values remain estimates.
        best_clock_trials = sorted(clock_trials, key=lambda row: row["rtt_ms"])[
            : max(1, args.clock_trials // 4)
        ]
        clock_offset_ms = statistics.median(
            float(row["clock_offset_ms"]) for row in best_clock_trials
        )
        for index in range(args.warmup):
            _one_request(sock, "inference", -(1000 + index), payload)
        rows = [
            _one_request(sock, "inference", index, payload)
            for index in range(args.trials)
        ]
        _send_message(sock, {"type": "close", "sequence": args.trials}, b"")

    for row in rows:
        # Apply the independently calibrated offset rather than the current
        # request's offset. Using the current request would tautologically
        # split every RTT into two equal halves and hide link asymmetry.
        row["upload_estimate_ms"] = (
            float(row.pop("upload_raw_clock_delta_ms")) - clock_offset_ms
        )
        row["download_estimate_ms"] = (
            float(row.pop("download_raw_clock_delta_ms")) + clock_offset_ms
        )

    metrics = {
        key: _summary([float(row[key]) for row in rows])
        for key in (
            "rtt_ms",
            "server_receive_ms",
            "server_process_ms",
            "upload_estimate_ms",
            "download_estimate_ms",
        )
    }
    communication_ms = [
        float(row["rtt_ms"]) - float(row["server_process_ms"]) for row in rows
    ]
    metrics["communication_plus_io_ms"] = _summary(communication_ms)
    result = {
        "host": args.host,
        "port": args.port,
        "payload_bytes": len(payload),
        "trials": args.trials,
        "clock_offset_server_minus_client_ms": clock_offset_ms,
        "metrics_ms": metrics,
        "samples": rows,
        "notes": [
            "rtt_ms is the authoritative client-observed image-to-action latency.",
            "communication_plus_io_ms subtracts server processing but includes socket and scheduler overhead.",
            "upload/download are NTP-style estimates; do not treat them as exact without PTP/chrony clock validation.",
        ],
    }
    print(f"payload: {len(payload)} bytes, trials: {args.trials}")
    print(f"estimated server-client clock offset: {clock_offset_ms:.3f} ms")
    for key, label in (
        ("rtt_ms", "image -> action RTT"),
        ("server_process_ms", "server processing"),
        ("communication_plus_io_ms", "communication + socket I/O"),
        ("upload_estimate_ms", "upload (estimated)"),
        ("download_estimate_ms", "download (estimated)"),
    ):
        print(_format_stats(label, metrics[key]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    server = subparsers.add_parser("server", help="run on the inference server")
    server.add_argument("--bind", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--backlog", type=int, default=8)
    server.add_argument("--inference-ms", type=float, default=0.0)
    server.add_argument("--decode-image", action="store_true")
    server.set_defaults(func=run_server)

    client = subparsers.add_parser("client", help="run on the robot laptop")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, default=8765)
    payload = client.add_mutually_exclusive_group()
    payload.add_argument("--image", type=Path)
    payload.add_argument("--payload-bytes", type=int, default=100_000)
    client.add_argument("--trials", type=int, default=100)
    client.add_argument("--warmup", type=int, default=5)
    client.add_argument("--clock-trials", type=int, default=20)
    client.add_argument("--timeout", type=float, default=10.0)
    client.add_argument("--output", type=Path)
    client.set_defaults(func=run_client)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "trials", 1) < 1 or getattr(args, "clock_trials", 1) < 1:
        raise SystemExit("trials and clock-trials must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
