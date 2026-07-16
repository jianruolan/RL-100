import argparse
import os
from pathlib import Path

import cv2
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from realsense import RealSense


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = depth > 0
    if np.any(valid):
        lo = np.percentile(depth[valid], 5)
        hi = np.percentile(depth[valid], 95)
        if hi <= lo:
            hi = lo + 1.0
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def save_preview(frame: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    color_bgr = frame["color"]
    depth = frame["depth"]
    pc = frame["point_cloud"]

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_color = depth_to_colormap(depth)
    depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    rgb_path = out_dir / "rgb.png"
    depth_path = out_dir / "depth_colormap.png"
    pc_path = out_dir / "point_cloud.npy"
    fig_path = out_dir / "preview.png"

    cv2.imwrite(str(rgb_path), color_bgr)
    cv2.imwrite(str(depth_path), depth_color)
    np.save(pc_path, pc)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(color_rgb)
    ax1.set_title("RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(depth_color_rgb)
    ax2.set_title("Depth colormap")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    if pc is not None and len(pc) > 0:
        sample = pc if len(pc) <= 2000 else pc[np.linspace(0, len(pc) - 1, 2000, dtype=np.int64)]
        ax3.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
        ax3.set_xlim(np.percentile(sample[:, 0], [1, 99]))
        ax3.set_ylim(np.percentile(sample[:, 1], [1, 99]))
        ax3.set_zlim(np.percentile(sample[:, 2], [1, 99]))
    ax3.set_title("Point cloud")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.text(
        0.02,
        0.98,
        f"timestamp: {frame['timestamp']:.3f}\n"
        f"color: {color_bgr.shape}\n"
        f"depth: {depth.shape}\n"
        f"pc: {None if pc is None else pc.shape}",
        va="top",
        family="monospace",
    )
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    return {
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "pc_path": pc_path,
        "fig_path": fig_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/realsense_preview"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pc", action="store_true", help="only grab RGB-D, skip point cloud generation")
    args = parser.parse_args()

    print("[1/4] initialize camera", flush=True)
    cam = RealSense(num_points=args.num_points)
    print("[2/4] start camera", flush=True)
    try:
        cam.start()
    except Exception as e:
        print(f"[error] camera start failed: {type(e).__name__}: {e}", flush=True)
        raise
    try:
        print("[3/4] capture frame", flush=True)
        frame = cam.get_frame(require_pc=not args.no_pc)
    finally:
        print("[4/4] stop camera", flush=True)
        cam.stop()

    print("[5/5] save preview files", flush=True)
    paths = save_preview(frame, args.out_dir)
    print("saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    if args.show and os.environ.get("DISPLAY"):
        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(frame["color"], cv2.COLOR_BGR2RGB))
        plt.title("RGB")
        plt.axis("off")
        plt.show()
    elif args.show:
        print("DISPLAY is not set; skipped interactive show. Open preview.png instead.", flush=True)


if __name__ == "__main__":
    main()
