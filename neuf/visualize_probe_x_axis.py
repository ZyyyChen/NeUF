from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from neuf.dataset import Quat


DEFAULT_INFOS_PATH = Path("/home/zchen/Code/NeUF/data/simu_56/us/infos.json")
DEFAULT_OUTPUT_PATH = Path("/home/zchen/Code/NeUF/data/simu_56/us/probe_x_axis.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize probe X-axis directions from an infos.json file."
    )
    parser.add_argument(
        "--infos",
        type=Path,
        default=DEFAULT_INFOS_PATH,
        help="Path to infos.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--axis",
        choices=("local-x", "image-x"),
        default="local-x",
        help=(
            "Axis to visualize. local-x uses rotmat[:, 0], matching the probe local X axis. "
            "image-x uses rotmat[:, 1], matching the image X coordinate in utils.get_oriented_points_and_views()."
        ),
    )
    parser.add_argument(
        "--reverse-quat",
        action="store_true",
        help="Match Dataset(reverse_quat=True) quaternion convention.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Draw one frame every N frames.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=2.0,
        help="Arrow length in millimeters.",
    )
    return parser.parse_args()


def parse_quaternion(frame: dict, reverse_quat: bool) -> Quat:
    if reverse_quat:
        return Quat(
            float(frame["w3"]),
            float(frame["w0"]),
            float(frame["w1"]),
            float(frame["w2"]),
        )

    return Quat(
        float(frame["w0"]),
        float(frame["w1"]),
        float(frame["w2"]),
        float(frame["w3"]),
    )


def load_positions_and_directions(
    infos_path: Path,
    axis: str,
    reverse_quat: bool,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if stride <= 0:
        raise ValueError(f"stride must be >= 1, got {stride}")

    infos_json = json.loads(infos_path.read_text(encoding="utf-8"))
    frame_keys = sorted((key for key in infos_json if key != "infos"), key=lambda key: int(key))
    frame_keys = frame_keys[::stride]
    if not frame_keys:
        raise ValueError(f"No frames found in {infos_path}")

    axis_column = 0 if axis == "local-x" else 1
    positions = []
    directions = []

    for frame_key in frame_keys:
        frame = infos_json[frame_key]
        position = np.array(
            [float(frame["x"]), float(frame["y"]), float(frame["z"])],
            dtype=np.float32,
        )
        rotation = parse_quaternion(frame, reverse_quat)
        direction = np.asarray(rotation.as_rotmat()[:, axis_column], dtype=np.float32)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm

        positions.append(position)
        directions.append(direction)

    return np.stack(positions), np.stack(directions), frame_keys


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float32)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_probe_x_axis(
    positions: np.ndarray,
    directions: np.ndarray,
    frame_keys: list[str],
    output_path: Path,
    axis: str,
    arrow_length: float,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="0.35",
        linewidth=1.2,
        marker="o",
        markersize=2.5,
        label="probe positions",
    )
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=arrow_length,
        normalize=True,
        color="crimson",
        linewidth=1.0,
        label=f"{axis} direction",
    )

    first = positions[0]
    last = positions[-1]
    ax.text(first[0], first[1], first[2], f"start {frame_keys[0]}", fontsize=8)
    ax.text(last[0], last[1], last[2], f"end {frame_keys[-1]}", fontsize=8)

    ax.set_title(f"Probe {axis} direction from infos.json")
    ax.set_xlabel("World X (mm)")
    ax.set_ylabel("World Y (mm)")
    ax.set_zlabel("World Z (mm)")
    ax.legend(loc="upper left")
    set_axes_equal(ax)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    positions, directions, frame_keys = load_positions_and_directions(
        args.infos,
        args.axis,
        args.reverse_quat,
        args.stride,
    )
    plot_probe_x_axis(
        positions,
        directions,
        frame_keys,
        args.output,
        args.axis,
        args.arrow_length,
    )

    mean_direction = directions.mean(axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    print(f"Loaded {len(frame_keys)} frames from {args.infos}")
    print(f"Mean {args.axis} direction: {mean_direction.tolist()}")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
