"""Plot a probe trajectory from estimated translation and rotation parameters.

This file is standalone and can be copied outside the NeUF repository.
It only requires NumPy, Matplotlib, and Pillow (for GIF output).

Example:
    python plot_probe_trajectory.py --frames 242 --output-dir trajectory_output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class TrajectoryParameters:
    """Inputs required by the trajectory model (pixels and radians)."""

    frames: int = 242
    image_width: float = 944.0
    image_height: float = 708.0
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    velocity_x: float = -0.08
    velocity_y: float = 0.0
    velocity_z: float = -0.0329759687
    theta: float = -0.411082084
    omega: float = 0.00489753927
    probe_offset: float = 28.1915615
    plot_scale: float = 4.0
    arrow_length: float = 40.0
    rectangle_y: float = 25.0
    rectangle_z: float = 5.0


@dataclass(frozen=True)
class Trajectories:
    center: np.ndarray
    probe: np.ndarray
    image_end: np.ndarray


def local_to_world(points: np.ndarray) -> np.ndarray:
    """Convert notebook coordinates using [x, y, z] -> [-y, z, -x]."""

    return np.stack((-points[:, 1], points[:, 2], -points[:, 0]), axis=1)


def calculate_trajectories(parameters: TrajectoryParameters) -> Trajectories:
    """Calculate the center, probe, and image-end position for every frame."""

    if parameters.frames < 1:
        raise ValueError("frames must be at least 1")
    if parameters.image_width <= 0 or parameters.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if parameters.plot_scale <= 0:
        raise ValueError("plot_scale must be positive")

    frame = np.arange(parameters.frames, dtype=np.float64)
    center_initial = np.array(
        [parameters.center_x, parameters.center_y, parameters.center_z]
    )
    center_velocity = np.array(
        [parameters.velocity_x, parameters.velocity_y, parameters.velocity_z]
    )
    center_local = center_initial + frame[:, None] * center_velocity

    angle = parameters.theta + frame * parameters.omega
    radial_direction = np.stack(
        (np.cos(angle), np.zeros_like(angle), np.sin(angle)), axis=1
    )
    probe_local = center_local + parameters.probe_offset * radial_direction
    image_end_local = center_local + (
        parameters.probe_offset
        + parameters.image_height / parameters.plot_scale
    ) * radial_direction

    return Trajectories(
        center=local_to_world(center_local),
        probe=local_to_world(probe_local),
        image_end=local_to_world(image_end_local),
    )


def image_rectangle(
    trajectories: Trajectories,
    frame_index: int,
    half_width: float,
) -> np.ndarray:
    """Return the four corners of the ultrasound plane at one frame."""

    top_left = trajectories.image_end[frame_index].copy()
    top_right = trajectories.image_end[frame_index].copy()
    bottom_right = trajectories.probe[frame_index].copy()
    bottom_left = trajectories.probe[frame_index].copy()
    top_left[0] += half_width
    top_right[0] -= half_width
    bottom_right[0] -= half_width
    bottom_left[0] += half_width
    return np.array([top_left, top_right, bottom_right, bottom_left])


def plot_static_3d(
    trajectories: Trajectories,
    parameters: TrajectoryParameters,
    output_path: Path,
) -> None:
    """Save a static 3-D overview with the first, middle, and last image planes."""

    half_width = parameters.image_width / (2.0 * parameters.plot_scale)
    frame_indices = sorted({0, parameters.frames // 2, parameters.frames - 1})
    rectangles = [
        image_rectangle(trajectories, frame_index, half_width)
        for frame_index in frame_indices
    ]
    all_points = np.vstack(
        [
            trajectories.center,
            trajectories.probe,
            trajectories.image_end,
            np.asarray(rectangles).reshape(-1, 3),
        ]
    )
    point_min = all_points.min(axis=0)
    point_max = all_points.max(axis=0)
    axis_span = point_max - point_min
    margin = max(5.0, 0.05 * axis_span.max())

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    styles = (
        (trajectories.image_end, "#66AC94", "-", "image end"),
        (trajectories.probe, "#457B9D", "-", "probe"),
        (trajectories.center, "#DDA520", "--", "center"),
    )
    for points, color, linestyle, label in styles:
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=2.8,
            linestyle=linestyle,
            label=label,
        )
        ax.scatter(*points[0], color=color, marker="o", s=35)
        ax.scatter(*points[-1], color=color, marker="^", s=50)

    rectangle_colors = ("#C2DDD3", "#B5D0E2", "#F0D98F")
    for frame_index, rectangle, color in zip(
        frame_indices, rectangles, rectangle_colors
    ):
        ax.add_collection3d(
            Poly3DCollection(
                [rectangle], facecolor=color, edgecolor=color, alpha=0.18
            )
        )
        ax.plot([], [], [], color=color, label=f"image plane {frame_index}")

    ax.set_xlim(point_min[0] - margin, point_max[0] + margin)
    ax.set_ylim(point_min[1] - margin, point_max[1] + margin)
    ax.set_zlim(point_min[2] - margin, point_max[2] + margin)
    ax.set_box_aspect(np.where(axis_span > 0, axis_span, 1.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated probe trajectory")
    ax.view_init(elev=28, azim=35)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_yz_projection(trajectories: Trajectories, output_path: Path) -> None:
    """Save the YZ projection of the probe and center trajectories."""

    probe_yz = trajectories.probe[:, [1, 2]]
    center_yz = trajectories.center[:, [1, 2]]
    all_points = np.vstack((probe_yz, center_yz))
    point_min = all_points.min(axis=0)
    point_max = all_points.max(axis=0)
    center = (point_min + point_max) * 0.5
    half_range = max(5.0, 0.55 * (point_max - point_min).max())

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        probe_yz[:, 0],
        probe_yz[:, 1],
        color="#457B9D",
        linewidth=4.2,
        label="probe",
    )
    ax.plot(
        center_yz[:, 0],
        center_yz[:, 1],
        color="#DDA520",
        linewidth=4.2,
        linestyle="--",
        label="center",
    )
    ax.scatter(*probe_yz[0], color="#457B9D", marker="o", s=35)
    ax.scatter(*probe_yz[-1], color="#457B9D", marker="^", s=110)
    ax.scatter(*center_yz[0], color="#DDA520", marker="o", s=35)
    ax.scatter(*center_yz[-1], color="#DDA520", marker="^", s=110)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("Estimated trajectory — YZ projection")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def calculate_rotation_matrices(parameters: TrajectoryParameters) -> np.ndarray:
    """Build the per-frame pose rotations used by the original GIF code."""

    angles = parameters.theta + np.arange(parameters.frames) * parameters.omega
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotation_x = np.zeros((parameters.frames, 3, 3), dtype=np.float64)
    rotation_x[:, 0, 0] = 1.0
    rotation_x[:, 1, 1] = cosine
    rotation_x[:, 1, 2] = -sine
    rotation_x[:, 2, 1] = sine
    rotation_x[:, 2, 2] = cosine

    canonical = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
    )
    return rotation_x @ canonical


def save_pose_animation(
    trajectories: Trajectories,
    parameters: TrajectoryParameters,
    output_path: Path,
    fps: int,
) -> None:
    """Save the pose-and-trajectory GIF from the supplied reference code."""

    if fps < 1:
        raise ValueError("fps must be at least 1")
    if parameters.arrow_length <= 0:
        raise ValueError("arrow_length must be positive")
    if parameters.rectangle_y <= 0 or parameters.rectangle_z <= 0:
        raise ValueError("rectangle dimensions must be positive")

    # ``positions`` corresponds to the positions read from infos.json in the
    # reference snippet. The notebook writes trajectories.probe to that file.
    positions = trajectories.probe
    rotations = calculate_rotation_matrices(parameters)
    colors = plt.colormaps["rainbow"](
        np.linspace(0.0, 1.0, parameters.frames)
    )

    probe_trajectory = (
        positions + rotations[:, :, 0] * parameters.arrow_length
    )
    center_trajectory = (
        positions + rotations[:, :, 0] * (parameters.rectangle_y * 0.5)
    )
    all_points = np.vstack((positions, probe_trajectory, center_trajectory))
    point_min = all_points.min(axis=0)
    point_max = all_points.max(axis=0)
    margin = max(30.0, parameters.rectangle_z)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int):
        ax.clear()
        rotation = rotations[frame_index]
        color = colors[frame_index]
        probe_point = probe_trajectory[frame_index]
        center_point = center_trajectory[frame_index]

        # Full trajectories in pale colors.
        ax.plot(
            probe_trajectory[:, 0],
            probe_trajectory[:, 1],
            probe_trajectory[:, 2],
            color="lightsteelblue",
            linewidth=1.5,
            alpha=0.9,
            label="Probe trajectory",
        )
        ax.plot(
            center_trajectory[:, 0],
            center_trajectory[:, 1],
            center_trajectory[:, 2],
            color="khaki",
            linewidth=1.5,
            alpha=0.9,
            linestyle="--",
            label="Center trajectory",
        )

        # Highlight the trajectories up to the current frame.
        stop = frame_index + 1
        ax.plot(
            probe_trajectory[:stop, 0],
            probe_trajectory[:stop, 1],
            probe_trajectory[:stop, 2],
            color="tab:blue",
            linewidth=2.5,
            label="Current probe",
        )
        ax.plot(
            center_trajectory[:stop, 0],
            center_trajectory[:stop, 1],
            center_trajectory[:stop, 2],
            color="goldenrod",
            linewidth=2.5,
            linestyle="--",
            label="Current center",
        )

        ax.scatter(*probe_point, color="tab:blue", s=45, edgecolor="k")
        ax.scatter(*center_point, color="goldenrod", s=45, edgecolor="k")
        ax.plot(
            [probe_point[0], center_point[0]],
            [probe_point[1], center_point[1]],
            [probe_point[2], center_point[2]],
            color="gray",
            linewidth=1.8,
            alpha=0.9,
        )

        # Current local X axis.
        vector_x = rotation[:, 0]
        ax.quiver(
            *probe_point,
            *vector_x,
            color=color,
            length=parameters.arrow_length,
            normalize=True,
            linewidth=2,
        )

        # Current local YZ rectangle centered at the probe point.
        vector_y = rotation[:, 1]
        vector_z = rotation[:, 2]
        half_y = parameters.rectangle_y * 0.5
        half_z = parameters.rectangle_z * 0.5
        rectangle = np.array(
            [
                probe_point - vector_y * half_y - vector_z * half_z,
                probe_point + vector_y * half_y - vector_z * half_z,
                probe_point + vector_y * half_y + vector_z * half_z,
                probe_point - vector_y * half_y + vector_z * half_z,
            ]
        )
        ax.add_collection3d(
            Poly3DCollection(
                [rectangle], alpha=0.35, facecolor=color, edgecolor=color
            )
        )

        ax.view_init(elev=30, azim=30)
        ax.set_xlim(point_min[0] - margin, point_max[0] + margin)
        ax.set_ylim(point_min[1] - margin, point_max[1] + margin)
        ax.set_zlim(point_min[2] - margin, point_max[2] + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Step {frame_index}: Probe / Position / Center"
        )
        ax.legend(loc="upper right")
        return ()

    animation = FuncAnimation(
        fig, update, frames=parameters.frames, interval=1000 / fps, blit=False
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw static and animated trajectories from estimated parameters."
    )
    parser.add_argument("--frames", type=int, default=242)
    parser.add_argument("--image-width", type=float, default=944.0)
    parser.add_argument("--image-height", type=float, default=708.0)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=0.0)
    parser.add_argument("--velocity-x", type=float, default=-0.08)
    parser.add_argument("--velocity-y", type=float, default=0.0)
    parser.add_argument("--velocity-z", type=float, default=-0.0329759687)
    parser.add_argument("--theta", type=float, default=-0.411082084)
    parser.add_argument("--omega", type=float, default=0.00489753927)
    parser.add_argument("--probe-offset", type=float, default=28.1915615)
    parser.add_argument("--plot-scale", type=float, default=4.0)
    parser.add_argument("--arrow-length", type=float, default=40.0)
    parser.add_argument("--rectangle-y", type=float, default=25.0)
    parser.add_argument("--rectangle-z", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("trajectory_output"))
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parameters = TrajectoryParameters(
        frames=args.frames,
        image_width=args.image_width,
        image_height=args.image_height,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        velocity_x=args.velocity_x,
        velocity_y=args.velocity_y,
        velocity_z=args.velocity_z,
        theta=args.theta,
        omega=args.omega,
        probe_offset=args.probe_offset,
        plot_scale=args.plot_scale,
        arrow_length=args.arrow_length,
        rectangle_y=args.rectangle_y,
        rectangle_z=args.rectangle_z,
    )
    trajectories = calculate_trajectories(parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_path = args.output_dir / "probe_trajectory_3d.png"
    yz_path = args.output_dir / "probe_trajectory_yz.png"
    plot_static_3d(trajectories, parameters, static_path)
    plot_yz_projection(trajectories, yz_path)
    print(f"Saved: {static_path}")
    print(f"Saved: {yz_path}")

    if not args.no_gif:
        gif_path = args.output_dir / "probe_pose_with_trajectory.gif"
        save_pose_animation(trajectories, parameters, gif_path, args.fps)
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
