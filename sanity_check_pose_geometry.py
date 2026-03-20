import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the notebook-expected probe plane and the current dataset plane "
            "for one or more frames from infos.json."
        )
    )
    parser.add_argument(
        "--infos",
        type=Path,
        default=Path(r"D:\0-Code\NeUF\data\cerebral_data\Pre_traitement_echo_v2\Recalage\Patient0\us_recal_original\infos.json"),
        help="Path to infos.json",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="first,mid,last",
        help="Comma-separated frame list. Supports integers and aliases: first, mid, last.",
    )
    parser.add_argument(
        "--extent-source",
        choices=("px", "scan_mm", "px_size_cm"),
        default="px",
        help=(
            "How to interpret image width/depth when building the comparison plane. "
            "'px' matches the current notebook-generated pose units."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to <infos_dir>/sanity_check_pose_geometry.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the matplotlib window in addition to saving the figure.",
    )
    return parser.parse_args()


def quat_to_rotmat(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def shortest_angle_deg(a, b):
    theta = angle_deg(a, b)
    return min(theta, 180.0 - theta)


def resolve_frame_indices(frame_expr, frame_keys):
    aliases = {
        "first": 0,
        "mid": len(frame_keys) // 2,
        "last": len(frame_keys) - 1,
    }
    indices = []
    for token in (item.strip().lower() for item in frame_expr.split(",")):
        if not token:
            continue
        if token in aliases:
            indices.append(aliases[token])
        else:
            indices.append(int(token))
    return [frame_keys[idx] for idx in indices]


def get_plane_dims(infos_meta, extent_source):
    roi = infos_meta.get("ROI")
    if roi and roi.get("width", 0) > 0 and roi.get("height", 0) > 0:
        width_px = int(roi["width"])
        height_px = int(roi["height"])
        offset_x_px = float(roi["x"])
        offset_y_px = float(roi["y"])
    else:
        width_px = int(infos_meta["scan_dims_px"]["width"])
        height_px = int(infos_meta["scan_dims_px"]["depth"])
        offset_x_px = 0.0
        offset_y_px = 0.0

    if extent_source == "px":
        width = float(width_px)
        height = float(height_px)
        offset_x = offset_x_px
        offset_y = offset_y_px
        unit_label = "px-like units"
    elif extent_source == "scan_mm":
        width = float(infos_meta["scan_dims_mm"]["width"])
        height = float(infos_meta["scan_dims_mm"]["depth"])
        scale_x = width / max(width_px, 1)
        scale_y = height / max(height_px, 1)
        offset_x = offset_x_px * scale_x
        offset_y = offset_y_px * scale_y
        unit_label = "scan_dims_mm units"
    else:
        px_size_cm = infos_meta["px_size_cm"]
        scale_x = float(px_size_cm["width"]) * 10.0
        scale_y = float(px_size_cm["height"]) * 10.0
        width = width_px * scale_x
        height = height_px * scale_y
        offset_x = offset_x_px * scale_x
        offset_y = offset_y_px * scale_y
        unit_label = "px_size_cm-derived mm"

    return {
        "width_px": width_px,
        "height_px": height_px,
        "width": width,
        "height": height,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "unit_label": unit_label,
    }


def build_notebook_expected_local_corners(width, height):
    half_z = width * 0.5
    return np.array(
        [
            [0.0, 0.0, -half_z],
            [0.0, height, -half_z],
            [0.0, height, half_z],
            [0.0, 0.0, half_z],
        ],
        dtype=np.float64,
    )


def build_dataset_actual_local_corners(width, height, offset_x, offset_y, width_px, height_px):
    pixel_size_w = width / max(width_px, 1)
    pixel_size_h = height / max(height_px, 1)

    start_x = -width * 0.5 + pixel_size_w * 0.5 + offset_x
    start_y = pixel_size_h * 0.5 + offset_y

    x_left = start_x
    x_right = start_x + (width_px - 1) * pixel_size_w
    y_top = start_y
    y_bottom = start_y + (height_px - 1) * pixel_size_h

    # Current utils.py logic:
    # local_points = (Y_base, 0, X_base)
    return np.array(
        [
            [y_top, 0.0, x_left],
            [y_bottom, 0.0, x_left],
            [y_bottom, 0.0, x_right],
            [y_top, 0.0, x_right],
        ],
        dtype=np.float64,
    )


def transform_points(local_points, rotmat, origin):
    return local_points @ rotmat.T + origin[None, :]


def make_poly(ax, corners, facecolor, edgecolor, label):
    poly = Poly3DCollection([corners], alpha=0.26, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8)
    poly.set_label(label)
    ax.add_collection3d(poly)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=edgecolor, linewidth=1.4)


def add_arrow(ax, origin, vec, color, label, linestyle="-"):
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vec[0],
        vec[1],
        vec[2],
        color=color,
        linewidth=2.2,
        linestyle=linestyle,
        arrow_length_ratio=0.12,
        label=label,
    )


def set_equal_axes(ax, points, margin=0.08):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    ranges = np.ptp(points, axis=0)
    half = ranges.max() * 0.5
    if half == 0:
        half = 1.0
    half *= 1.0 + margin
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def get_plane_vectors(world_corners):
    edge01 = normalize(world_corners[1] - world_corners[0])
    edge03 = normalize(world_corners[3] - world_corners[0])
    normal = normalize(np.cross(edge01, edge03))
    return edge01, edge03, normal


def get_combo_definitions():
    return [
        {
            "name": "Runtime/reference: notebook basis + raw quat",
            "short": "run/ref",
            "basis": "notebook",
            "quat_mode": "raw",
            "viewdir_mode": "notebook_x",
            "color": "tab:green",
            "is_reference": True,
        },
        {
            "name": "Notebook basis + inverse quat",
            "short": "nb+invq",
            "basis": "notebook",
            "quat_mode": "inverse",
            "viewdir_mode": "notebook_x",
            "color": "tab:orange",
            "is_reference": False,
        },
        {
            "name": "Legacy current basis + raw quat",
            "short": "cur+rawq",
            "basis": "current",
            "quat_mode": "raw",
            "viewdir_mode": "dataset_neg_x",
            "color": "tab:blue",
            "is_reference": False,
        },
        {
            "name": "Legacy current basis + inverse quat",
            "short": "cur+invq",
            "basis": "current",
            "quat_mode": "inverse",
            "viewdir_mode": "dataset_neg_x",
            "color": "tab:red",
            "is_reference": False,
        },
    ]


def get_world_viewdir(rotmat, viewdir_mode):
    if viewdir_mode == "notebook_x":
        local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif viewdir_mode == "dataset_neg_x":
        local = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown viewdir mode: {viewdir_mode}")
    return normalize(rotmat @ local)


def describe_frame(frame_key, origin, combo_results):
    reference = next(item for item in combo_results if item["definition"]["is_reference"])
    ref_edge01, ref_edge03, ref_normal = reference["plane_vectors"]

    print(f"\nFrame {frame_key}")
    print(f"  origin: {np.round(origin, 6)}")
    print(f"  reference normal: {np.round(ref_normal, 6)}")
    print("  combos ranked by mean(normal, edge01, edge03) angular error:")

    ranked = sorted(
        combo_results,
        key=lambda item: (
            item["metrics"]["mean_plane_error_deg"],
            item["metrics"]["viewdir_vs_plane_normal_deg"],
        ),
    )
    for item in ranked:
        d = item["definition"]
        m = item["metrics"]
        plane_edge01, plane_edge03, plane_normal = item["plane_vectors"]
        print(
            "   "
            f"{d['short']:<8} | "
            f"normal={m['normal_vs_ref_deg']:.3f} deg, "
            f"edge01={m['edge01_vs_ref_deg']:.3f} deg, "
            f"edge03={m['edge03_vs_ref_deg']:.3f} deg, "
            f"mean={m['mean_plane_error_deg']:.3f} deg, "
            f"viewdir_vs_normal={m['viewdir_vs_plane_normal_deg']:.3f} deg"
        )
        print(
            " " * 14
            + f"plane_normal={np.round(plane_normal, 6)} "
            + f"viewdir={np.round(item['viewdir_world'], 6)}"
        )


def main():
    args = parse_args()
    infos = json.loads(args.infos.read_text(encoding="utf-8"))
    infos_meta = infos["infos"]
    frame_keys = sorted((k for k in infos.keys() if k != "infos"), key=lambda k: int(k))
    selected_keys = resolve_frame_indices(args.frames, frame_keys)

    dims = get_plane_dims(infos_meta, args.extent_source)
    notebook_local = build_notebook_expected_local_corners(dims["width"], dims["height"])
    dataset_local = build_dataset_actual_local_corners(
        dims["width"],
        dims["height"],
        dims["offset_x"],
        dims["offset_y"],
        dims["width_px"],
        dims["height_px"],
    )

    print("Comparison assumptions")
    print(f"  infos: {args.infos}")
    print(f"  extent source: {args.extent_source} ({dims['unit_label']})")
    print(f"  plane width x height: {dims['width']:.6f} x {dims['height']:.6f}")
    print(f"  ROI offset: x={dims['offset_x']:.6f}, y={dims['offset_y']:.6f}")
    print("  runtime/reference: notebook YZ plane with Y starting at the probe origin + raw quaternion")
    print("  legacy code path: utils.py local_points = (Y, 0, X) and dataset.py loads (w0, -w1, -w2, -w3)")


    combo_defs = get_combo_definitions()
    basis_lookup = {
        "notebook": notebook_local,
        "current": dataset_local,
    }

    n_rows = len(selected_keys)
    n_cols = len(combo_defs)
    fig = plt.figure(figsize=(4.8 * n_cols, 4.8 * n_rows))
    axes = []
    for row in range(n_rows):
        row_axes = []
        for col in range(n_cols):
            row_axes.append(fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection="3d"))
        axes.append(row_axes)

    for row_axes, frame_key in zip(axes, selected_keys):
        frame = infos[frame_key]
        origin = np.array([frame["x"], frame["y"], frame["z"]], dtype=np.float64)

        quat_file = np.array([frame["w0"], frame["w1"], frame["w2"], frame["w3"]], dtype=np.float64)
        quat_inverse = np.array([frame["w0"], -frame["w1"], -frame["w2"], -frame["w3"]], dtype=np.float64)

        quat_lookup = {
            "raw": quat_to_rotmat(quat_file),
            "inverse": quat_to_rotmat(quat_inverse),
        }

        combo_results = []
        for combo_def in combo_defs:
            rotmat = quat_lookup[combo_def["quat_mode"]]
            world_corners = transform_points(basis_lookup[combo_def["basis"]], rotmat, origin)
            edge01, edge03, normal = get_plane_vectors(world_corners)
            viewdir_world = get_world_viewdir(rotmat, combo_def["viewdir_mode"])
            combo_results.append(
                {
                    "definition": combo_def,
                    "rotmat": rotmat,
                    "world_corners": world_corners,
                    "plane_vectors": (edge01, edge03, normal),
                    "viewdir_world": viewdir_world,
                }
            )

        reference = next(item for item in combo_results if item["definition"]["is_reference"])
        ref_edge01, ref_edge03, ref_normal = reference["plane_vectors"]

        scale = max(dims["width"], dims["height"]) * 0.22
        for item in combo_results:
            edge01, edge03, normal = item["plane_vectors"]
            item["metrics"] = {
                "normal_vs_ref_deg": shortest_angle_deg(ref_normal, normal),
                "edge01_vs_ref_deg": shortest_angle_deg(ref_edge01, edge01),
                "edge03_vs_ref_deg": shortest_angle_deg(ref_edge03, edge03),
                "mean_plane_error_deg": float(
                    np.mean(
                        [
                            shortest_angle_deg(ref_normal, normal),
                            shortest_angle_deg(ref_edge01, edge01),
                            shortest_angle_deg(ref_edge03, edge03),
                        ]
                    )
                ),
                "viewdir_vs_plane_normal_deg": shortest_angle_deg(item["viewdir_world"], normal),
            }

        describe_frame(frame_key, origin, combo_results)

        for ax, item in zip(row_axes, combo_results):
            combo_def = item["definition"]
            metrics = item["metrics"]
            edge01, edge03, normal = item["plane_vectors"]

            make_poly(ax, reference["world_corners"], facecolor="lightgray", edgecolor="dimgray", label="Reference plane")
            make_poly(
                ax,
                item["world_corners"],
                facecolor=combo_def["color"],
                edgecolor=combo_def["color"],
                label=combo_def["name"],
            )

            add_arrow(ax, origin, ref_normal * scale, color="dimgray", label="Reference normal (+X)")
            add_arrow(ax, origin, normal * scale, color=combo_def["color"], label="Combo plane normal")
            add_arrow(ax, origin, item["viewdir_world"] * scale, color="tab:purple", label="Combo viewdir", linestyle="--")
            ax.scatter(origin[0], origin[1], origin[2], color="black", s=18, label="Pose origin")

            points_all = np.vstack(
                [
                    reference["world_corners"],
                    item["world_corners"],
                    origin[None, :],
                    origin[None, :] + ref_normal[None, :] * scale,
                    origin[None, :] + normal[None, :] * scale,
                    origin[None, :] + item["viewdir_world"][None, :] * scale,
                ]
            )
            set_equal_axes(ax, points_all)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(
                f"Frame {frame_key} | {combo_def['short']}\n"
                f"N {metrics['normal_vs_ref_deg']:.1f}°  "
                f"E1 {metrics['edge01_vs_ref_deg']:.1f}°  "
                f"E3 {metrics['edge03_vs_ref_deg']:.1f}°\n"
                f"V/N {metrics['viewdir_vs_plane_normal_deg']:.1f}°"
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.suptitle("Pose geometry sanity check: basis choice x quaternion interpretation", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = args.output if args.output is not None else args.infos.parent / "sanity_check_pose_geometry.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"\nSaved figure to: {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
