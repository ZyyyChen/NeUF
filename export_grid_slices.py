import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nerf_network import NeRF


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a NeUF checkpoint on an absolute 3D grid and save grid-aligned slices."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to a NeUF .pkl checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the rendered volume and slices will be saved.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.0134468004107475,
        help="Voxel spacing in mm for x/y/z.",
    )
    parser.add_argument(
        "--point-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
        help="Absolute world-coordinate grid start point. Defaults to checkpoint bounding box min.",
    )
    parser.add_argument(
        "--point-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
        help="Absolute world-coordinate grid max point. Defaults to checkpoint bounding box max.",
    )
    parser.add_argument(
        "--slice-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis along which slices are exported.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
        help="Number of grid points queried per forward pass.",
    )
    parser.add_argument(
        "--save-raw-npy",
        action="store_true",
        help="Also save the raw float32 volume as volume.npy.",
    )
    return parser.parse_args()


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = NeRF(ckpt).to(DEVICE)
    model.eval()
    return ckpt, model


def build_axis_coords(point_min: np.ndarray, point_max: np.ndarray, spacing: float):
    coords = []
    for dim in range(3):
        extent = point_max[dim] - point_min[dim]
        n = max(1, int(np.ceil(extent / spacing)))
        axis = point_min[dim] + (np.arange(n, dtype=np.float32) + 0.5) * spacing
        coords.append(axis)
    return coords


def build_volume_points(x_coords, y_coords, z_coords):
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1)
    return points


def query_volume(model: NeRF, ckpt: dict, points_xyz: np.ndarray, chunk_size: int):
    flat_points = points_xyz.reshape(-1, 3)
    bbox_min = ckpt["bounding_box"][0].detach().cpu().numpy().astype(np.float32)
    bbox_max = ckpt["bounding_box"][1].detach().cpu().numpy().astype(np.float32)

    inside_mask = np.all((flat_points >= bbox_min) & (flat_points <= bbox_max), axis=1)
    output = np.zeros((flat_points.shape[0],), dtype=np.float32)

    if not np.any(inside_mask):
        return output.reshape(points_xyz.shape[:3])

    inside_points = flat_points[inside_mask]
    inside_dirs = np.zeros_like(inside_points, dtype=np.float32)

    if model.use_direction:
        # Absolute grid sampling is not probe-based, so directions are undefined.
        # Use a constant zero view direction to keep inference deterministic.
        inside_dirs[:] = 0.0

    with torch.no_grad():
        for start in tqdm(
            range(0, inside_points.shape[0], chunk_size),
            desc="Querying grid",
        ):
            stop = min(start + chunk_size, inside_points.shape[0])
            batch_points = torch.from_numpy(inside_points[start:stop]).to(DEVICE)
            batch_dirs = torch.from_numpy(inside_dirs[start:stop]).to(DEVICE)

            if model.encoding_type != "HASH" or not model.use_encoding:
                bbox_min_dev = torch.from_numpy(bbox_min).to(DEVICE)
                bbox_size_dev = torch.from_numpy(bbox_max - bbox_min).to(DEVICE)
                batch_points = ((batch_points - bbox_min_dev) / bbox_size_dev) * 2.0 - 1.0

            batch_values = model.query(batch_points, batch_dirs).reshape(-1)
            output[np.flatnonzero(inside_mask)[start:stop]] = (
                batch_values.detach().cpu().numpy().astype(np.float32)
            )

    return output.reshape(points_xyz.shape[:3])


def normalize_volume_for_png(volume: np.ndarray):
    volume = volume.astype(np.float32, copy=False)
    inside = volume > 0
    if not np.any(inside):
        return np.zeros_like(volume, dtype=np.uint8)

    vmax = np.max(volume[inside])
    vmin = np.min(volume[inside])
    if vmax <= 1.5 and vmin >= -0.5:
        scaled = np.clip(volume * 255.0, 0.0, 255.0)
    else:
        scaled = np.clip(volume, 0.0, 255.0)
    scaled[~inside] = 0.0
    return scaled.astype(np.uint8)


def iter_slices(volume: np.ndarray, axis: str):
    if axis == "z":
        for idx in range(volume.shape[0]):
            yield idx, volume[idx, :, :]
    elif axis == "y":
        for idx in range(volume.shape[1]):
            yield idx, volume[:, idx, :]
    else:
        for idx in range(volume.shape[2]):
            yield idx, volume[:, :, idx]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt, model = load_model(args.ckpt)
    ckpt_min = ckpt["bounding_box"][0].detach().cpu().numpy().astype(np.float32)
    ckpt_max = ckpt["bounding_box"][1].detach().cpu().numpy().astype(np.float32)

    point_min = ckpt_min if args.point_min is None else np.array(args.point_min, dtype=np.float32)
    point_max = ckpt_max if args.point_max is None else np.array(args.point_max, dtype=np.float32)

    x_coords, y_coords, z_coords = build_axis_coords(point_min, point_max, args.spacing)
    points_xyz = build_volume_points(x_coords, y_coords, z_coords)
    volume = query_volume(model, ckpt, points_xyz, args.chunk_size)
    volume_png = normalize_volume_for_png(volume)

    if args.save_raw_npy:
        np.save(args.output_dir / "volume.npy", volume.astype(np.float32))
    np.save(args.output_dir / "volume_uint8.npy", volume_png)

    slices_dir = args.output_dir / f"slices_{args.slice_axis}"
    slices_dir.mkdir(parents=True, exist_ok=True)

    for idx, slice_img in iter_slices(volume_png, args.slice_axis):
        plt.imsave(slices_dir / f"{args.slice_axis}_{idx:04d}.png", slice_img, cmap="gray", vmin=0, vmax=255)

    metadata = {
        "ckpt": str(args.ckpt),
        "spacing": args.spacing,
        "point_min": point_min.tolist(),
        "point_max": point_max.tolist(),
        "bounding_box_ckpt_min": ckpt_min.tolist(),
        "bounding_box_ckpt_max": ckpt_max.tolist(),
        "shape_zyx": list(volume.shape),
        "slice_axis": args.slice_axis,
        "output_slices_dir": str(slices_dir),
    }
    with (args.output_dir / "grid_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Grid rendering finished.")
    print(f"Output dir: {args.output_dir}")
    print(f"Volume shape (z, y, x): {volume.shape}")
    print(f"Slices saved to: {slices_dir}")


if __name__ == "__main__":
    main()
