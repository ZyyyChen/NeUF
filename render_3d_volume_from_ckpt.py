from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nerf_network import NeRF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_BBOX_EXTENT = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 3D Cartesian volume directly from a NeUF checkpoint. "
            "This path follows the current SliceRenderer point-normalization rules, "
            "including DUAL_HASH physical-coordinate queries and dual training progress."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pkl")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--output-exact",
        action="store_true",
        help="Write directly into --output instead of creating a dated checkpoint subdirectory.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        default=None,
        metavar="S",
        help="Voxel spacing in mm. Pass one value for isotropic, or three values for x y z.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Scale default voxel resolution by dividing spacing by this factor.",
    )
    parser.add_argument("--point-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--point-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--use-bbox-mask", action="store_true")
    parser.add_argument(
        "--training-progress",
        type=float,
        default=None,
        help=(
            "Override dual encoder progress in [0, 1]. "
            "By default this is inferred as checkpoint start / --nb-iters-max."
        ),
    )
    parser.add_argument(
        "--nb-iters-max",
        type=int,
        default=10000,
        help="Training iteration count used to infer dual training progress.",
    )
    parser.add_argument(
        "--output-type",
        choices=("float", "uint8"),
        default="uint8",
        help="MHD element type to write for volume.mhd.",
    )
    parser.add_argument(
        "--save-volume-npy",
        action="store_true",
        help="Also save the rendered float volume as volume_zyx.npy.",
    )
    return parser.parse_args()


def make_output_dir(base_output_dir: Path, ckpt_path: Path, output_exact: bool) -> Path:
    if output_exact:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir

    dated_root = base_output_dir / date.today().strftime("%d-%m-%Y")
    dated_root.mkdir(parents=True, exist_ok=True)
    ckpt_name = ckpt_path.stem
    run_idx = 0
    while True:
        candidate = dated_root / f"{ckpt_name}_{run_idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        run_idx += 1


def load_checkpoint_and_model(ckpt_path: Path) -> tuple[dict, NeRF]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = NeRF(ckpt).to(DEVICE)
    model.eval()
    return ckpt, model


def load_baked_dataset(ckpt: dict):
    dataset_path = ckpt.get("baked_dataset_file", ckpt.get("dataset_folder"))
    if dataset_path is None:
        raise KeyError("Checkpoint does not contain 'baked_dataset_file' or 'dataset_folder'.")

    dataset_path = Path(dataset_path)
    saved = torch.load(dataset_path, map_location=DEVICE, weights_only=False)
    if "dataset" not in saved:
        raise KeyError(f"'dataset' key not found in {dataset_path}")
    return saved["dataset"], dataset_path


def numpy_bbox(ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
    bb_min = ckpt["bounding_box"][0].detach().cpu().numpy().astype(np.float32)
    bb_max = ckpt["bounding_box"][1].detach().cpu().numpy().astype(np.float32)
    return bb_min, bb_max


def default_spacing_xyz(baked_dataset) -> np.ndarray:
    spacing_x = float(baked_dataset.roi_px_size_width_mm)
    spacing_y = float(baked_dataset.roi_px_size_height_mm)
    spacing_z = min(spacing_x, spacing_y)
    spacing = np.array([spacing_x, spacing_y, spacing_z], dtype=np.float32)
    if np.any(spacing <= 0):
        raise ValueError(f"Invalid dataset spacing: {spacing.tolist()}")
    return spacing


def resolve_spacing(args: argparse.Namespace, baked_dataset) -> tuple[np.ndarray, np.ndarray]:
    if args.resolution_scale <= 0:
        raise ValueError("--resolution-scale must be > 0")

    if args.spacing is None:
        base_spacing = default_spacing_xyz(baked_dataset)
    else:
        values = np.asarray(args.spacing, dtype=np.float32)
        if values.size == 1:
            base_spacing = np.repeat(values[0], 3).astype(np.float32)
        elif values.size == 3:
            base_spacing = values.astype(np.float32)
        else:
            raise ValueError("--spacing expects either one value or three values.")
        if np.any(base_spacing <= 0):
            raise ValueError("--spacing values must be > 0")

    spacing = base_spacing / float(args.resolution_scale)
    return spacing.astype(np.float32), base_spacing.astype(np.float32)


def build_axis(point_min: np.ndarray, point_max: np.ndarray, spacing_xyz: np.ndarray) -> list[np.ndarray]:
    axes = []
    for dim in range(3):
        spacing = float(spacing_xyz[dim])
        count = max(1, int(np.ceil((point_max[dim] - point_min[dim]) / spacing)))
        axis = point_min[dim] + (np.arange(count, dtype=np.float32) + 0.5) * spacing
        axes.append(axis.astype(np.float32, copy=False))
    return axes


def infer_training_progress(args: argparse.Namespace, ckpt: dict, model: NeRF) -> float:
    if args.training_progress is not None:
        progress = float(args.training_progress)
    elif "training_progress" in ckpt:
        progress = float(ckpt["training_progress"])
    elif str(getattr(model, "encoding_type", "")).startswith("DUAL_"):
        progress = float(ckpt.get("start", args.nb_iters_max)) / float(max(1, args.nb_iters_max))
    else:
        progress = 1.0

    return float(np.clip(progress, 0.0, 1.0))


def normalize_points_for_model(
    model: NeRF,
    points: torch.Tensor,
    bb_min: torch.Tensor,
    bb_max: torch.Tensor,
) -> torch.Tensor:
    if model.encoding_type in {"HASH", "DUAL_HASH"} and model.use_encoding:
        return points

    bbox_extent = torch.clamp(bb_max - bb_min, min=MIN_BBOX_EXTENT)
    return ((points - bb_min) / bbox_extent) * 2.0 - 1.0


def query_volume(
    model: NeRF,
    ckpt: dict,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    chunk_size: int,
    use_bbox_mask: bool,
) -> np.ndarray:
    bb_min_np, bb_max_np = numpy_bbox(ckpt)
    bb_min = torch.from_numpy(bb_min_np).to(DEVICE)
    bb_max = torch.from_numpy(bb_max_np).to(DEVICE)

    xx, yy = np.meshgrid(x_axis.astype(np.float32), y_axis.astype(np.float32), indexing="xy")
    plane_x = xx.reshape(-1)
    plane_y = yy.reshape(-1)
    volume_zyx = np.zeros((len(z_axis), len(y_axis), len(x_axis)), dtype=np.float32)

    total_active = 0
    total_points = int(volume_zyx.size)
    with torch.no_grad():
        for z_idx, z_value in enumerate(tqdm(z_axis, desc="Rendering 3D volume")):
            plane_points = np.stack(
                (
                    plane_x,
                    plane_y,
                    np.full(plane_x.shape, z_value, dtype=np.float32),
                ),
                axis=-1,
            )

            active_mask = np.ones((plane_points.shape[0],), dtype=bool)
            if use_bbox_mask:
                active_mask &= np.all((plane_points >= bb_min_np) & (plane_points <= bb_max_np), axis=1)

            active_indices = np.flatnonzero(active_mask)
            if active_indices.size == 0:
                continue

            total_active += int(active_indices.size)
            plane_values = np.zeros((plane_points.shape[0],), dtype=np.float32)
            query_points_np = plane_points[active_indices]

            for start in range(0, query_points_np.shape[0], chunk_size):
                stop = min(start + chunk_size, query_points_np.shape[0])
                batch_points = torch.from_numpy(query_points_np[start:stop]).to(DEVICE)
                batch_dirs = torch.zeros_like(batch_points)
                batch_points = normalize_points_for_model(model, batch_points, bb_min, bb_max)
                batch_values = model.query(batch_points, batch_dirs, netchunk=chunk_size).reshape(-1)
                plane_values[active_indices[start:stop]] = batch_values.detach().cpu().numpy()

            volume_zyx[z_idx] = plane_values.reshape(len(y_axis), len(x_axis))

    if use_bbox_mask:
        print(f"Active queried voxels after bbox mask: {total_active} / {total_points}")
    return volume_zyx


def volume_to_uint8_preserve_zero(volume: np.ndarray) -> tuple[np.ndarray, float | None, float | None, int]:
    volume = np.asarray(volume, dtype=np.float32)
    output = np.zeros_like(volume, dtype=np.uint8)
    foreground = volume != 0.0
    count = int(np.count_nonzero(foreground))
    if count == 0:
        return output, None, None, count

    values = volume[foreground]
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value > min_value:
        output[foreground] = (((values - min_value) / (max_value - min_value)) * 254.0 + 1.0).astype(np.uint8)
    else:
        output[foreground] = 255
    return output, min_value, max_value, count


def write_mhd(
    volume_zyx: np.ndarray,
    output_dir: Path,
    spacing_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    element_type: str,
) -> None:
    raw_path = output_dir / "volume.raw"
    mhd_path = output_dir / "volume.mhd"

    array = np.ascontiguousarray(volume_zyx)
    array.tofile(raw_path)

    dim_z, dim_y, dim_x = [int(v) for v in volume_zyx.shape]
    spacing_x, spacing_y, spacing_z = [float(v) for v in spacing_xyz]
    offset_x, offset_y, offset_z = [float(v) for v in origin_xyz]
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            f"Offset = {offset_x} {offset_y} {offset_z}",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_x} {spacing_y} {spacing_z}",
            f"DimSize = {dim_x} {dim_y} {dim_z}",
            f"ElementType = {element_type}",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    mhd_path.write_text(header, encoding="ascii")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def ensure_disk_space(output_dir: Path, shape_zyx: tuple[int, int, int], output_type: str, save_npy: bool) -> None:
    dtype = np.uint8 if output_type == "uint8" else np.float32
    required = int(np.prod(shape_zyx, dtype=np.int64)) * np.dtype(dtype).itemsize
    if save_npy:
        required += int(np.prod(shape_zyx, dtype=np.int64)) * np.dtype(np.float32).itemsize
    required += 16 * 1024 * 1024

    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise RuntimeError(
            f"Not enough free disk space in {output_dir}: "
            f"{format_bytes(free)} free, {format_bytes(required)} estimated."
        )


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.output, args.ckpt, args.output_exact)

    ckpt, model = load_checkpoint_and_model(args.ckpt)
    baked_dataset, dataset_path = load_baked_dataset(ckpt)
    bb_min, bb_max = numpy_bbox(ckpt)

    progress = infer_training_progress(args, ckpt, model)
    model.training_progress = progress

    point_min = (
        np.asarray(baked_dataset.point_min, dtype=np.float32)
        if args.point_min is None
        else np.asarray(args.point_min, dtype=np.float32)
    )
    point_max = (
        np.asarray(baked_dataset.point_max, dtype=np.float32)
        if args.point_max is None
        else np.asarray(args.point_max, dtype=np.float32)
    )
    spacing_xyz, base_spacing_xyz = resolve_spacing(args, baked_dataset)
    x_axis, y_axis, z_axis = build_axis(point_min, point_max, spacing_xyz)
    shape_zyx = (len(z_axis), len(y_axis), len(x_axis))

    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset: {dataset_path}")
    print(f"Encoding: {model.encoding_type}")
    print(f"Training progress used for dual encoder: {model.training_progress:.6f}")
    print(f"Grid shape (z, y, x): {shape_zyx}")
    print(f"Spacing xyz mm: {tuple(float(v) for v in spacing_xyz)}")
    print(f"Point min xyz mm: {tuple(float(v) for v in point_min)}")
    print(f"Point max xyz mm: {tuple(float(v) for v in point_max)}")
    print(f"Checkpoint bbox min xyz mm: {tuple(float(v) for v in bb_min)}")
    print(f"Checkpoint bbox max xyz mm: {tuple(float(v) for v in bb_max)}")

    ensure_disk_space(output_dir, shape_zyx, args.output_type, args.save_volume_npy)
    volume_zyx = query_volume(
        model=model,
        ckpt=ckpt,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        chunk_size=args.chunk_size,
        use_bbox_mask=args.use_bbox_mask,
    )

    if args.save_volume_npy:
        np.save(output_dir / "volume_zyx.npy", volume_zyx)

    normalization_min = None
    normalization_max = None
    foreground_count = int(np.count_nonzero(volume_zyx))
    if args.output_type == "uint8":
        volume_to_write, normalization_min, normalization_max, foreground_count = volume_to_uint8_preserve_zero(volume_zyx)
        element_type = "MET_UCHAR"
    else:
        volume_to_write = volume_zyx.astype(np.float32, copy=False)
        element_type = "MET_FLOAT"

    origin_xyz = np.array([x_axis[0], y_axis[0], z_axis[0]], dtype=np.float32)
    write_mhd(volume_to_write, output_dir, spacing_xyz, origin_xyz, element_type)

    metadata = {
        "ckpt": str(args.ckpt),
        "dataset_pkl": str(dataset_path),
        "encoding": model.encoding_type,
        "training_progress": float(model.training_progress),
        "checkpoint_start": int(ckpt.get("start", -1)),
        "nb_iters_max_for_progress": int(args.nb_iters_max),
        "normalization_rule": (
            "none_float32"
            if args.output_type == "float"
            else "preserve_zero_background_foreground_min_max_to_1_255"
        ),
        "normalization_min": normalization_min,
        "normalization_max": normalization_max,
        "foreground_voxels": foreground_count,
        "shape_zyx": list(shape_zyx),
        "spacing_mm_xyz": spacing_xyz.tolist(),
        "base_spacing_mm_xyz": base_spacing_xyz.tolist(),
        "point_min_mm_xyz": point_min.tolist(),
        "point_max_mm_xyz": point_max.tolist(),
        "mhd_origin_mm_xyz": origin_xyz.tolist(),
        "bounding_box_min_mm_xyz": bb_min.tolist(),
        "bounding_box_max_mm_xyz": bb_max.tolist(),
        "use_bbox_mask": bool(args.use_bbox_mask),
        "output_type": args.output_type,
        "mhd_array_order": "raw array is z, y, x; MHD DimSize is x y z",
        "dual_hash_coordinate_rule": "physical mm coordinates, not normalized",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved volume to: {output_dir}")
    print(f"MHD: {output_dir / 'volume.mhd'}")


if __name__ == "__main__":
    main()
