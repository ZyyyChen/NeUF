"""
Masked column-wise baseline volume reconstruction.

This exporter keeps the same output-grid definition as
export_full_grid_from_ckpt.py / export_knn_baseline.py, but switches the
reconstruction logic to a recons3D-inspired workflow:

1. Load the checkpoint and baked dataset to reuse the same bounds / spacing.
2. Build a sweep-aligned canonical frame from the ordered ultrasound slices.
3. Reconstruct only voxels inside the sweep mask and the slice footprint mask.
4. Use one small KDTree per image column (built in the canonical 2-D plane).
5. Reuse neighbouring sweep slices when accumulating intensities.

The default behaviour follows the branch used in recons3D.py:
- only masked voxels are reconstructed
- two nearest columns are used
- three nearest row neighbours are used per column
- the current sweep slice and its immediate neighbours contribute

Usage
-----
python export_recons3d_style_baseline.py --ckpt latest/ckpt.pkl --output exports/recons3d_style
python export_recons3d_style_baseline.py --ckpt latest/ckpt.pkl --output exports/recons3d_style \
    --row-k 3 --column-k 2 --slice-neighbours 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import dataset
import numpy as np
import torch
from scipy.spatial import KDTree
from tqdm import tqdm

from utils import get_base_points, get_oriented_points_and_views


LOAD_DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class SliceRecord:
    pixels: np.ndarray
    position: np.ndarray
    rotation: Any
    sweep_coord: float
    source: str
    index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a recons3D-style masked baseline on the ckpt-aligned volume grid."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pkl")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--spacing", type=float, nargs="+", default=None, metavar="S",
        help="Override voxel spacing in mm. 1 value for isotropic spacing, or 3 for x/y/z.",
    )
    parser.add_argument(
        "--resolution-scale", type=float, default=1.0,
        help="Divide spacing by this factor (doubles resolution when set to 2).",
    )
    parser.add_argument(
        "--bounds-dataset", type=Path, default=None,
        help="Path to a separate baked_dataset.pkl to read point_min/point_max from.",
    )
    parser.add_argument(
        "--point-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="Override grid minimum corner in mm.",
    )
    parser.add_argument(
        "--point-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="Override grid maximum corner in mm.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Number of active voxels processed per interpolation batch (default: 100000).",
    )
    parser.add_argument(
        "--row-k", type=int, default=3,
        help="Number of nearest row neighbours queried inside each column KDTree (default: 3).",
    )
    parser.add_argument(
        "--column-k", type=int, default=2,
        help="Number of nearest image columns used per voxel (default: 2).",
    )
    parser.add_argument(
        "--slice-neighbours", type=int, default=1,
        help="How many neighbouring sweep slices on each side contribute (default: 1).",
    )
    parser.add_argument(
        "--disable-sequence-plane-mask",
        action="store_true",
        dest="disable_sequence_plane_mask",
        help="Disable the stored first/last slice sweep-plane mask.",
    )
    return parser.parse_args()


def make_dated_output_dir(base_output_dir: Path, ckpt_path: Path) -> Path:
    date_str = date.today().strftime("%d-%m-%Y")
    dated_root = base_output_dir / date_str
    dated_root.mkdir(parents=True, exist_ok=True)
    ckpt_name = ckpt_path.stem
    num = 0
    while True:
        candidate = dated_root / f"{ckpt_name}_{num}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        num += 1


def load_dataset_from_ckpt(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=LOAD_DEVICE, weights_only=False)
    dataset_path = ckpt.get("baked_dataset_file", ckpt.get("dataset_folder"))
    if dataset_path is None:
        raise KeyError("Checkpoint does not contain 'baked_dataset_file'.")
    dataset_path = Path(dataset_path)
    saved = torch.load(dataset_path, map_location=LOAD_DEVICE, weights_only=False)
    if "dataset" not in saved:
        raise KeyError(f"'dataset' key not found in {dataset_path}")
    return saved["dataset"], dataset_path, ckpt


def to_numpy_bbox(ckpt):
    bb_min = ckpt["bounding_box"][0].detach().cpu().numpy().astype(np.float32)
    bb_max = ckpt["bounding_box"][1].detach().cpu().numpy().astype(np.float32)
    return bb_min, bb_max


def get_default_spacing_xyz(baked_dataset) -> np.ndarray:
    spacing_x = float(baked_dataset.roi_px_size_width_mm)
    spacing_y = float(baked_dataset.roi_px_size_height_mm)
    spacing_z = min(spacing_x, spacing_y)
    spacing_xyz = np.array([spacing_x, spacing_y, spacing_z], dtype=np.float32)
    if np.any(spacing_xyz <= 0):
        raise ValueError(f"Invalid default spacing from dataset: {spacing_xyz.tolist()}")
    return spacing_xyz


def resolve_spacing_xyz(args, baked_dataset):
    if args.resolution_scale <= 0:
        raise ValueError("--resolution-scale must be > 0")
    if args.spacing is None:
        base_spacing = get_default_spacing_xyz(baked_dataset)
    else:
        vals = np.array(args.spacing, dtype=np.float32)
        if vals.size == 1:
            base_spacing = np.repeat(vals[0], 3).astype(np.float32)
        elif vals.size == 3:
            base_spacing = vals.astype(np.float32)
        else:
            raise ValueError("--spacing expects 1 or 3 values")
        if np.any(base_spacing <= 0):
            raise ValueError("--spacing values must all be > 0")
    spacing = base_spacing / float(args.resolution_scale)
    return spacing.astype(np.float32), base_spacing.astype(np.float32)


def build_axes(point_min: np.ndarray, point_max: np.ndarray, spacing_xyz: np.ndarray):
    axes = []
    for dim in range(3):
        extent = point_max[dim] - point_min[dim]
        count = max(1, int(np.ceil(extent / float(spacing_xyz[dim]))))
        axis = point_min[dim] + (np.arange(count, dtype=np.float32) + 0.5) * spacing_xyz[dim]
        axes.append(axis)
    return axes


def save_mhd(volume_zyx: np.ndarray, output_dir: Path, spacing_xyz: np.ndarray) -> None:
    volume_hzw = np.flip(volume_zyx, axis=2).astype(np.float32, copy=False)

    raw_path = output_dir / "volume.raw"
    mhd_path = output_dir / "volume.mhd"

    with raw_path.open("wb") as f:
        for s in range(volume_hzw.shape[0]):
            np.ascontiguousarray(volume_hzw[s]).tofile(f)

    dim_z, dim_y, dim_x = volume_hzw.shape
    sx, sy, sz = [float(v) for v in spacing_xyz]
    header = "\n".join([
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        "CompressedData = False",
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        "Offset = 0 0 0",
        "CenterOfRotation = 0 0 0",
        "AnatomicalOrientation = RAI",
        f"ElementSpacing = {sy} {sz} {sx}",
        f"DimSize = {dim_x} {dim_y} {dim_z}",
        "ElementType = MET_FLOAT",
        f"ElementDataFile = {raw_path.name}",
        "",
    ])
    mhd_path.write_text(header, encoding="ascii")


def normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vec / norm


def get_sequence_plane_mask_data_from_infos(dataset_path: Path, baked_dataset):
    infos_path = None
    stored_infos_path = getattr(baked_dataset, "infos_json_path", None)
    if stored_infos_path:
        candidate = Path(stored_infos_path)
        if candidate.exists():
            infos_path = candidate
    if infos_path is None:
        candidate = dataset_path.parent / "infos.json"
        if candidate.exists():
            infos_path = candidate
    if infos_path is None:
        return None

    infos_json = json.loads(infos_path.read_text(encoding="utf-8"))
    frame_keys = sorted((k for k in infos_json.keys() if k != "infos"), key=lambda k: int(k))
    image_step = max(1, int(getattr(baked_dataset, "image_step", 1)))
    frame_keys = frame_keys[::image_step]
    if len(frame_keys) < 2:
        return None

    reverse_quat = bool(getattr(baked_dataset, "reverse_quat", False))

    def parse_frame(frame_key):
        frame = infos_json[frame_key]
        point = np.array(
            [float(frame["x"]), float(frame["y"]), float(frame["z"])],
            dtype=np.float32,
        )
        if reverse_quat:
            quat = dataset.Quat(
                float(frame["w3"]),
                float(frame["w0"]),
                float(frame["w1"]),
                float(frame["w2"]),
            )
        else:
            quat = dataset.Quat(
                float(frame["w0"]),
                float(frame["w1"]),
                float(frame["w2"]),
                float(frame["w3"]),
            )
        rotmat = quat.as_rotmat()
        normal = normalize_vector(rotmat[:, 2])
        return point, normal

    front_point, front_normal = parse_frame(frame_keys[0])
    back_point, back_normal = parse_frame(frame_keys[-1])
    mid_point, _ = parse_frame(frame_keys[len(frame_keys) // 2])

    if np.dot(mid_point - front_point, front_normal) < 0:
        front_normal = -front_normal
    if np.dot(mid_point - back_point, back_normal) < 0:
        back_normal = -back_normal

    return {
        "front_point": front_point,
        "front_normal": front_normal,
        "back_point": back_point,
        "back_normal": back_normal,
        "source": f"slice planes from {infos_path}",
    }


def get_sequence_plane_mask_data_from_dataset_metadata(baked_dataset):
    front_point = getattr(baked_dataset, "front_plane_point", None)
    back_point = getattr(baked_dataset, "back_plane_point", None)
    front_normal = getattr(baked_dataset, "front_plane_normal", None)
    back_normal = getattr(baked_dataset, "back_plane_normal", None)
    scan_axis = getattr(baked_dataset, "scan_axis", None)

    if front_point is None or back_point is None:
        return None

    if front_normal is None:
        front_normal = scan_axis
    if back_normal is None:
        back_normal = scan_axis
    if front_normal is None or back_normal is None:
        return None

    front_point = np.asarray(front_point, dtype=np.float32)
    back_point = np.asarray(back_point, dtype=np.float32)
    front_normal = normalize_vector(front_normal)
    back_normal = normalize_vector(back_normal)

    scan_vec = back_point - front_point
    if np.linalg.norm(scan_vec) <= 0:
        return None

    if np.dot(scan_vec, front_normal) < 0:
        front_normal = -front_normal
    if np.dot(-scan_vec, back_normal) < 0:
        back_normal = -back_normal

    return {
        "front_point": front_point,
        "front_normal": front_normal,
        "back_point": back_point,
        "back_normal": back_normal,
        "source": "stored baked_dataset metadata",
    }


def get_sequence_plane_mask_data(baked_dataset, dataset_path: Path):
    from_infos = get_sequence_plane_mask_data_from_infos(dataset_path, baked_dataset)
    if from_infos is not None:
        return from_infos
    return get_sequence_plane_mask_data_from_dataset_metadata(baked_dataset)


def compute_sequence_plane_mask(points_xyz, plane_mask_data):
    front_signed = np.sum(
        (points_xyz - plane_mask_data["front_point"][None, :]) * plane_mask_data["front_normal"][None, :],
        axis=1,
    )
    back_signed = np.sum(
        (points_xyz - plane_mask_data["back_point"][None, :]) * plane_mask_data["back_normal"][None, :],
        axis=1,
    )
    return (front_signed >= 0.0) & (back_signed >= 0.0)


def _iter_all_slices(baked_dataset):
    for idx, sl in enumerate(baked_dataset.slices):
        yield "train", idx, sl, baked_dataset.get_slice_pixels(idx)
    for idx, sl in enumerate(baked_dataset.slices_valid):
        yield "valid", idx, sl, baked_dataset.get_slice_valid_pixels(idx)


def compute_sweep_axis(baked_dataset, plane_mask_data, fallback_positions):
    stored_scan_axis = getattr(baked_dataset, "scan_axis", None)
    if stored_scan_axis is not None and np.linalg.norm(stored_scan_axis) > 0:
        return normalize_vector(stored_scan_axis)
    if plane_mask_data is not None:
        sweep_vec = plane_mask_data["back_point"] - plane_mask_data["front_point"]
        if np.linalg.norm(sweep_vec) > 0:
            return normalize_vector(sweep_vec)
    if len(fallback_positions) < 2:
        raise ValueError("Need at least two slice positions to infer a sweep axis.")
    return normalize_vector(fallback_positions[-1] - fallback_positions[0])


def collect_sorted_slice_records(baked_dataset, sweep_axis):
    records = []
    for source, idx, sl, pixels_tensor in tqdm(
        list(_iter_all_slices(baked_dataset)),
        desc="Loading slice stack",
    ):
        pixels = torch.reshape(
            pixels_tensor, (baked_dataset.px_height, baked_dataset.px_width)
        ).detach().cpu().numpy().astype(np.float32)
        position = np.asarray(sl.position, dtype=np.float32)
        sweep_coord = float(np.dot(position, sweep_axis))
        records.append(
            SliceRecord(
                pixels=pixels,
                position=position,
                rotation=sl.rotation,
                sweep_coord=sweep_coord,
                source=source,
                index=idx,
            )
        )
    records.sort(key=lambda rec: rec.sweep_coord)
    return records


def compute_canonical_frame(records, baked_dataset, plane_mask_data):
    if not records:
        raise ValueError("No slice records available.")

    sweep_axis = compute_sweep_axis(
        baked_dataset,
        plane_mask_data,
        [rec.position for rec in records],
    )
    ref_record = records[len(records) // 2]
    ref_rot = ref_record.rotation.as_rotmat()
    ref_row = normalize_vector(ref_rot[:, 0])
    ref_col = normalize_vector(ref_rot[:, 1])

    row_vectors = []
    col_vectors = []
    for rec in records:
        rotmat = rec.rotation.as_rotmat()
        row_vec = rotmat[:, 0].astype(np.float32)
        col_vec = rotmat[:, 1].astype(np.float32)
        if np.dot(row_vec, ref_row) < 0:
            row_vec = -row_vec
        if np.dot(col_vec, ref_col) < 0:
            col_vec = -col_vec

        row_proj = row_vec - np.dot(row_vec, sweep_axis) * sweep_axis
        col_proj = col_vec - np.dot(col_vec, sweep_axis) * sweep_axis
        if np.linalg.norm(row_proj) > 1e-6:
            row_vectors.append(normalize_vector(row_proj))
        if np.linalg.norm(col_proj) > 1e-6:
            col_vectors.append(normalize_vector(col_proj))

    if not row_vectors:
        raise ValueError("Could not estimate a canonical row axis.")
    row_axis = normalize_vector(np.mean(np.stack(row_vectors, axis=0), axis=0))
    row_axis = normalize_vector(row_axis - np.dot(row_axis, sweep_axis) * sweep_axis)

    if col_vectors:
        col_seed = np.mean(np.stack(col_vectors, axis=0), axis=0)
        col_seed = col_seed - np.dot(col_seed, row_axis) * row_axis
        col_seed = col_seed - np.dot(col_seed, sweep_axis) * sweep_axis
        if np.linalg.norm(col_seed) > 1e-6:
            col_axis = normalize_vector(col_seed)
        else:
            col_axis = normalize_vector(np.cross(sweep_axis, row_axis))
    else:
        col_axis = normalize_vector(np.cross(sweep_axis, row_axis))

    if np.dot(col_axis, ref_col) < 0:
        col_axis = -col_axis
    row_axis = normalize_vector(
        row_axis - np.dot(row_axis, col_axis) * col_axis - np.dot(row_axis, sweep_axis) * sweep_axis
    )
    if np.dot(row_axis, ref_row) < 0:
        row_axis = -row_axis

    origin = ref_record.position.astype(np.float32)
    slice_sweep_coords = np.array(
        [np.dot(rec.position - origin, sweep_axis) for rec in records],
        dtype=np.float32,
    )
    return origin, row_axis, col_axis, sweep_axis, slice_sweep_coords, ref_record


def project_world_to_canonical(points_xyz, origin, row_axis, col_axis, sweep_axis):
    rel = points_xyz - origin[None, :]
    row_coords = rel @ row_axis
    col_coords = rel @ col_axis
    sweep_coords = rel @ sweep_axis
    return row_coords, col_coords, sweep_coords


def build_reference_geometry(baked_dataset, ref_record, origin, row_axis, col_axis):
    base_x, base_y = get_base_points(
        baked_dataset.width,
        baked_dataset.height,
        baked_dataset.px_width,
        baked_dataset.px_height,
        offset_x_mm=baked_dataset.roi_offset_x_mm,
        offset_y_mm=baked_dataset.roi_offset_y_mm,
    )
    points_world, _ = get_oriented_points_and_views(
        base_x,
        base_y,
        ref_record.position,
        ref_record.rotation,
    )
    rel = points_world - origin[None, :]
    row_coords = (rel @ row_axis).reshape(baked_dataset.px_height, baked_dataset.px_width)
    col_coords = (rel @ col_axis).reshape(baked_dataset.px_height, baked_dataset.px_width)
    ref_points_2d = np.stack([row_coords, col_coords], axis=-1).astype(np.float32)
    return ref_points_2d


def build_column_trees(ref_points_2d):
    column_trees = []
    for col_idx in tqdm(range(ref_points_2d.shape[1]), desc="Building column KDTrees"):
        column_trees.append(KDTree(ref_points_2d[:, col_idx, :]))
    anchor_points = ref_points_2d[ref_points_2d.shape[0] // 2].astype(np.float32)
    anchor_tree = KDTree(anchor_points)
    return column_trees, anchor_points, anchor_tree


def find_nearest_slice_indices(query_sweep, slice_sweep_coords):
    right = np.searchsorted(slice_sweep_coords, query_sweep, side="left")
    right = np.clip(right, 0, len(slice_sweep_coords) - 1)
    left = np.clip(right - 1, 0, len(slice_sweep_coords) - 1)

    left_dist = np.abs(query_sweep - slice_sweep_coords[left])
    right_dist = np.abs(query_sweep - slice_sweep_coords[right])
    use_right = right_dist < left_dist
    return np.where(use_right, right, left).astype(np.int32)


def accumulate_column_contributions(
    accum,
    sum_w,
    q2,
    q_sweep,
    primary_slice_idx,
    column_idx,
    local_indices,
    column_tree,
    slice_images,
    slice_sweep_coords,
    row_k,
    slice_neighbours,
):
    local_q2 = q2[local_indices]
    local_sweep = q_sweep[local_indices]
    local_primary = primary_slice_idx[local_indices]

    dists, row_idxs = column_tree.query(local_q2, k=row_k, workers=-1)
    if row_k == 1:
        dists = dists[:, None]
        row_idxs = row_idxs[:, None]

    eps = 1e-8
    safe_d = np.where(dists < eps, eps, dists)

    for offset in range(-slice_neighbours, slice_neighbours + 1):
        neighbour_idx = local_primary + offset
        valid = (neighbour_idx >= 0) & (neighbour_idx < slice_images.shape[0])
        if not np.any(valid):
            continue

        neighbour_idx = np.clip(neighbour_idx, 0, slice_images.shape[0] - 1)
        sampled = slice_images[neighbour_idx[:, None], row_idxs, column_idx]

        if offset == 0:
            weights = 1.0 / safe_d
        else:
            sweep_delta = np.abs(local_sweep - slice_sweep_coords[neighbour_idx])
            weights = 1.0 / np.sqrt(safe_d ** 2 + sweep_delta[:, None] ** 2)

        weights = np.where(valid[:, None], weights, 0.0)
        accum[local_indices] += (weights * sampled).sum(axis=1)
        sum_w[local_indices] += weights.sum(axis=1)


def reconstruct_volume(
    x_axis,
    y_axis,
    z_axis,
    slice_images,
    slice_sweep_coords,
    origin,
    row_axis,
    col_axis,
    sweep_axis,
    plane_mask_data,
    row_bounds,
    col_bounds,
    column_trees,
    anchor_tree,
    args,
):
    volume_zyx = np.zeros((len(z_axis), len(y_axis), len(x_axis)), dtype=np.float32)
    plane_x, plane_y = np.meshgrid(x_axis, y_axis, indexing="xy")
    flat_plane_x = plane_x.reshape(-1)
    flat_plane_y = plane_y.reshape(-1)

    sweep_min = float(slice_sweep_coords.min())
    sweep_max = float(slice_sweep_coords.max())
    sweep_margin = 0.5 * float(np.mean(np.diff(slice_sweep_coords))) if len(slice_sweep_coords) > 1 else 0.0

    total_active_voxels = 0
    for z_idx, z_value in enumerate(tqdm(z_axis, desc="Recons-style interpolation")):
        plane_points = np.stack(
            [
                flat_plane_x,
                flat_plane_y,
                np.full_like(flat_plane_x, z_value, dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        row_coords, col_coords, sweep_coords = project_world_to_canonical(
            plane_points,
            origin,
            row_axis,
            col_axis,
            sweep_axis,
        )
        active_mask = (
            (row_coords >= row_bounds[0]) &
            (row_coords <= row_bounds[1]) &
            (col_coords >= col_bounds[0]) &
            (col_coords <= col_bounds[1]) &
            (sweep_coords >= sweep_min - sweep_margin) &
            (sweep_coords <= sweep_max + sweep_margin)
        )
        if plane_mask_data is not None:
            active_mask &= compute_sequence_plane_mask(plane_points, plane_mask_data)

        active_indices = np.flatnonzero(active_mask)
        if active_indices.size == 0:
            continue

        total_active_voxels += int(active_indices.size)
        plane_values = np.zeros((plane_points.shape[0],), dtype=np.float32)
        active_q2 = np.stack([row_coords[active_mask], col_coords[active_mask]], axis=-1).astype(np.float32)
        active_sweep = sweep_coords[active_mask].astype(np.float32)
        active_primary_slice = find_nearest_slice_indices(active_sweep, slice_sweep_coords)

        for start in range(0, active_indices.size, args.chunk_size):
            stop = min(start + args.chunk_size, active_indices.size)
            batch_indices = active_indices[start:stop]
            batch_q2 = active_q2[start:stop]
            batch_sweep = active_sweep[start:stop]
            batch_primary = active_primary_slice[start:stop]

            anchor_dists, anchor_cols = anchor_tree.query(
                batch_q2,
                k=min(args.column_k, len(column_trees)),
                workers=-1,
            )
            if args.column_k == 1 or anchor_cols.ndim == 1:
                anchor_cols = anchor_cols[:, None]

            batch_accum = np.zeros((batch_q2.shape[0],), dtype=np.float64)
            batch_sum_w = np.zeros((batch_q2.shape[0],), dtype=np.float64)

            for column_slot in range(anchor_cols.shape[1]):
                column_ids = anchor_cols[:, column_slot].astype(np.int32)
                for column_idx in np.unique(column_ids):
                    local_indices = np.flatnonzero(column_ids == column_idx)
                    accumulate_column_contributions(
                        batch_accum,
                        batch_sum_w,
                        batch_q2,
                        batch_sweep,
                        batch_primary,
                        int(column_idx),
                        local_indices,
                        column_trees[int(column_idx)],
                        slice_images,
                        slice_sweep_coords,
                        args.row_k,
                        args.slice_neighbours,
                    )

            valid = batch_sum_w > 0
            if np.any(valid):
                plane_values[batch_indices[valid]] = (
                    batch_accum[valid] / batch_sum_w[valid]
                ).astype(np.float32)

        volume_zyx[z_idx] = plane_values.reshape(len(y_axis), len(x_axis))

    return volume_zyx, total_active_voxels


def main():
    args = parse_args()
    if args.row_k <= 0:
        raise ValueError("--row-k must be >= 1")
    if args.column_k <= 0:
        raise ValueError("--column-k must be >= 1")
    if args.slice_neighbours < 0:
        raise ValueError("--slice-neighbours must be >= 0")

    output_dir = make_dated_output_dir(args.output, args.ckpt)
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint/data load device: {LOAD_DEVICE}")

    baked_dataset, dataset_path, ckpt = load_dataset_from_ckpt(args.ckpt)
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Training slices: {len(baked_dataset.slices)}, validation slices: {len(baked_dataset.slices_valid)}")

    if args.bounds_dataset is not None:
        bounds_saved = torch.load(args.bounds_dataset, map_location=LOAD_DEVICE, weights_only=False)
        bounds_ds = bounds_saved["dataset"]
        default_point_min = np.array(bounds_ds.point_min, dtype=np.float32)
        default_point_max = np.array(bounds_ds.point_max, dtype=np.float32)
        print(f"Bounds loaded from: {args.bounds_dataset}")
    else:
        default_point_min = np.array(baked_dataset.point_min, dtype=np.float32)
        default_point_max = np.array(baked_dataset.point_max, dtype=np.float32)

    point_min = np.array(args.point_min, dtype=np.float32) if args.point_min is not None else default_point_min
    point_max = np.array(args.point_max, dtype=np.float32) if args.point_max is not None else default_point_max
    spacing_xyz, base_spacing_xyz = resolve_spacing_xyz(args, baked_dataset)
    x_axis, y_axis, z_axis = build_axes(point_min, point_max, spacing_xyz)
    bb_min, bb_max = to_numpy_bbox(ckpt)

    print(f"Grid shape (z, y, x): {(len(z_axis), len(y_axis), len(x_axis))}")
    print(f"Spacing (x, y, z) mm: {spacing_xyz.tolist()}")
    print(f"Point min (x, y, z) mm: {point_min.tolist()}")
    print(f"Point max (x, y, z) mm: {point_max.tolist()}")
    print(f"Checkpoint bbox min/max: {bb_min.tolist()} / {bb_max.tolist()}")

    use_sequence_plane_mask = not args.disable_sequence_plane_mask
    plane_mask_data = get_sequence_plane_mask_data(baked_dataset, dataset_path) if use_sequence_plane_mask else None
    if use_sequence_plane_mask:
        if plane_mask_data is None:
            print("Sequence plane mask requested, but no valid front/back plane metadata was found. Falling back to the sweep-coordinate bounds only.")
        else:
            print("Using sequence plane mask:")
            print(f"  Source: {plane_mask_data.get('source', 'unknown')}")
            print(f"  Front plane point: {plane_mask_data['front_point']}")
            print(f"  Back plane point: {plane_mask_data['back_point']}")

    fallback_positions = [np.asarray(sl.position, dtype=np.float32) for sl in baked_dataset.slices]
    if not fallback_positions:
        fallback_positions = [np.asarray(sl.position, dtype=np.float32) for sl in baked_dataset.slices_valid]
    sweep_axis = compute_sweep_axis(baked_dataset, plane_mask_data, fallback_positions)
    slice_records = collect_sorted_slice_records(baked_dataset, sweep_axis)
    origin, row_axis, col_axis, sweep_axis, slice_sweep_coords, ref_record = compute_canonical_frame(
        slice_records,
        baked_dataset,
        plane_mask_data,
    )
    print(f"Canonical origin: {origin.tolist()}")
    print(f"Canonical row axis: {row_axis.tolist()}")
    print(f"Canonical col axis: {col_axis.tolist()}")
    print(f"Canonical sweep axis: {sweep_axis.tolist()}")
    print(f"Slice sweep-coordinate range: [{float(slice_sweep_coords.min()):.3f}, {float(slice_sweep_coords.max()):.3f}] mm")

    slice_images = np.stack([rec.pixels for rec in slice_records], axis=0).astype(np.float32, copy=False)
    ref_points_2d = build_reference_geometry(
        baked_dataset,
        ref_record,
        origin,
        row_axis,
        col_axis,
    )
    row_margin = 0.5 * float(spacing_xyz[2])
    col_margin = 0.5 * float(spacing_xyz[0])
    row_bounds = (
        float(ref_points_2d[..., 0].min()) - row_margin,
        float(ref_points_2d[..., 0].max()) + row_margin,
    )
    col_bounds = (
        float(ref_points_2d[..., 1].min()) - col_margin,
        float(ref_points_2d[..., 1].max()) + col_margin,
    )
    print(f"Canonical in-plane row bounds: {row_bounds}")
    print(f"Canonical in-plane col bounds: {col_bounds}")

    column_trees, anchor_points, anchor_tree = build_column_trees(ref_points_2d)
    print(f"Built {len(column_trees)} column KDTrees with {ref_points_2d.shape[0]} rows each.")

    volume_zyx, active_voxels = reconstruct_volume(
        x_axis,
        y_axis,
        z_axis,
        slice_images,
        slice_sweep_coords,
        origin,
        row_axis,
        col_axis,
        sweep_axis,
        plane_mask_data,
        row_bounds,
        col_bounds,
        column_trees,
        anchor_tree,
        args,
    )
    filled = int(np.count_nonzero(volume_zyx))
    total_voxels = int(volume_zyx.size)
    print(f"Active voxels reconstructed: {active_voxels:,} / {total_voxels:,}")
    print(f"Filled voxels: {filled:,} / {total_voxels:,}")
    print(f"Intensity range: [{float(volume_zyx.min()):.2f}, {float(volume_zyx.max()):.2f}]")

    print("Saving MHD volume...")
    save_mhd(volume_zyx, output_dir, spacing_xyz)

    metadata = {
        "method": "recons3d_style_masked_columnwise",
        "ckpt": str(args.ckpt),
        "dataset_pkl": str(dataset_path),
        "bounds_dataset_pkl": str(args.bounds_dataset) if args.bounds_dataset else str(dataset_path),
        "grid_shape_zyx": list(volume_zyx.shape),
        "spacing_mm_xyz": spacing_xyz.tolist(),
        "base_spacing_mm_xyz": base_spacing_xyz.tolist(),
        "resolution_scale": float(args.resolution_scale),
        "point_min_mm": point_min.tolist(),
        "point_max_mm": point_max.tolist(),
        "row_k": int(args.row_k),
        "column_k": int(args.column_k),
        "slice_neighbours": int(args.slice_neighbours),
        "use_sequence_plane_mask": bool(plane_mask_data is not None),
        "sequence_plane_mask_requested": bool(use_sequence_plane_mask),
        "sequence_plane_mask_source": None if plane_mask_data is None else plane_mask_data.get("source"),
        "canonical_origin_mm": origin.tolist(),
        "canonical_row_axis": row_axis.tolist(),
        "canonical_col_axis": col_axis.tolist(),
        "canonical_sweep_axis": sweep_axis.tolist(),
        "canonical_row_bounds_mm": list(row_bounds),
        "canonical_col_bounds_mm": list(col_bounds),
        "slice_sweep_min_mm": float(slice_sweep_coords.min()),
        "slice_sweep_max_mm": float(slice_sweep_coords.max()),
        "num_slices_used": int(slice_images.shape[0]),
        "active_voxels": int(active_voxels),
        "filled_voxels": int(filled),
        "total_voxels": int(total_voxels),
        "anchor_points_shape": list(anchor_points.shape),
        "mhd_grid_axis_order": ["h", "z", "w"],
        "mhd_grid_axis_mapping": {"h": "-x", "z": "-z", "w": "-y"},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Saved to: {output_dir}")
    print(f"  volume.raw / volume.mhd  — shape (h, z, w): {np.flip(volume_zyx, axis=2).shape}")


if __name__ == "__main__":
    main()
