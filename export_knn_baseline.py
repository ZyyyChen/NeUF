"""
KNN baseline volume reconstruction.

Reconstructs the same 3D grid as export_full_grid_from_ckpt.py, but instead
of querying a NeRF network it uses inverse-distance-weighted k-nearest-neighbour
interpolation directly in 3D world space — the same spirit as recons3D.py.

Pipeline
--------
1. Load a checkpoint (.pkl) to obtain grid bounds and spacing (identical to
   the NeRF export so volumes are directly comparable).
2. Collect every observed (world_xyz, intensity) pair from all training and
   validation frames stored in the baked dataset.
3. Build a 3-D KDTree over those points.
4. For each voxel centre in the NeRF grid, query the k nearest neighbours and
   fill the voxel with their inverse-distance-weighted average.
5. Save the result as an MHD/raw pair with the same layout as the NeRF export.

Usage
-----
python export_knn_baseline.py --ckpt latest/ckpt.pkl --output exports/knn_baseline
python export_knn_baseline.py --ckpt latest/ckpt.pkl --output exports/knn_baseline \\
    --k 5 --max-dist 5.0 --chunk-size 200000
"""

import argparse
import json
from datetime import date
from pathlib import Path

import dataset
import numpy as np
import torch
from scipy.spatial import KDTree
from tqdm import tqdm

from utils import get_base_points, get_oriented_points_and_views

# This baseline only uses torch to deserialize saved objects and convert loaded
# tensors back to NumPy. Keep all loads on CPU so export still works when CUDA
# is visible but temporarily unavailable or busy.
LOAD_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="KNN baseline 3-D volume reconstruction from a NeUF checkpoint."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pkl")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--exact-output-dir", type=Path, default=None,
        help="Write directly into this directory instead of auto-creating a dated ckpt_* folder.",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of nearest neighbours for IDW interpolation (default: 3)",
    )
    parser.add_argument(
        "--max-dist", type=float, default=None,
        help="Neighbours farther than this distance (mm) are ignored. "
             "Voxels with no neighbour within max-dist are left as 0. "
             "Default: no limit.",
    )
    parser.add_argument(
        "--spacing", type=float, nargs="+", default=None, metavar="S",
        help="Override voxel spacing in mm. 1 value for isotropic, or 3 for x/y/z.",
    )
    parser.add_argument(
        "--resolution-scale", type=float, default=1.0,
        help="Divide spacing by this factor (doubles resolution when set to 2).",
    )
    parser.add_argument(
        "--bounds-dataset", type=Path, default=None,
        help="Path to a separate baked_dataset.pkl to read point_min/point_max from "
             "(observations still come from the checkpoint dataset).",
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
        "--chunk-size", type=int, default=200_000,
        help="Number of voxels queried per KDTree batch (default: 200000).",
    )
    parser.add_argument(
        "--query-workers", type=int, default=-1,
        help="Worker threads passed to KDTree.query (default: -1, use all local cores).",
    )
    parser.add_argument(
        "--use-bbox-mask",
        action="store_true",
        help="Only interpolate voxels inside the checkpoint bounding box.",
    )
    parser.add_argument(
        "--disable-sequence-plane-mask",
        action="store_true",
        dest="disable_sequence_plane_mask",
        help="Disable querying only voxels between the first and last slice boundary planes.",
    )
    parser.add_argument(
        "--shard-rank", type=int, default=0,
        help="Index of this z-shard among --shard-world-size shards (default: 0).",
    )
    parser.add_argument(
        "--shard-world-size", type=int, default=1,
        help="Total number of z-shards used to split the export (default: 1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers (shared logic with export_full_grid_from_ckpt)
# ---------------------------------------------------------------------------

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


def resolve_output_dir(base_output_dir: Path, ckpt_path: Path, exact_output_dir: Path | None) -> Path:
    if exact_output_dir is None:
        return make_dated_output_dir(base_output_dir, ckpt_path)

    exact_output_dir = exact_output_dir.expanduser()
    if exact_output_dir.exists():
        if any(exact_output_dir.iterdir()):
            raise FileExistsError(
                f"Exact output directory already exists and is not empty: {exact_output_dir}"
            )
    else:
        exact_output_dir.mkdir(parents=True, exist_ok=False)
    return exact_output_dir


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
    return axes  # [x_axis, y_axis, z_axis]


def get_shard_range(total_count: int, shard_rank: int, shard_world_size: int) -> tuple[int, int]:
    if shard_world_size <= 0:
        raise ValueError("--shard-world-size must be >= 1")
    if shard_rank < 0 or shard_rank >= shard_world_size:
        raise ValueError(
            f"--shard-rank must be in [0, {shard_world_size - 1}], got {shard_rank}"
        )
    start = (total_count * shard_rank) // shard_world_size
    stop = (total_count * (shard_rank + 1)) // shard_world_size
    return start, stop


def save_mhd(volume_zyx: np.ndarray, output_dir: Path, spacing_xyz: np.ndarray) -> None:
    """Save volume as MHD/raw with the same layout as export_full_grid_from_ckpt."""
    # Apply the same flip as convert_grid_zyx_to_hzw in the NeRF exporter.
    volume_hzw = np.flip(volume_zyx, axis=2).astype(np.float32, copy=False)

    raw_path = output_dir / "volume.raw"
    mhd_path = output_dir / "volume.mhd"

    # Write raw slice by slice to avoid non-contiguous stride issues after flip.
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


# ---------------------------------------------------------------------------
# Query-space masking (aligned with export_full_grid_from_ckpt)
# ---------------------------------------------------------------------------

def normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-length plane normal.")
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


def build_query_points(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    zz_g, yy_g, xx_g = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    return np.stack([xx_g.ravel(), yy_g.ravel(), zz_g.ravel()], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Observation collection
# ---------------------------------------------------------------------------

def collect_observations(baked_dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return all (N, 3) world points and (N,) intensity values from the dataset."""
    base_x, base_y = get_base_points(
        baked_dataset.width,
        baked_dataset.height,
        baked_dataset.px_width,
        baked_dataset.px_height,
        offset_x_mm=baked_dataset.roi_offset_x_mm,
        offset_y_mm=baked_dataset.roi_offset_y_mm,
    )

    all_points = []
    all_values = []

    def _add_slices(slices, get_pixels):
        for idx, sl in enumerate(slices):
            pixels_tensor = get_pixels(idx)
            pixels = torch.reshape(pixels_tensor, (baked_dataset.px_height, baked_dataset.px_width))
            pixels_np = pixels.detach().cpu().numpy().astype(np.float32).reshape(-1)

            points_world, _ = get_oriented_points_and_views(base_x, base_y, sl.position, sl.rotation)
            all_points.append(points_world.astype(np.float32))
            all_values.append(pixels_np)

    _add_slices(baked_dataset.slices, baked_dataset.get_slice_pixels)
    _add_slices(baked_dataset.slices_valid, baked_dataset.get_slice_valid_pixels)

    points = np.concatenate(all_points, axis=0)   # (N, 3)
    values = np.concatenate(all_values, axis=0)   # (N,)
    return points, values


# ---------------------------------------------------------------------------
# KNN interpolation
# ---------------------------------------------------------------------------

def knn_interpolate(
    tree: KDTree,
    values: np.ndarray,
    query_points: np.ndarray,
    k: int,
    max_dist: float | None,
    chunk_size: int,
    query_workers: int,
) -> np.ndarray:
    """
    Inverse-distance-weighted interpolation for query_points using a pre-built KDTree.

    Returns a float32 array of shape (len(query_points),).
    """
    result = np.zeros(len(query_points), dtype=np.float32)

    for start in tqdm(range(0, len(query_points), chunk_size), desc="KNN interpolation", total=(len(query_points) + chunk_size - 1) // chunk_size):
        stop = min(start + chunk_size, len(query_points))
        batch = query_points[start:stop]

        dists, idxs = tree.query(batch, k=k, workers=query_workers)

        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Build valid mask: neighbours within max_dist
        if max_dist is not None:
            valid = dists <= max_dist
        else:
            valid = np.ones_like(dists, dtype=bool)

        # Replace zero distances to avoid div/0 (exact hit → weight=inf → use value directly)
        eps = 1e-8
        safe_dists = np.where(dists < eps, eps, dists)
        weights = np.where(valid, 1.0 / safe_dists, 0.0)

        sum_w = weights.sum(axis=1)
        weighted_vals = (weights * values[idxs]).sum(axis=1)

        # Voxels with no valid neighbour stay 0
        has_data = sum_w > 0
        result[start:stop][has_data] = (weighted_vals[has_data] / sum_w[has_data]).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output, args.ckpt, args.exact_output_dir)
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint/data load device: {LOAD_DEVICE}")

    # ---- Load dataset ----
    baked_dataset, dataset_path, ckpt = load_dataset_from_ckpt(args.ckpt)
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Training slices: {len(baked_dataset.slices)}, validation slices: {len(baked_dataset.slices_valid)}")

    # ---- Grid definition (identical to NeRF export) ----
    # Resolve bounds source: --point-min/max > --bounds-dataset > checkpoint dataset
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

    full_grid_shape = (len(z_axis), len(y_axis), len(x_axis))
    shard_z_start, shard_z_stop = get_shard_range(
        len(z_axis), args.shard_rank, args.shard_world_size
    )
    local_z_axis = z_axis[shard_z_start:shard_z_stop]
    if len(local_z_axis) == 0:
        raise ValueError(
            "Resolved shard contains no z slices. Reduce --shard-world-size or use a finer grid."
        )

    grid_shape = (len(local_z_axis), len(y_axis), len(x_axis))
    print(f"Global grid shape (z, y, x): {full_grid_shape}")
    print(
        f"Shard {args.shard_rank}/{args.shard_world_size - 1}: "
        f"z indices [{shard_z_start}, {shard_z_stop}) -> local grid {grid_shape}"
    )
    print(f"Spacing (x, y, z) mm: {spacing_xyz.tolist()}")
    print(f"Point min (x, y, z) mm: {point_min.tolist()}")
    print(f"Point max (x, y, z) mm: {point_max.tolist()}")
    print(f"KDTree query workers: {args.query_workers}")

    use_sequence_plane_mask = not args.disable_sequence_plane_mask
    plane_mask_data = get_sequence_plane_mask_data(baked_dataset, dataset_path) if use_sequence_plane_mask else None
    if use_sequence_plane_mask:
        if plane_mask_data is None:
            print("Sequence plane mask requested, but no valid front/back plane metadata was found. Falling back to unmasked interpolation.")
        else:
            print("Using sequence plane mask:")
            print(f"  Source: {plane_mask_data.get('source', 'unknown')}")
            print(f"  Front plane point: {plane_mask_data['front_point']}")
            print(f"  Front plane normal (inward): {plane_mask_data['front_normal']}")
            print(f"  Back plane point: {plane_mask_data['back_point']}")
            print(f"  Back plane normal (inward): {plane_mask_data['back_normal']}")
    if args.use_bbox_mask:
        print("Using checkpoint bounding-box mask:")
        print(f"  bb_min: {bb_min}")
        print(f"  bb_max: {bb_max}")

    # ---- Collect observations ----
    print("\nCollecting observations from all frames...")
    src_points, src_values = collect_observations(baked_dataset)
    print(f"Total observation points: {len(src_points):,}")

    # ---- Build KDTree ----
    print("Building KDTree...")
    tree = KDTree(src_points)

    # ---- Build query grid ----
    print("Building query grid...")
    query_points = build_query_points(x_axis, y_axis, local_z_axis)

    active_mask = np.ones((query_points.shape[0],), dtype=bool)
    if args.use_bbox_mask:
        active_mask &= np.all((query_points >= bb_min) & (query_points <= bb_max), axis=1)
    if plane_mask_data is not None:
        active_mask &= compute_sequence_plane_mask(query_points, plane_mask_data)

    local_total_voxels = query_points.shape[0]
    active_voxels = int(np.count_nonzero(active_mask))
    print(f"Local voxels to fill: {local_total_voxels:,}")
    if args.use_bbox_mask or plane_mask_data is not None:
        print(f"Active voxels after masking: {active_voxels:,} / {local_total_voxels:,}")

    # ---- KNN interpolation ----
    print(f"\nRunning KNN interpolation (k={args.k}, max_dist={args.max_dist})...")
    flat_values = np.zeros(local_total_voxels, dtype=np.float32)
    if active_voxels > 0:
        flat_values[active_mask] = knn_interpolate(
            tree,
            src_values,
            query_points[active_mask],
            k=args.k,
            max_dist=args.max_dist,
            chunk_size=args.chunk_size,
            query_workers=args.query_workers,
        )

    volume_zyx = flat_values.reshape(grid_shape)  # (nz, ny, nx)

    filled = np.count_nonzero(flat_values)
    print(
        f"Filled voxels: {filled:,} / {local_total_voxels:,} "
        f"({100 * filled / local_total_voxels:.1f}%)"
    )
    print(f"Intensity range: [{volume_zyx.min():.2f}, {volume_zyx.max():.2f}]")

    # ---- Save ----
    print("\nSaving MHD volume...")
    save_mhd(volume_zyx, output_dir, spacing_xyz)

    metadata = {
        "method": "knn_idw_3d",
        "ckpt": str(args.ckpt),
        "dataset_pkl": str(dataset_path),
        "bounds_dataset_pkl": str(args.bounds_dataset) if args.bounds_dataset else str(dataset_path),
        "k": args.k,
        "max_dist_mm": args.max_dist,
        "spacing_mm_xyz": spacing_xyz.tolist(),
        "base_spacing_mm_xyz": base_spacing_xyz.tolist(),
        "resolution_scale": float(args.resolution_scale),
        "point_min_mm": point_min.tolist(),
        "point_max_mm": point_max.tolist(),
        "grid_shape_zyx": list(grid_shape),
        "global_grid_shape_zyx": list(full_grid_shape),
        "total_observation_points": int(len(src_points)),
        "use_bbox_mask": bool(args.use_bbox_mask),
        "use_sequence_plane_mask": bool(plane_mask_data is not None),
        "sequence_plane_mask_requested": bool(use_sequence_plane_mask),
        "sequence_plane_mask_source": None if plane_mask_data is None else plane_mask_data.get("source"),
        "active_voxels_after_masking": int(active_voxels),
        "filled_voxels": int(filled),
        "total_voxels": int(local_total_voxels),
        "global_total_voxels": int(np.prod(full_grid_shape, dtype=np.int64)),
        "query_workers": int(args.query_workers),
        "shard_rank": int(args.shard_rank),
        "shard_world_size": int(args.shard_world_size),
        "shard_axis": "z",
        "shard_z_start": int(shard_z_start),
        "shard_z_stop": int(shard_z_stop),
        "mhd_grid_axis_order": ["h", "z", "w"],
        "mhd_grid_axis_mapping": {"h": "-x", "z": "-z", "w": "-y"},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Saved to: {output_dir}")
    print(f"  volume.raw / volume.mhd  — shape (h, z, w): {np.flip(volume_zyx, axis=2).shape}")


if __name__ == "__main__":
    main()
