import argparse
import json
import shutil
import h5py
from datetime import date
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dataset
from nerf_network import NeRF
from utils import get_base_points, get_oriented_points_and_views


# python .\export_full_grid_from_ckpt.py --ckpt .\latest\ckpt.pkl --output .\exports\full_grid
# python export_full_grid_from_ckpt.py --ckpt latest/ckpt.pkl --output exports/full_grid --recons-common-grid-h5 /home/zchen/Code/NeUF/exports/recons3d_style_cluster/03-04-2026/ckpt_3/recons_common_grid.h5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a 3D volume from a NeUF checkpoint. "
            "Supports the default baked-dataset full grid, an explicit Cartesian grid, "
            "or the exact recons_common_grid.h5 support grid."
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
        help="Override voxel spacing in mm. Pass 1 value for isotropic spacing, or 3 values for x/y/z.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Multiply voxel resolution uniformly in x/y/z by dividing spacing by this factor. Example: 2.0 halves spacing and doubles voxel counts per axis.",
    )
    parser.add_argument(
        "--point-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
        help="Override grid minimum corner in mm",
    )
    parser.add_argument(
        "--point-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
        help="Override grid maximum corner in mm",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
        help="Number of queried points per forward pass",
    )
    parser.add_argument(
        "--use-bbox-mask",
        action="store_true",
        help="Only query points inside checkpoint bounding box",
    )
    parser.add_argument(
        "--disable-sequence-plane-mask",
        action="store_true",
        dest="disable_sequence_plane_mask",
        help="Disable querying only points between the first and last slice boundary planes of the sequence.",
    )
    parser.add_argument(
        "--save-large-npy",
        action="store_true",
        help="Save large intermediate .npy arrays such as volume_zyx.npy / volume.npy / gt_*.npy.",
    )
    parser.add_argument(
        "--save-gt-exports",
        action="store_true",
        help="Save GT side products such as gt_stacked_slices.",
    )
    parser.add_argument(
        "--x-axis-npy",
        type=Path,
        default=None,
        help="Optional path to a 1D float32 .npy file of voxel-center x coordinates in mm.",
    )
    parser.add_argument(
        "--y-axis-npy",
        type=Path,
        default=None,
        help="Optional path to a 1D float32 .npy file of voxel-center y coordinates in mm.",
    )
    parser.add_argument(
        "--z-axis-npy",
        type=Path,
        default=None,
        help="Optional path to a 1D float32 .npy file of voxel-center z coordinates in mm.",
    )
    parser.add_argument(
        "--recons-common-grid-h5",
        type=Path,
        default=None,
        help="Optional path to a recons_common_grid.h5 bundle produced by export_recons3d_style_baseline.py.",
    )
    return parser.parse_args()


def make_dated_output_dir(base_output_dir: Path, ckpt_path: Path):
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


def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = NeRF(ckpt).to(DEVICE)
    model.eval()
    return ckpt, model


def load_dataset_from_ckpt(ckpt):
    dataset_path = ckpt.get("baked_dataset_file", ckpt.get("dataset_folder"))
    if dataset_path is None:
        raise KeyError("Checkpoint does not contain 'baked_dataset_file' or 'dataset_folder'.")

    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() != ".pkl":
        raise ValueError(
            f"Checkpoint dataset path is not a dataset.pkl file: {dataset_path}"
        )

    saved = torch.load(dataset_path, map_location=DEVICE, weights_only=False)
    if "dataset" not in saved:
        raise KeyError(f"'dataset' key not found in {dataset_path}")
    return saved["dataset"], dataset_path


def to_numpy_bbox(ckpt):
    bb_min = ckpt["bounding_box"][0].detach().cpu().numpy().astype(np.float32)
    bb_max = ckpt["bounding_box"][1].detach().cpu().numpy().astype(np.float32)
    return bb_min, bb_max


def get_default_spacing_xyz(baked_dataset):
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
        base_spacing_xyz = get_default_spacing_xyz(baked_dataset)
    else:
        spacing_values = np.array(args.spacing, dtype=np.float32)
        if spacing_values.size == 1:
            base_spacing_xyz = np.repeat(spacing_values[0], 3).astype(np.float32)
        elif spacing_values.size == 3:
            base_spacing_xyz = spacing_values.astype(np.float32)
        else:
            raise ValueError("--spacing expects either 1 value or 3 values (x y z)")
        if np.any(base_spacing_xyz <= 0):
            raise ValueError("--spacing values must all be > 0")

    spacing_xyz = base_spacing_xyz / float(args.resolution_scale)
    if np.any(spacing_xyz <= 0):
        raise ValueError(f"Resolved spacing must be > 0, got {spacing_xyz.tolist()}")
    return spacing_xyz.astype(np.float32), base_spacing_xyz.astype(np.float32)


def build_axis(point_min, point_max, spacing_xyz):
    axes = []
    for dim in range(3):
        spacing = float(spacing_xyz[dim])
        extent = point_max[dim] - point_min[dim]
        count = max(1, int(np.ceil(extent / spacing)))
        axis = point_min[dim] + (np.arange(count, dtype=np.float32) + 0.5) * spacing
        axes.append(axis)
    return axes


def load_axis_npy(path: Path, axis_name: str):
    axis = np.load(path)
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{axis_name} axis must be a non-empty 1D array, got shape {axis.shape}.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{axis_name} axis contains non-finite values.")
    if axis.size > 1:
        diffs = np.diff(axis.astype(np.float64))
        if np.any(diffs <= 0):
            raise ValueError(f"{axis_name} axis must be strictly increasing.")
    return axis.astype(np.float32, copy=False)


def infer_uniform_spacing_from_axis(axis: np.ndarray, axis_name: str):
    if axis.size < 2:
        raise ValueError(
            f"{axis_name} axis must contain at least 2 entries to infer spacing for MHD export."
        )
    diffs = np.diff(axis.astype(np.float64))
    spacing = float(np.mean(diffs))
    atol = max(1e-5, abs(spacing) * 1e-4)
    if not np.allclose(diffs, spacing, rtol=1e-4, atol=atol):
        raise ValueError(
            f"{axis_name} axis must be uniformly spaced for Cartesian MHD export. "
            f"Observed step range: [{float(diffs.min())}, {float(diffs.max())}]"
        )
    return np.float32(spacing)


def resolve_query_grid_spec(args, baked_dataset):
    has_recons_grid = args.recons_common_grid_h5 is not None
    has_axis_files = any(v is not None for v in (args.x_axis_npy, args.y_axis_npy, args.z_axis_npy))
    has_minimal_cartesian = any(v is not None for v in (args.point_min, args.point_max, args.spacing))

    if has_recons_grid and (has_axis_files or has_minimal_cartesian):
        raise ValueError(
            "--recons-common-grid-h5 cannot be combined with --x-axis-npy/--y-axis-npy/--z-axis-npy "
            "or with --point-min/--point-max/--spacing."
        )

    if has_axis_files:
        if not all(v is not None for v in (args.x_axis_npy, args.y_axis_npy, args.z_axis_npy)):
            raise ValueError(
                "When using direct Cartesian axes, you must provide all of "
                "--x-axis-npy, --y-axis-npy, and --z-axis-npy."
            )
        if has_minimal_cartesian:
            raise ValueError(
                "--x-axis-npy/--y-axis-npy/--z-axis-npy cannot be combined with "
                "--point-min/--point-max/--spacing."
            )

        x_axis = load_axis_npy(args.x_axis_npy, "x")
        y_axis = load_axis_npy(args.y_axis_npy, "y")
        z_axis = load_axis_npy(args.z_axis_npy, "z")
        spacing_xyz = np.array([
            infer_uniform_spacing_from_axis(x_axis, "x"),
            infer_uniform_spacing_from_axis(y_axis, "y"),
            infer_uniform_spacing_from_axis(z_axis, "z"),
        ], dtype=np.float32)
        point_min = np.array([
            float(x_axis[0] - 0.5 * spacing_xyz[0]),
            float(y_axis[0] - 0.5 * spacing_xyz[1]),
            float(z_axis[0] - 0.5 * spacing_xyz[2]),
        ], dtype=np.float32)
        point_max = np.array([
            float(x_axis[-1] + 0.5 * spacing_xyz[0]),
            float(y_axis[-1] + 0.5 * spacing_xyz[1]),
            float(z_axis[-1] + 0.5 * spacing_xyz[2]),
        ], dtype=np.float32)
        return {
            "mode": "cartesian",
            "source": "external_axes_npy",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "point_min": point_min,
            "point_max": point_max,
            "spacing_xyz": spacing_xyz,
            "base_spacing_xyz": spacing_xyz.copy(),
        }

    if has_recons_grid:
        bundle_path = args.recons_common_grid_h5
        if not bundle_path.exists():
            raise FileNotFoundError(f"recons_common_grid bundle not found: {bundle_path}")
        with h5py.File(bundle_path, "r") as f:
            required_keys = [
                "data_3d_mask_uint8",
                "coord_mask_xz_zero_based",
                "active_sag_indices_zero_based",
            ]
            missing = [k for k in required_keys if k not in f]
            if missing:
                raise KeyError(
                    f"{bundle_path} is missing required datasets for recons-grid export: {missing}"
                )
            common_shape = tuple(int(v) for v in f["data_3d_mask_uint8"].shape)
            support_voxels = int(f["coord_mask_xz_zero_based"].shape[0])
            active_sag_count = int(f["active_sag_indices_zero_based"].shape[0])
            has_cached_world_coords = "support_world_coords_mm" in f
        return {
            "mode": "recons_common_grid",
            "source": "recons_common_grid_h5",
            "bundle_path": bundle_path,
            "common_shape_x_sag_z": common_shape,
            "support_voxels_per_plane": support_voxels,
            "active_sag_count": active_sag_count,
            "has_cached_world_coords": has_cached_world_coords,
        }

    point_min = (
        np.array(baked_dataset.point_min, dtype=np.float32)
        if args.point_min is None
        else np.array(args.point_min, dtype=np.float32)
    )
    point_max = (
        np.array(baked_dataset.point_max, dtype=np.float32)
        if args.point_max is None
        else np.array(args.point_max, dtype=np.float32)
    )
    spacing_xyz, base_spacing_xyz = resolve_spacing_xyz(args, baked_dataset)
    x_axis, y_axis, z_axis = build_axis(point_min, point_max, spacing_xyz)
    return {
        "mode": "cartesian",
        "source": "baked_dataset_defaults" if not has_minimal_cartesian else "point_min_point_max_spacing",
        "x_axis": np.asarray(x_axis, dtype=np.float32),
        "y_axis": np.asarray(y_axis, dtype=np.float32),
        "z_axis": np.asarray(z_axis, dtype=np.float32),
        "point_min": point_min,
        "point_max": point_max,
        "spacing_xyz": spacing_xyz,
        "base_spacing_xyz": base_spacing_xyz,
    }


def flip_volume_xy(volume_zyx):
    return np.flip(np.flip(volume_zyx, axis=1), axis=2)


def format_bytes(num_bytes):
    num_bytes = int(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def array_nbytes(shape, dtype):
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def estimate_export_bytes(grid_shape_zyx, baked_dataset, save_large_npy, save_gt_exports):
    total = 0

    # Main queried volume: one raw always, plus optional zyx/hzw npy snapshots.
    total += array_nbytes(grid_shape_zyx, np.uint8)
    if save_large_npy:
        total += 2 * array_nbytes(grid_shape_zyx, np.float32)

    if save_gt_exports:
        total_slices = len(baked_dataset.slices) + len(baked_dataset.slices_valid)
        stacked_shape = (total_slices, baked_dataset.px_height, baked_dataset.px_width)
        total += array_nbytes(stacked_shape, np.float32)
        if save_large_npy:
            total += array_nbytes(stacked_shape, np.float32)

    # Small side files: headers, metadata, axis vectors.
    total += 16 * 1024 * 1024
    return total


def ensure_sufficient_disk_space(output_dir: Path, required_bytes: int):
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough free disk space for export.\n"
            f"  Output dir: {output_dir}\n"
            f"  Free space: {format_bytes(free_bytes)}\n"
            f"  Estimated required: {format_bytes(required_bytes)}\n"
            "  Tip: delete older exports, choose another output drive, keep "
            "--save-large-npy and --save-gt-exports disabled."
        )


def save_mhd_array(volume_for_raw, output_dir: Path, base_name: str, dim_sizes, spacing_sizes, element_type="MET_FLOAT"):
    raw_path = output_dir / f"{base_name}.raw"
    mhd_path = output_dir / f"{base_name}.mhd"

    if element_type == "MET_FLOAT":
        array_to_write = volume_for_raw.astype(np.float32, copy=False)
    else:
        array_to_write = volume_for_raw

    # Avoid allocating a second full-volume temporary when the array is a flipped
    # or transposed view with negative / non-contiguous strides.
    try:
        with raw_path.open("wb") as f:
            if array_to_write.flags.c_contiguous:
                array_to_write.tofile(f)
            else:
                for slice_idx in range(array_to_write.shape[0]):
                    np.ascontiguousarray(array_to_write[slice_idx]).tofile(f)
    except OSError as exc:
        partial_size = raw_path.stat().st_size if raw_path.exists() else 0
        free_bytes = shutil.disk_usage(output_dir).free
        if raw_path.exists():
            try:
                raw_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed while writing {raw_path.name}.\n"
            f"  Partial bytes written: {format_bytes(partial_size)}\n"
            f"  Remaining free space: {format_bytes(free_bytes)}\n"
            "  This is usually caused by insufficient disk space."
        ) from exc

    dim_0, dim_1, dim_2 = [int(v) for v in dim_sizes]
    spacing_0, spacing_1, spacing_2 = [float(v) for v in spacing_sizes]

    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_0} {spacing_1} {spacing_2}",
            f"DimSize = {dim_0} {dim_1} {dim_2}",
            f"ElementType = {element_type}",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    mhd_path.write_text(header, encoding="ascii")


def save_mhd(volume_zyx, output_dir: Path, spacing_xyz, point_min_xyz):
    raw_path = output_dir / "volume.raw"
    mhd_path = output_dir / "volume.mhd"

    volume_zyx.astype(np.float32).tofile(raw_path)

    dim_z, dim_y, dim_x = volume_zyx.shape
    spacing_x, spacing_y, spacing_z = [float(v) for v in spacing_xyz]
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_x} {spacing_y} {spacing_z}",
            f"DimSize = {dim_x} {dim_y} {dim_z}",
            "ElementType = MET_FLOAT",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    mhd_path.write_text(header, encoding="ascii")


def save_mhd_named(volume_zyx, output_dir: Path, base_name: str, spacing_xyz, point_min_xyz):
    raw_path = output_dir / f"{base_name}.raw"
    mhd_path = output_dir / f"{base_name}.mhd"

    volume_zyx.astype(np.float32).tofile(raw_path)

    dim_z, dim_y, dim_x = volume_zyx.shape
    spacing_x, spacing_y, spacing_z = [float(v) for v in spacing_xyz]
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_x} {spacing_y} {spacing_z}",
            f"DimSize = {dim_x} {dim_y} {dim_z}",
            "ElementType = MET_FLOAT",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    mhd_path.write_text(header, encoding="ascii")


def convert_grid_zyx_to_mhd(volume_zyx):
    # NeUF axes to exported volume axes:
    # h = -x, z = -z, w = -y

    volume_mhd = volume_zyx
    # volume_hzw = np.transpose(volume_zyx, (1, 0, 2))
    # volume_hzw = np.flip(volume_hzw, axis=0)
    volume_mhd = np.flip(volume_mhd, axis=2)
    # volume_hzw = np.flip(volume_hzw, axis=1)

    return volume_mhd


def save_grid_mhd_zyx(volume_mhd, output_dir: Path, base_name: str, spacing_xyz, element_type="MET_FLOAT"):
    # Match the notebook logic that already reads back correctly in MITK:
    # keep the array in (h, z, w), write it directly to raw,
    # and store DimSize / ElementSpacing in reversed order.
    dim_sizes = (volume_mhd.shape[2], volume_mhd.shape[1], volume_mhd.shape[0])
    spacing_sizes = (spacing_xyz[1], spacing_xyz[2], spacing_xyz[0])
    save_mhd_array(volume_mhd, output_dir, base_name, dim_sizes, spacing_sizes, element_type=element_type)


def save_recons_common_grid_mhd(volume_x_sag_z, output_dir: Path, base_name: str = "volume", mask_x_sag_z=None):
    """Save a volume defined on the recons common grid as MITK-readable MHD/raw.

    The recons common grid is stored internally as (x_common, sagittal, z_common).
    Historic recons3D MITK exports are read correctly when the raw array is
    written as (z_common, x_common, sagittal) with the MHD header declaring
    DimSize = (sagittal, x_common, z_common).
    """
    volume_mitk = np.transpose(volume_x_sag_z, (2, 0, 1))
    x_common, sagittal, z_common = [int(v) for v in volume_x_sag_z.shape]
    save_mhd_array(
        volume_mitk,
        output_dir,
        base_name,
        dim_sizes=(sagittal, x_common, z_common),
        spacing_sizes=(1.0, 1.0, 1.0),
        element_type="MET_UCHAR",
    )

    if mask_x_sag_z is not None:
        mask_mitk = np.transpose(mask_x_sag_z, (2, 0, 1))
        save_mhd_array(
            mask_mitk,
            output_dir,
            f"{base_name}_mask",
            dim_sizes=(sagittal, x_common, z_common),
            spacing_sizes=(1.0, 1.0, 1.0),
            element_type="MET_UCHAR",
        )


def build_recons_world_column(frame_positions, frame_rotmats, depth_axis, sag_value):
    local_column = np.stack(
        (
            depth_axis,
            np.full_like(depth_axis, sag_value, dtype=np.float32),
            np.zeros_like(depth_axis, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    world = np.einsum("dc,fkc->fdk", local_column, frame_rotmats) + frame_positions[:, None, :]
    return world.astype(np.float32, copy=False)


def compute_recons_world_coords_for_sag_slice(compact_bundle, sag_idx):
    I_complet_1 = compact_bundle["I_complet_1"].astype(np.int64, copy=False)
    d_complet_1 = compact_bundle["d_complet_1"].astype(np.float64, copy=False)
    d_complet_1_1 = compact_bundle["d_complet_1_1"].astype(np.float64, copy=False)

    w_primary = 1.0 / np.maximum(d_complet_1, 1e-8)
    w_sagadj = 1.0 / np.maximum(d_complet_1_1, 1e-8)
    sum_w = np.sum(w_primary, axis=1) + 2.0 * np.sum(w_sagadj, axis=1)
    sum_w[sum_w == 0] = 1.0

    world_j = build_recons_world_column(
        compact_bundle["frame_positions_mm"],
        compact_bundle["frame_rotmats"],
        compact_bundle["depth_axis_mm"],
        float(compact_bundle["sag_axis_mm"][sag_idx]),
    )
    world_m1 = build_recons_world_column(
        compact_bundle["frame_positions_mm"],
        compact_bundle["frame_rotmats"],
        compact_bundle["depth_axis_mm"],
        float(compact_bundle["sag_axis_mm"][sag_idx - 1]),
    )
    world_p1 = build_recons_world_column(
        compact_bundle["frame_positions_mm"],
        compact_bundle["frame_rotmats"],
        compact_bundle["depth_axis_mm"],
        float(compact_bundle["sag_axis_mm"][sag_idx + 1]),
    )

    world_j_flat = world_j.reshape(-1, 3)
    world_m1_flat = world_m1.reshape(-1, 3)
    world_p1_flat = world_p1.reshape(-1, 3)

    accum_world = np.zeros((I_complet_1.shape[0], 3), dtype=np.float64)
    for k in range(I_complet_1.shape[1]):
        idx_k = I_complet_1[:, k]
        accum_world += w_primary[:, k, None] * world_j_flat[idx_k]
        accum_world += w_sagadj[:, k, None] * (
            world_m1_flat[idx_k] + world_p1_flat[idx_k]
        )

    return (accum_world / sum_w[:, None]).astype(np.float32)


def query_recons_common_grid(model, ckpt, bundle_path: Path, chunk_size, use_bbox_mask, plane_mask_data=None):
    bb_min, bb_max = to_numpy_bbox(ckpt)
    bb_min_dev = torch.from_numpy(bb_min).to(DEVICE)
    bb_size_dev = torch.from_numpy(bb_max - bb_min).to(DEVICE)

    with h5py.File(bundle_path, "r") as f:
        volume_shape = tuple(int(v) for v in f["data_3d_mask_uint8"].shape)
        volume_x_sag_z = np.zeros(volume_shape, dtype=np.float32)
        mask_x_sag_z = np.asarray(f["data_3d_mask_uint8"], dtype=np.uint8)
        coord_mask_xz = np.asarray(f["coord_mask_xz_zero_based"], dtype=np.int64)
        active_sag_indices = np.asarray(f["active_sag_indices_zero_based"], dtype=np.int64)

        support_world_ds = f["support_world_coords_mm"] if "support_world_coords_mm" in f else None
        compact_bundle = None
        if support_world_ds is None:
            required_compact = [
                "frame_positions_mm",
                "frame_rotmats",
                "depth_axis_mm",
                "sag_axis_mm",
                "I_complet_1",
                "d_complet_1",
                "d_complet_1_1",
            ]
            missing = [k for k in required_compact if k not in f]
            if missing:
                raise KeyError(
                    f"{bundle_path} does not contain cached world coords or the compact mapping fields {missing}."
                )
            compact_bundle = {
                "frame_positions_mm": np.asarray(f["frame_positions_mm"], dtype=np.float32),
                "frame_rotmats": np.asarray(f["frame_rotmats"], dtype=np.float32),
                "depth_axis_mm": np.asarray(f["depth_axis_mm"], dtype=np.float32),
                "sag_axis_mm": np.asarray(f["sag_axis_mm"], dtype=np.float32),
                "I_complet_1": np.asarray(f["I_complet_1"], dtype=np.int32),
                "d_complet_1": np.asarray(f["d_complet_1"], dtype=np.float32),
                "d_complet_1_1": np.asarray(f["d_complet_1_1"], dtype=np.float32),
            }

        total_points = int(len(active_sag_indices) * coord_mask_xz.shape[0])
        total_active = 0

        with torch.no_grad():
            for out_idx, sag_idx in enumerate(tqdm(active_sag_indices, desc="Querying recons common grid")):
                if support_world_ds is not None:
                    query_world = np.asarray(support_world_ds[out_idx], dtype=np.float32)
                else:
                    query_world = compute_recons_world_coords_for_sag_slice(compact_bundle, int(sag_idx))

                active_mask = np.ones((query_world.shape[0],), dtype=bool)
                if use_bbox_mask:
                    active_mask &= np.all((query_world >= bb_min) & (query_world <= bb_max), axis=1)
                if plane_mask_data is not None:
                    active_mask &= compute_sequence_plane_mask(query_world, plane_mask_data)

                active_indices = np.flatnonzero(active_mask)
                if active_indices.size == 0:
                    continue

                total_active += int(active_indices.size)
                query_points = query_world[active_indices]
                point_values = np.zeros((query_world.shape[0],), dtype=np.float32)

                for start in range(0, query_points.shape[0], chunk_size):
                    stop = min(start + chunk_size, query_points.shape[0])
                    batch_points = torch.from_numpy(query_points[start:stop]).to(DEVICE)
                    batch_dirs = torch.zeros_like(batch_points, device=DEVICE)

                    if model.encoding_type != "HASH" or not model.use_encoding:
                        batch_points = ((batch_points - bb_min_dev) / bb_size_dev) * 2.0 - 1.0

                    batch_values = model.query(batch_points, batch_dirs).reshape(-1)
                    point_values[active_indices[start:stop]] = (
                        batch_values.detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                volume_x_sag_z[
                    coord_mask_xz[:, 0],
                    int(sag_idx),
                    coord_mask_xz[:, 1],
                ] = point_values

    if use_bbox_mask or plane_mask_data is not None:
        print(f"Active queried voxels after masking: {total_active} / {total_points}")

    return volume_x_sag_z, mask_x_sag_z, {
        "total_points": total_points,
        "active_points": total_active,
        "active_sag_indices_zero_based": active_sag_indices.tolist(),
        "support_voxels_per_plane": int(coord_mask_xz.shape[0]),
        "used_cached_world_coords": support_world_ds is not None,
    }


def save_stacked_mhd_zyx(volume_zhw, output_dir: Path, base_name: str, spacing_wzh):
    volume_hzw = np.transpose(volume_zhw, (1, 0, 2))
    volume_hzw = np.flip(volume_hzw, axis=0)
    dim_sizes = (volume_zhw.shape[2], volume_zhw.shape[0], volume_zhw.shape[1])
    save_mhd_array(volume_hzw.astype(np.float32), output_dir, base_name, dim_sizes, spacing_wzh, element_type="MET_FLOAT")


def stretch_volume_to_uint8_preserve_zero_background(
    volume: np.ndarray,
) -> tuple[np.ndarray, float | None, float | None, int]:
    volume = np.asarray(volume, dtype=np.float32)
    volume_uint8 = np.zeros_like(volume, dtype=np.uint8)
    foreground_mask = volume != 0.0
    foreground_count = int(np.count_nonzero(foreground_mask))

    if foreground_count == 0:
        return volume_uint8, None, None, foreground_count

    foreground_values = volume[foreground_mask]
    min_val = float(np.min(foreground_values))
    max_val = float(np.max(foreground_values))
    value_range = max_val - min_val

    if value_range > 0.0:
        # Reserve 0 for background and spread foreground into 1..255.
        scaled = ((foreground_values - min_val) / value_range * 254.0) + 1.0
        volume_uint8[foreground_mask] = scaled.astype(np.uint8)
    else:
        volume_uint8[foreground_mask] = 255

    return volume_uint8, min_val, max_val, foreground_count


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
        # utils.get_oriented_points_and_views() builds local slice points as
        # [Y, X, 0], so the slice plane is local z=0 and its normal is the
        # rotated local +Z axis (rotation matrix column 2).
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

    # Re-orient both normals so they point inward, making the mask test symmetric.
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


def query_grid(model, ckpt, x_axis, y_axis, z_axis, chunk_size, use_bbox_mask, plane_mask_data=None):
    bb_min, bb_max = to_numpy_bbox(ckpt)
    bb_min_dev = torch.from_numpy(bb_min).to(DEVICE)
    bb_size_dev = torch.from_numpy(bb_max - bb_min).to(DEVICE)
    x_coords = np.asarray(x_axis, dtype=np.float32)
    y_coords = np.asarray(y_axis, dtype=np.float32)
    volume_zyx = np.zeros((len(z_axis), len(y_axis), len(x_axis)), dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    plane_x = xx.reshape(-1)
    plane_y = yy.reshape(-1)

    total_points = len(z_axis) * len(y_axis) * len(x_axis)
    total_active = 0

    with torch.no_grad():
        for z_idx, z_value in enumerate(tqdm(z_axis, desc="Querying full grid")):
            plane_points = np.stack(
                (
                    plane_x,
                    plane_y,
                    np.full(plane_x.size, z_value, dtype=np.float32),
                ),
                axis=-1,
            )

            active_mask = np.ones((plane_points.shape[0],), dtype=bool)

            if use_bbox_mask:
                active_mask &= np.all((plane_points >= bb_min) & (plane_points <= bb_max), axis=1)

            if plane_mask_data is not None:
                active_mask &= compute_sequence_plane_mask(plane_points, plane_mask_data)

            active_indices = np.flatnonzero(active_mask)
            if active_indices.size == 0:
                continue

            total_active += int(active_indices.size)
            plane_values = np.zeros((plane_points.shape[0],), dtype=np.float32)
            query_points = plane_points[active_indices]

            for start in range(0, query_points.shape[0], chunk_size):
                stop = min(start + chunk_size, query_points.shape[0])
                batch_points = torch.from_numpy(query_points[start:stop]).to(DEVICE)
                batch_dirs = torch.zeros_like(batch_points, device=DEVICE)

                if model.encoding_type != "HASH" or not model.use_encoding:
                    batch_points = ((batch_points - bb_min_dev) / bb_size_dev) * 2.0 - 1.0

                batch_values = model.query(batch_points, batch_dirs).reshape(-1)
                plane_values[active_indices[start:stop]] = batch_values.detach().cpu().numpy().astype(np.float32, copy=False)

            volume_zyx[z_idx] = plane_values.reshape(len(y_axis), len(x_axis))

    if use_bbox_mask or plane_mask_data is not None:
        print(f"Active queried voxels after masking: {total_active} / {total_points}")

    return volume_zyx


def iter_dataset_slices(baked_dataset):
    for slice_idx, slice_info in enumerate(baked_dataset.slices):
        slice_pixels = baked_dataset.get_slice_pixels(slice_idx)
        slice_pixels = torch.reshape(slice_pixels, (baked_dataset.px_height, baked_dataset.px_width))
        yield slice_info.position, slice_info.rotation, slice_pixels.detach().cpu().numpy().astype(np.float32)

    for slice_idx, slice_info in enumerate(baked_dataset.slices_valid):
        slice_pixels = baked_dataset.get_slice_valid_pixels(slice_idx)
        slice_pixels = torch.reshape(slice_pixels, (baked_dataset.px_height, baked_dataset.px_width))
        yield slice_info.position, slice_info.rotation, slice_pixels.detach().cpu().numpy().astype(np.float32)


def build_stacked_slice_volume(baked_dataset):
    stacked_slices = []
    for _, _, slice_pixels in iter_dataset_slices(baked_dataset):
        stacked_slices.append(slice_pixels.astype(np.float32))
    if not stacked_slices:
        return None
    return np.stack(stacked_slices, axis=0)


def voxelize_gt_observations(baked_dataset, point_min, point_max, spacing_xyz):
    grid_shape_xyz = []
    for dim in range(3):
        extent = point_max[dim] - point_min[dim]
        count = max(1, int(np.ceil(extent / float(spacing_xyz[dim]))))
        grid_shape_xyz.append(count)

    sum_volume = np.zeros((grid_shape_xyz[2], grid_shape_xyz[1], grid_shape_xyz[0]), dtype=np.float32)
    count_volume = np.zeros_like(sum_volume, dtype=np.uint32)

    base_x, base_y = get_base_points(
        baked_dataset.width,
        baked_dataset.height,
        baked_dataset.px_width,
        baked_dataset.px_height,
        offset_x_mm=baked_dataset.roi_offset_x_mm,
        offset_y_mm=baked_dataset.roi_offset_y_mm,
    )

    total_slices = len(baked_dataset.slices) + len(baked_dataset.slices_valid)
    for position, rotation, slice_pixels in tqdm(
        iter_dataset_slices(baked_dataset),
        total=total_slices,
        desc="Voxelizing GT observations",
    ):
        points_world, _ = get_oriented_points_and_views(base_x, base_y, position, rotation)
        flat_values = slice_pixels.reshape(-1)

        voxel_coords = np.floor((points_world - point_min[None, :]) / spacing_xyz[None, :]).astype(np.int64)
        inside_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_shape_xyz[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_shape_xyz[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_shape_xyz[2])
        )
        if not np.any(inside_mask):
            continue

        voxel_coords = voxel_coords[inside_mask]
        flat_values = flat_values[inside_mask]

        vx = voxel_coords[:, 0]
        vy = voxel_coords[:, 1]
        vz = voxel_coords[:, 2]

        np.add.at(sum_volume, (vz, vy, vx), flat_values)
        np.add.at(count_volume, (vz, vy, vx), 1)

    observed_mask = count_volume > 0
    fused_volume = np.zeros_like(sum_volume, dtype=np.float32)
    fused_volume[observed_mask] = sum_volume[observed_mask] / count_volume[observed_mask].astype(np.float32)

    return fused_volume, count_volume, observed_mask.astype(np.uint8)


def main():
    args = parse_args()
    if args.output_exact:
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_dated_output_dir(args.output, args.ckpt)

    ckpt, model = load_checkpoint(args.ckpt)
    baked_dataset, dataset_path = load_dataset_from_ckpt(ckpt)
    bb_min, bb_max = to_numpy_bbox(ckpt)
    grid_spec = resolve_query_grid_spec(args, baked_dataset)

    use_sequence_plane_mask = not args.disable_sequence_plane_mask
    plane_mask_data = get_sequence_plane_mask_data(baked_dataset, dataset_path) if use_sequence_plane_mask else None
    if use_sequence_plane_mask:
        if plane_mask_data is None:
            print("Sequence plane mask requested, but the baked dataset does not contain valid front/back plane metadata. Falling back to unmasked query.")
        else:
            print("Using sequence plane mask:")
            print(f"  Source: {plane_mask_data.get('source', 'unknown')}")
            print(f"  Front plane point: {plane_mask_data['front_point']}")
            print(f"  Front plane normal (inward): {plane_mask_data['front_normal']}")
            print(f"  Back plane point: {plane_mask_data['back_point']}")
            print(f"  Back plane normal (inward): {plane_mask_data['back_normal']}")
    if grid_spec["mode"] == "recons_common_grid":
        common_shape = tuple(int(v) for v in grid_spec["common_shape_x_sag_z"])
        required_bytes = 2 * array_nbytes(common_shape, np.uint8)
        if args.save_large_npy:
            required_bytes += array_nbytes(common_shape, np.float32)
            required_bytes += array_nbytes(common_shape, np.uint8)
        required_bytes += 16 * 1024 * 1024

        print(f"Grid source mode: {grid_spec['source']}")
        print(f"recons_common_grid.h5: {grid_spec['bundle_path']}")
        print(f"Common-grid shape (x, sagittal, z): {common_shape}")
        print(f"Support voxels per active sagittal slice: {grid_spec['support_voxels_per_plane']:,}")
        print(f"Active sagittal slices: {grid_spec['active_sag_count']:,}")
        print(f"Using cached world coords from bundle: {grid_spec['has_cached_world_coords']}")
        print(f"Estimated export disk usage: {format_bytes(required_bytes)}")
        ensure_sufficient_disk_space(output_dir, required_bytes)
        if args.save_gt_exports:
            print("GT stacked-slice exports are not applicable to recons_common_grid mode and will be skipped.")

        volume_x_sag_z, mask_x_sag_z, recons_query_meta = query_recons_common_grid(
            model,
            ckpt,
            grid_spec["bundle_path"],
            args.chunk_size,
            args.use_bbox_mask,
            plane_mask_data=plane_mask_data,
        )
        print(f"Queried common-grid volume shape (x, sagittal, z): {volume_x_sag_z.shape}")
        (
            volume_uint8,
            volume_export_min,
            volume_export_max,
            volume_foreground_count,
        ) = stretch_volume_to_uint8_preserve_zero_background(volume_x_sag_z)
        if volume_foreground_count > 0:
            print(
                "Export intensity normalization: "
                f"background stays 0, foreground min-max "
                f"[{volume_export_min:.6f}, {volume_export_max:.6f}] -> [1, 255]"
            )
        else:
            print("Export intensity normalization: volume is all background zeros")

        if args.save_large_npy:
            np.save(output_dir / "volume_x_sag_z.npy", volume_x_sag_z)
            np.save(output_dir / "volume_mask_x_sag_z.npy", mask_x_sag_z)

        save_recons_common_grid_mhd(
            volume_uint8,
            output_dir,
            base_name="volume",
            mask_x_sag_z=mask_x_sag_z,
        )

        metadata = {
            "ckpt": str(args.ckpt),
            "dataset_pkl": str(dataset_path),
            "grid_source_mode": grid_spec["source"],
            "grid_source_bundle_h5": str(grid_spec["bundle_path"]),
            "bounding_box_min_mm": bb_min.tolist(),
            "bounding_box_max_mm": bb_max.tolist(),
            "shape_x_sag_z": list(volume_x_sag_z.shape),
            "shape_hzw": [int(volume_x_sag_z.shape[1]), int(volume_x_sag_z.shape[0]), int(volume_x_sag_z.shape[2])],
            "export_element_type": "MET_UCHAR",
            "export_intensity_normalization": "preserve_zero_background_foreground_min_max_to_1_255",
            "export_intensity_normalization_min": volume_export_min,
            "export_intensity_normalization_max": volume_export_max,
            "export_intensity_normalization_foreground_voxels": volume_foreground_count,
            "mhd_grid_axis_order": ["sagittal", "x_common", "z_common"],
            "mhd_grid_axis_mapping": {"sagittal": "common-grid sagittal index", "x_common": "common-grid x", "z_common": "common-grid z"},
            "mitk_raw_array_shape_z_x_sag": [int(volume_x_sag_z.shape[2]), int(volume_x_sag_z.shape[0]), int(volume_x_sag_z.shape[1])],
            "use_bbox_mask": args.use_bbox_mask,
            "use_sequence_plane_mask": use_sequence_plane_mask and bool(plane_mask_data is not None),
            "sequence_plane_mask_source": plane_mask_data.get("source") if plane_mask_data is not None else None,
            "front_plane_point_mm": plane_mask_data["front_point"].tolist() if plane_mask_data is not None else None,
            "front_plane_normal_inward": plane_mask_data["front_normal"].tolist() if plane_mask_data is not None else None,
            "back_plane_point_mm": plane_mask_data["back_point"].tolist() if plane_mask_data is not None else None,
            "back_plane_normal_inward": plane_mask_data["back_normal"].tolist() if plane_mask_data is not None else None,
            "save_large_npy": args.save_large_npy,
            "save_gt_exports": False,
            "estimated_export_bytes": int(required_bytes),
            "recons_support_mask_voxels": int(np.count_nonzero(mask_x_sag_z)),
            "recons_query_total_points": int(recons_query_meta["total_points"]),
            "recons_query_active_points": int(recons_query_meta["active_points"]),
            "recons_support_voxels_per_plane": int(recons_query_meta["support_voxels_per_plane"]),
            "recons_active_sag_indices_zero_based": recons_query_meta["active_sag_indices_zero_based"],
            "recons_used_cached_world_coords": bool(recons_query_meta["used_cached_world_coords"]),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved recons common-grid volume to: {output_dir}")
        print(f"Saved volume shape (x, sagittal, z): {volume_x_sag_z.shape}")
        print(f"MHD saved to: {output_dir / 'volume.mhd'}")
        print(f"Mask MHD saved to: {output_dir / 'volume_mask.mhd'}")
        return

    point_min = grid_spec["point_min"]
    point_max = grid_spec["point_max"]
    spacing_xyz = grid_spec["spacing_xyz"]
    base_spacing_xyz = grid_spec["base_spacing_xyz"]
    x_axis = grid_spec["x_axis"]
    y_axis = grid_spec["y_axis"]
    z_axis = grid_spec["z_axis"]

    print(f"Grid source mode: {grid_spec['source']}")
    print(f"Base spacing (x, y, z) mm: {tuple(float(v) for v in base_spacing_xyz)}")
    print(f"Final spacing (x, y, z) mm: {tuple(float(v) for v in spacing_xyz)}")
    print(f"Grid shape (x, y, z): {(len(x_axis), len(y_axis), len(z_axis))}")
    required_bytes = estimate_export_bytes(
        (len(z_axis), len(y_axis), len(x_axis)),
        baked_dataset,
        save_large_npy=args.save_large_npy,
        save_gt_exports=args.save_gt_exports,
    )
    print(f"Estimated export disk usage: {format_bytes(required_bytes)}")
    ensure_sufficient_disk_space(output_dir, required_bytes)

    volume_zyx = query_grid(
        model,
        ckpt,
        x_axis,
        y_axis,
        z_axis,
        args.chunk_size,
        args.use_bbox_mask,
        plane_mask_data=plane_mask_data,
    )
    print(f"Queried grid volume shape (z, y, x): {volume_zyx.shape}")
    if args.save_large_npy:
        np.save(output_dir / "volume_zyx.npy", volume_zyx)
    volume = convert_grid_zyx_to_mhd(volume_zyx).astype(np.float32, copy=False)
    (
        volume_uint8,
        volume_export_min,
        volume_export_max,
        volume_foreground_count,
    ) = stretch_volume_to_uint8_preserve_zero_background(volume)
    if volume_foreground_count > 0:
        print(
            "Export intensity normalization: "
            f"background stays 0, foreground min-max "
            f"[{volume_export_min:.6f}, {volume_export_max:.6f}] -> [1, 255]"
        )
    else:
        print("Export intensity normalization: volume is all background zeros")
    if args.save_large_npy:
        np.save(output_dir / "volume.npy", volume)
        np.save(output_dir / "x_axis.npy", x_axis)
        np.save(output_dir / "y_axis.npy", y_axis)
        np.save(output_dir / "z_axis.npy", z_axis)
    save_grid_mhd_zyx(volume_uint8, output_dir, "volume", spacing_xyz, element_type="MET_UCHAR")

    stacked_slice_volume = None

    if args.save_gt_exports:
        stacked_slice_volume = build_stacked_slice_volume(baked_dataset)
    if stacked_slice_volume is not None:
        stacked_spacing_xyz = np.array(
            [
                float(baked_dataset.roi_px_size_width_mm),
                float(baked_dataset.roi_px_size_height_mm),
                max(
                    float(baked_dataset.scan_length_mm) / max(stacked_slice_volume.shape[0] - 1, 1),
                    min(
                        float(baked_dataset.roi_px_size_width_mm),
                        float(baked_dataset.roi_px_size_height_mm),
                    ),
                ),
            ],
            dtype=np.float32,
        )
        if args.save_large_npy:
            np.save(output_dir / "gt_stacked_slices.npy", stacked_slice_volume)
        save_stacked_mhd_zyx(
            stacked_slice_volume,
            output_dir,
            "gt_stacked_slices",
            (
                float(baked_dataset.roi_px_size_width_mm),
                float(stacked_spacing_xyz[2]),
                float(baked_dataset.roi_px_size_height_mm),
            ),
        )

    metadata = {
        "ckpt": str(args.ckpt),
        "dataset_pkl": str(dataset_path),
        "grid_source_mode": grid_spec["source"],
        "base_spacing_mm_xyz": base_spacing_xyz.tolist(),
        "spacing_mm_xyz": spacing_xyz.tolist(),
        "resolution_scale": float(args.resolution_scale),
        "mhd_grid_axis_order": ["h", "z", "w"],
        "mhd_grid_axis_mapping": {"h": "-x", "z": "-z", "w": "-y"},
        "point_min_mm": point_min.tolist(),
        "point_max_mm": point_max.tolist(),
        "dataset_point_min_mm": np.array(baked_dataset.point_min, dtype=np.float32).tolist(),
        "dataset_point_max_mm": np.array(baked_dataset.point_max, dtype=np.float32).tolist(),
        "dataset_roi_px_size_mm": [
            float(baked_dataset.roi_px_size_width_mm),
            float(baked_dataset.roi_px_size_height_mm),
        ],
        "bounding_box_min_mm": bb_min.tolist(),
        "bounding_box_max_mm": bb_max.tolist(),
        "shape_hzw": list(volume.shape),
        "export_element_type": "MET_UCHAR",
        "export_intensity_normalization": "preserve_zero_background_foreground_min_max_to_1_255",
        "export_intensity_normalization_min": volume_export_min,
        "export_intensity_normalization_max": volume_export_max,
        "export_intensity_normalization_foreground_voxels": volume_foreground_count,
        "gt_stacked_slices_shape_zhw": list(stacked_slice_volume.shape) if stacked_slice_volume is not None else None,
        "gt_stacked_mhd_axis_order": ["width", "Z", "height"] if stacked_slice_volume is not None else None,
        "use_bbox_mask": args.use_bbox_mask,
        "use_sequence_plane_mask": use_sequence_plane_mask and bool(plane_mask_data is not None),
        "sequence_plane_mask_source": plane_mask_data.get("source") if plane_mask_data is not None else None,
        "front_plane_point_mm": plane_mask_data["front_point"].tolist() if plane_mask_data is not None else None,
        "front_plane_normal_inward": plane_mask_data["front_normal"].tolist() if plane_mask_data is not None else None,
        "back_plane_point_mm": plane_mask_data["back_point"].tolist() if plane_mask_data is not None else None,
        "back_plane_normal_inward": plane_mask_data["back_normal"].tolist() if plane_mask_data is not None else None,
        "save_large_npy": args.save_large_npy,
        "save_gt_exports": args.save_gt_exports,
        "estimated_export_bytes": int(required_bytes),
    }
    if grid_spec["source"] == "external_axes_npy":
        metadata["x_axis_npy"] = str(args.x_axis_npy)
        metadata["y_axis_npy"] = str(args.y_axis_npy)
        metadata["z_axis_npy"] = str(args.z_axis_npy)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved full grid to: {output_dir}")
    print(f"Saved volume shape (h, z, w): {volume.shape}")
    print(f"MHD saved to: {output_dir / 'volume.mhd'}")


if __name__ == "__main__":
    main()
