"""
Export a full-grid volume using saved recons3D geometry and interpolation.

The output voxel lattice follows export_full_grid_from_ckpt.py:
- point_min / point_max from the checkpoint baked dataset
- same spacing logic
- same exported MHD axis convention

The reconstruction logic follows the intended recons3D order:
1. Put all known observed points onto the full grid.
2. Rebuild the recons3D 2.5D interpolation in its saved common coordinate
   system using the saved manual points / sag-plane parameters.
3. Voxelize the intermediate reconstructed points back into the full grid.
4. Fill the remaining blank positions plane-wise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import dataset
import h5py
import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils import get_base_points

# python export_recons3d_style_baseline.py --ckpt latest/ckpt.pkl --output exports/recons3d_style
LOAD_DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class FrameRecord:
    position: np.ndarray
    rotmat: np.ndarray
    frame_index: int


@dataclass(frozen=True)
class ReconsFiles:
    ref: str
    data_recal_mat: Path
    recons_points_mat: Path
    recons_sagplan_mat: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a full-grid volume using saved recons3D interpolation and the ckpt-aligned lattice."
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
        help="Number of full-grid hole-fill queries processed per batch (default: 100000).",
    )
    parser.add_argument(
        "--hole-fill-k", type=int, default=4,
        help="Number of in-plane neighbours used to fill residual holes after recons voxelization.",
    )
    parser.add_argument(
        "--disable-hole-fill",
        action="store_true",
        help="Skip the final in-plane hole-filling pass.",
    )
    parser.add_argument(
        "--disable-sequence-plane-mask",
        action="store_true",
        dest="disable_sequence_plane_mask",
        help="Disable the stored first/last slice sweep-plane mask.",
    )
    parser.add_argument(
        "--data-recal-mat", type=Path, default=None,
        help="Override the source recons data_recal_*.mat file.",
    )
    parser.add_argument(
        "--recons-points-mat", type=Path, default=None,
        help="Override the saved recons reconstruction-points .mat file.",
    )
    parser.add_argument(
        "--recons-sagplan-mat", type=Path, default=None,
        help="Override the saved recons sag-plan parameters .mat file.",
    )
    parser.add_argument(
        "--recons-ref", type=str, default=None,
        help="Explicit recons ref stem such as Patient0_J35_2.",
    )
    parser.add_argument(
        "--delta-x-cc", type=float, default=None,
        help="Override recons sagittal reference spacing in mm/px.",
    )
    parser.add_argument(
        "--delta-x-seqdyn", type=float, default=None,
        help="Override recons dynamic sequence spacing in mm/px.",
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


def load_h5_or_mat_array(path: Path, key: str) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            return np.array(f[key])
    except OSError:
        data = sio.loadmat(str(path))
        return np.array(data[key])


def autodetect_recons_files(dataset_path: Path, args) -> ReconsFiles:
    if args.data_recal_mat and args.recons_points_mat and args.recons_sagplan_mat:
        if args.recons_ref is None:
            ref = args.recons_points_mat.name.removeprefix("data_3D_").removesuffix("_reconstruction_points.mat")
        else:
            ref = args.recons_ref
        return ReconsFiles(
            ref=ref,
            data_recal_mat=args.data_recal_mat,
            recons_points_mat=args.recons_points_mat,
            recons_sagplan_mat=args.recons_sagplan_mat,
        )

    patient_dir = dataset_path.parents[1].name
    pre_root = dataset_path.parents[3]
    recons_dir = pre_root / "Reconstruction_3D" / patient_dir
    recal_dir = pre_root / "Recalage" / patient_dir

    points_map = {
        p.name.removeprefix("data_3D_").removesuffix("_reconstruction_points.mat"): p
        for p in sorted(recons_dir.glob("data_3D_*_reconstruction_points.mat"))
    }
    sagplan_map = {
        p.name.removeprefix("data_3D_").removesuffix("_sagplan_params.mat"): p
        for p in sorted(recons_dir.glob("data_3D_*_sagplan_params.mat"))
    }
    recal_map = {}
    for p in sorted(recal_dir.glob("data_recal_*_d_*.mat")):
        if p.name.endswith("_matlab.mat"):
            continue
        ref = p.name.removeprefix("data_recal_").split("_d_", 1)[0]
        recal_map[ref] = p

    refs = sorted(set(points_map) & set(sagplan_map) & set(recal_map))
    if args.recons_ref is not None:
        if args.recons_ref not in refs:
            raise FileNotFoundError(f"Could not find saved recons files for ref '{args.recons_ref}'. Available refs: {refs}")
        ref = args.recons_ref
    else:
        if len(refs) != 1:
            raise FileNotFoundError(
                "Could not auto-detect a unique saved recons ref. "
                f"Available refs: {refs}. Pass --recons-ref / --data-recal-mat / --recons-points-mat / --recons-sagplan-mat."
            )
        ref = refs[0]

    return ReconsFiles(
        ref=ref,
        data_recal_mat=args.data_recal_mat or recal_map[ref],
        recons_points_mat=args.recons_points_mat or points_map[ref],
        recons_sagplan_mat=args.recons_sagplan_mat or sagplan_map[ref],
    )


def load_saved_recons_inputs(recons_files: ReconsFiles):
    points_mat = sio.loadmat(str(recons_files.recons_points_mat))
    coord_pts_img_ref = np.asarray(points_mat["coord_pts_img_ref"], dtype=np.float64)
    coord_pts_img_seqdyn = np.asarray(points_mat["coord_pts_img_seqdyn"], dtype=np.float64)

    P = load_h5_or_mat_array(recons_files.recons_sagplan_mat, "P").astype(np.float64)
    tr_coord = load_h5_or_mat_array(recons_files.recons_sagplan_mat, "tr_coord").astype(np.float64).reshape(-1)
    idx_sag = int(np.asarray(load_h5_or_mat_array(recons_files.recons_sagplan_mat, "idx_sag")).reshape(-1)[0])
    data_recal = load_h5_or_mat_array(recons_files.data_recal_mat, "data_recal").astype(np.float32)

    return {
        "coord_pts_img_ref": coord_pts_img_ref,
        "coord_pts_img_seqdyn": coord_pts_img_seqdyn,
        "P": P,
        "tr_coord": tr_coord,
        "idx_sag": idx_sag,
        "data_recal": data_recal,
    }


def load_frame_records_from_infos(infos_json_path: Path, baked_dataset, expected_frames: int):
    infos = json.loads(infos_json_path.read_text(encoding="utf-8"))
    frame_keys = sorted([k for k in infos.keys() if k != "infos"], key=lambda x: int(x))
    if len(frame_keys) < expected_frames:
        raise ValueError(
            f"infos.json contains only {len(frame_keys)} frames, but data_recal expects {expected_frames}."
        )

    reverse_quat = bool(getattr(baked_dataset, "reverse_quat", False))
    records = []
    for frame_key in frame_keys[:expected_frames]:
        frame = infos[frame_key]
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
        records.append(
            FrameRecord(
                position=point,
                rotmat=quat.as_rotmat().astype(np.float32),
                frame_index=int(frame_key),
            )
        )
    return records


def build_local_pixel_geometry(baked_dataset):
    base_x, base_y = get_base_points(
        baked_dataset.width,
        baked_dataset.height,
        baked_dataset.px_width,
        baked_dataset.px_height,
        offset_x_mm=baked_dataset.roi_offset_x_mm,
        offset_y_mm=baked_dataset.roi_offset_y_mm,
    )
    depth_axis = base_y.reshape(baked_dataset.px_height, baked_dataset.px_width)[:, 0].astype(np.float32)
    sag_axis = base_x.reshape(baked_dataset.px_height, baked_dataset.px_width)[0].astype(np.float32)
    local_points = np.stack((base_y, base_x, np.zeros_like(base_x)), axis=1).astype(np.float32, copy=False)
    return local_points, depth_axis, sag_axis


def transform_coord(coord_pts_img_sag, coord_pts_img_seqdyn, delta_X_img_sag, delta_X_seqdyn, idx_sag, P, tr_coord):
    scale_factor = delta_X_img_sag / delta_X_seqdyn
    coord_pts_img_sag_scaled = coord_pts_img_sag * scale_factor
    coord_pts_img_sag_tr = np.column_stack((coord_pts_img_sag_scaled[:, 1], coord_pts_img_sag_scaled[:, 0]))

    pts_seq_yx = np.column_stack((coord_pts_img_seqdyn[:, 1], coord_pts_img_seqdyn[:, 0]))
    num_pts = len(pts_seq_yx)
    ones_col = np.full((num_pts, 1), idx_sag)
    pts_3d = np.vstack((pts_seq_yx[:, 0], ones_col.flatten(), pts_seq_yx[:, 1]))
    pts_3d_transformed = P @ (pts_3d + tr_coord.reshape(3, 1) - 1)
    coord_pts_img_seqdyn_tr = pts_3d_transformed[[0, 2], :].T
    return coord_pts_img_sag_tr.astype(np.float64), coord_pts_img_seqdyn_tr.astype(np.float64)


def determine_transform_v2(coord_pts_img_ref, coord_pts_img_seqdyn, data_recal):
    sz3 = data_recal.shape[2]
    b_inf = np.array([-30, 200, -0.08, -2, -80, (-np.pi / 2) / sz3, -np.pi / 3], dtype=np.float64)
    b_sup = np.array([80, 800, 0.08, 2, 30, (np.pi / 2) / sz3, np.pi / 3], dtype=np.float64)
    x0 = np.array([0, 350, 0, 0, 0, 0, 0], dtype=np.float64)

    r = coord_pts_img_seqdyn[:, 0].astype(np.float64)
    t = coord_pts_img_seqdyn[:, 1].astype(np.float64)
    ref_x = coord_pts_img_ref[:, 0].astype(np.float64)
    ref_y = coord_pts_img_ref[:, 1].astype(np.float64)

    def residual_fn(x):
        res_x = x[0] + x[2] * t + (r + x[4]) * np.cos(t * x[5] + x[6]) - ref_x
        res_y = x[1] + x[3] * t + (r + x[4]) * np.sin(t * x[5] + x[6]) - ref_y
        res = np.empty(res_x.size * 2, dtype=np.float64)
        res[0::2] = res_x
        res[1::2] = res_y
        return res

    res = least_squares(
        residual_fn,
        x0,
        bounds=(b_inf, b_sup),
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=2000,
        diff_step=np.full_like(x0, 1e-6, dtype=np.float64),
        loss="linear",
    )
    return res.x.astype(np.float64)


def data_transform_v2(data_recal, X_transform):
    Cx, Ct, vx, vt, delta, omega, theta = X_transform

    n_rows = data_recal.shape[0]
    n_cols = data_recal.shape[2]

    r = np.arange(1, n_rows + 1, dtype=np.float64).reshape(-1, 1)
    t = np.arange(1, n_cols + 1, dtype=np.float64).reshape(1, -1)

    xp = Cx + vx * t + (r + delta) * np.cos(t * omega + theta)
    zp = Ct + vt * t + (r + delta) * np.sin(t * omega + theta)

    offset_x = np.abs(np.minimum(np.min(xp), 0.0)) + 1.0
    offset_z = np.abs(np.minimum(np.min(zp), 0.0)) + 1.0
    xp += offset_x
    zp += offset_z
    return xp.astype(np.float32), zp.astype(np.float32), float(offset_x), float(offset_z)


def build_recons_mask_2d(xp, zp, pts_aug=9, closing_size=3):
    x_max = int(np.ceil(float(np.max(xp)))) + 1
    z_max = int(np.ceil(float(np.max(zp)))) + 1
    mask = np.zeros((x_max, z_max), dtype=bool)

    x_orig = np.rint(xp).astype(np.int64)
    z_orig = np.rint(zp).astype(np.int64)
    valid = (x_orig >= 0) & (x_orig < x_max) & (z_orig >= 0) & (z_orig < z_max)
    mask[x_orig[valid], z_orig[valid]] = True

    def interpolate_line(coords_start, coords_end, n_interp):
        t = np.linspace(0.0, 1.0, n_interp + 1, dtype=np.float32)[:-1][:, None]
        interp = (1.0 - t) * coords_start[None, :] + t * coords_end[None, :]
        return interp.reshape(-1)

    xp_top = interpolate_line(xp[-1, :-1], xp[-1, 1:], pts_aug)
    zp_top = interpolate_line(zp[-1, :-1], zp[-1, 1:], pts_aug)
    xp_bottom = interpolate_line(xp[0, :-1], xp[0, 1:], pts_aug)
    zp_bottom = interpolate_line(zp[0, :-1], zp[0, 1:], pts_aug)

    for x_interp, z_interp in ((xp_top, zp_top), (xp_bottom, zp_bottom)):
        xr = np.rint(x_interp).astype(np.int64)
        zr = np.rint(z_interp).astype(np.int64)
        valid_interp = (xr >= 0) & (xr < x_max) & (zr >= 0) & (zr < z_max)
        mask[xr[valid_interp], zr[valid_interp]] = True

    filled = binary_fill_holes(mask)
    closed = binary_closing(filled, structure=np.ones((closing_size, closing_size), dtype=bool))
    return closed.astype(np.uint8)


def coord_to_fill_fortran(mask_2d):
    idx_mask = np.flatnonzero(mask_2d.ravel(order="F") > 0)
    X, Z = mask_2d.shape

    x_coords = np.repeat(np.arange(1, X + 1, dtype=np.float32)[:, None], Z, axis=1)
    z_coords = np.repeat(np.arange(1, Z + 1, dtype=np.float32)[None, :], X, axis=0)

    coord_mask = np.column_stack((
        x_coords.ravel(order="F")[idx_mask],
        z_coords.ravel(order="F")[idx_mask],
    ))
    return coord_mask.astype(np.float32), idx_mask.astype(np.int64)


def precompute_recons_neighbours(coord_mask_xz, xp, zp, nb_pts=3):
    if nb_pts != 3:
        raise ValueError("This exporter currently supports the recons3D default nb_pts=3 only.")

    n_voxels = coord_mask_xz.shape[0]
    n_frames, n_depth = xp.shape

    best_mid = np.full((n_voxels,), -1, dtype=np.int32)
    best_first = np.full((n_voxels,), np.inf, dtype=np.float32)
    best_idxs = np.zeros((n_voxels, nb_pts), dtype=np.int32)
    best_dists = np.full((n_voxels, nb_pts), np.inf, dtype=np.float32)

    for depth_idx in tqdm(range(n_depth), desc="Stage 2 KNN columns (pass 1/2)"):
        points_i = np.column_stack((xp[:, depth_idx], zp[:, depth_idx])).astype(np.float32, copy=False)
        tree = cKDTree(points_i)
        dists, idxs = tree.query(coord_mask_xz, k=nb_pts, workers=-1)
        if nb_pts == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        update = dists[:, 0] < best_first
        if np.any(update):
            best_first[update] = dists[update, 0].astype(np.float32, copy=False)
            best_mid[update] = depth_idx
            best_idxs[update] = idxs[update].astype(np.int32, copy=False)
            best_dists[update] = dists[update].astype(np.float32, copy=False)

    left_col = np.where(best_mid > 0, best_mid - 1, -1).astype(np.int32)
    right_col = np.where(best_mid < n_depth - 1, best_mid + 1, -1).astype(np.int32)

    left_first = np.full((n_voxels,), np.inf, dtype=np.float32)
    right_first = np.full((n_voxels,), np.inf, dtype=np.float32)
    left_idxs = np.zeros((n_voxels, nb_pts), dtype=np.int32)
    right_idxs = np.zeros((n_voxels, nb_pts), dtype=np.int32)
    left_dists = np.full((n_voxels, nb_pts), np.inf, dtype=np.float32)
    right_dists = np.full((n_voxels, nb_pts), np.inf, dtype=np.float32)

    needed_cols = sorted(set(left_col[left_col >= 0].tolist()) | set(right_col[right_col >= 0].tolist()))
    for depth_idx in tqdm(needed_cols, desc="Stage 2 KNN columns (pass 2/2)"):
        points_i = np.column_stack((xp[:, depth_idx], zp[:, depth_idx])).astype(np.float32, copy=False)
        tree = cKDTree(points_i)
        dists, idxs = tree.query(coord_mask_xz, k=nb_pts, workers=-1)
        if nb_pts == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        mask_left = left_col == depth_idx
        if np.any(mask_left):
            left_first[mask_left] = dists[mask_left, 0].astype(np.float32, copy=False)
            left_idxs[mask_left] = idxs[mask_left].astype(np.int32, copy=False)
            left_dists[mask_left] = dists[mask_left].astype(np.float32, copy=False)

        mask_right = right_col == depth_idx
        if np.any(mask_right):
            right_first[mask_right] = dists[mask_right, 0].astype(np.float32, copy=False)
            right_idxs[mask_right] = idxs[mask_right].astype(np.int32, copy=False)
            right_dists[mask_right] = dists[mask_right].astype(np.float32, copy=False)

    choose_right = np.zeros((n_voxels,), dtype=bool)
    interior = (best_mid > 0) & (best_mid < n_depth - 1)
    choose_right[interior] = right_first[interior] < left_first[interior]
    choose_right[best_mid == 0] = True

    neighbour_col = np.where(choose_right, right_col, left_col).astype(np.int32)
    neighbour_idxs = np.where(choose_right[:, None], right_idxs, left_idxs).astype(np.int32)
    neighbour_dists = np.where(choose_right[:, None], right_dists, left_dists).astype(np.float32)

    base_depth_cols = np.concatenate([
        np.repeat(best_mid[:, None], nb_pts, axis=1),
        np.repeat(neighbour_col[:, None], nb_pts, axis=1),
    ], axis=1).astype(np.int32)
    base_frame_idxs = np.concatenate([best_idxs, neighbour_idxs], axis=1).astype(np.int32)
    base_dists = np.concatenate([best_dists, neighbour_dists], axis=1).astype(np.float32)

    return {
        "coord_mask_xz": coord_mask_xz.astype(np.float32),
        "best_mid_col": best_mid.astype(np.int32),
        "neighbour_col": neighbour_col.astype(np.int32),
        "base_depth_cols": base_depth_cols,
        "base_frame_idxs": base_frame_idxs,
        "base_dists": base_dists,
    }


def build_grid_shape_zyx(point_min, point_max, spacing_xyz):
    counts = []
    for dim in range(3):
        extent = point_max[dim] - point_min[dim]
        counts.append(max(1, int(np.ceil(extent / float(spacing_xyz[dim])))))
    return counts[2], counts[1], counts[0]


def voxelize_observed_recal(data_recal, frame_records, local_points, point_min, point_max, spacing_xyz):
    grid_shape_zyx = build_grid_shape_zyx(point_min, point_max, spacing_xyz)
    sum_volume = np.zeros(grid_shape_zyx, dtype=np.float32)
    count_volume = np.zeros(grid_shape_zyx, dtype=np.uint16)

    num_frames = min(data_recal.shape[0], len(frame_records))
    for frame_idx in tqdm(range(num_frames), desc="Stage 1 observed voxelization"):
        frame = frame_records[frame_idx]
        points_world = local_points @ frame.rotmat.T + frame.position[None, :]
        flat_values = data_recal[frame_idx].T.reshape(-1).astype(np.float32, copy=False)

        voxel_coords = np.floor((points_world - point_min[None, :]) / spacing_xyz[None, :]).astype(np.int64)
        inside_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_shape_zyx[2]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_shape_zyx[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_shape_zyx[0])
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

    filled_mask = count_volume > 0
    volume_zyx = np.zeros_like(sum_volume, dtype=np.float32)
    volume_zyx[filled_mask] = sum_volume[filled_mask] / count_volume[filled_mask].astype(np.float32)
    return volume_zyx, filled_mask


def build_world_column(frame_positions, frame_rotmats, depth_axis, sag_value):
    local_column = np.stack((
        depth_axis,
        np.full_like(depth_axis, sag_value, dtype=np.float32),
        np.zeros_like(depth_axis, dtype=np.float32),
    ), axis=1).astype(np.float32)
    world = np.einsum("dc,fkc->fdk", local_column, frame_rotmats) + frame_positions[:, None, :]
    return world.astype(np.float32, copy=False)


def voxelize_recons_intermediate(
    data_recal,
    recons_knn,
    frame_records,
    depth_axis,
    sag_axis,
    point_min,
    point_max,
    spacing_xyz,
    observed_mask,
    plane_mask_data,
):
    grid_shape_zyx = build_grid_shape_zyx(point_min, point_max, spacing_xyz)
    sum_volume = np.zeros(grid_shape_zyx, dtype=np.float32)
    count_volume = np.zeros(grid_shape_zyx, dtype=np.uint16)

    frame_positions = np.stack([rec.position for rec in frame_records], axis=0).astype(np.float32, copy=False)
    frame_rotmats = np.stack([rec.rotmat for rec in frame_records], axis=0).astype(np.float32, copy=False)

    n_frames, n_sag, _ = data_recal.shape
    if n_frames != len(frame_records):
        raise ValueError(
            f"Frame-count mismatch between data_recal ({n_frames}) and infos.json ({len(frame_records)})."
        )

    base_depth_cols = recons_knn["base_depth_cols"]
    base_frame_idxs = recons_knn["base_frame_idxs"]
    base_dists = recons_knn["base_dists"]
    num_samples = base_frame_idxs.shape[1]
    num_voxels = base_frame_idxs.shape[0]

    column_cache = {}
    recons_added = 0
    contributing_sag_slices = 0

    for sag_idx in tqdm(range(2, n_sag - 2), desc="Stage 2 recons interpolation"):
        for needed_idx in (sag_idx - 1, sag_idx, sag_idx + 1):
            if needed_idx not in column_cache:
                column_cache[needed_idx] = build_world_column(
                    frame_positions,
                    frame_rotmats,
                    depth_axis,
                    float(sag_axis[needed_idx]),
                )
        for stale_idx in list(column_cache.keys()):
            if stale_idx not in {sag_idx - 1, sag_idx, sag_idx + 1}:
                del column_cache[stale_idx]

        plane_j = data_recal[:, sag_idx, :]
        plane_m1 = data_recal[:, sag_idx - 1, :]
        plane_p1 = data_recal[:, sag_idx + 1, :]
        world_j = column_cache[sag_idx]
        world_m1 = column_cache[sag_idx - 1]
        world_p1 = column_cache[sag_idx + 1]

        accum_intensity = np.zeros((num_voxels,), dtype=np.float64)
        accum_world = np.zeros((num_voxels, 3), dtype=np.float64)
        sum_w = np.zeros((num_voxels,), dtype=np.float64)

        for sample_idx in range(num_samples):
            frame_idx = base_frame_idxs[:, sample_idx]
            depth_idx = base_depth_cols[:, sample_idx]
            dist = base_dists[:, sample_idx]
            valid = (depth_idx >= 0) & np.isfinite(dist)
            if not np.any(valid):
                continue

            safe_dist = np.where(dist < 1e-8, 1e-8, dist)
            w0 = np.zeros_like(dist, dtype=np.float64)
            w0[valid] = 1.0 / safe_dist[valid]

            accum_intensity[valid] += w0[valid] * plane_j[frame_idx[valid], depth_idx[valid]]
            accum_world[valid] += w0[valid, None] * world_j[frame_idx[valid], depth_idx[valid]]
            sum_w[valid] += w0[valid]

            w1 = np.zeros_like(dist, dtype=np.float64)
            w1[valid] = 1.0 / np.sqrt(safe_dist[valid] ** 2 + 1.0)
            accum_intensity[valid] += w1[valid] * (
                plane_m1[frame_idx[valid], depth_idx[valid]] +
                plane_p1[frame_idx[valid], depth_idx[valid]]
            )
            accum_world[valid] += w1[valid, None] * (
                world_m1[frame_idx[valid], depth_idx[valid]] +
                world_p1[frame_idx[valid], depth_idx[valid]]
            )
            sum_w[valid] += 2.0 * w1[valid]

        valid_recons = sum_w > 0
        if not np.any(valid_recons):
            continue

        contributing_sag_slices += 1
        intensities = (accum_intensity[valid_recons] / sum_w[valid_recons]).astype(np.float32)
        world_points = (accum_world[valid_recons] / sum_w[valid_recons, None]).astype(np.float32)

        if plane_mask_data is not None:
            inside_plane = compute_sequence_plane_mask(world_points, plane_mask_data)
            if not np.any(inside_plane):
                continue
            intensities = intensities[inside_plane]
            world_points = world_points[inside_plane]

        voxel_coords = np.floor((world_points - point_min[None, :]) / spacing_xyz[None, :]).astype(np.int64)
        inside_grid = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_shape_zyx[2]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_shape_zyx[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_shape_zyx[0])
        )
        if not np.any(inside_grid):
            continue

        voxel_coords = voxel_coords[inside_grid]
        intensities = intensities[inside_grid]

        vx = voxel_coords[:, 0]
        vy = voxel_coords[:, 1]
        vz = voxel_coords[:, 2]
        blank_mask = ~observed_mask[vz, vy, vx]
        if not np.any(blank_mask):
            continue

        vx = vx[blank_mask]
        vy = vy[blank_mask]
        vz = vz[blank_mask]
        intensities = intensities[blank_mask]
        np.add.at(sum_volume, (vz, vy, vx), intensities)
        np.add.at(count_volume, (vz, vy, vx), 1)
        recons_added += int(np.count_nonzero(blank_mask))

    stage2_mask = count_volume > 0
    return sum_volume, count_volume, stage2_mask, recons_added, contributing_sag_slices


def fill_remaining_holes(volume_zyx, filled_mask, support_mask, x_axis, y_axis, hole_fill_k, chunk_size):
    if hole_fill_k <= 0:
        raise ValueError("--hole-fill-k must be >= 1")

    hole_filled_voxels = 0
    for z_idx in tqdm(range(volume_zyx.shape[0]), desc="Stage 3 hole filling"):
        support_2d = support_mask[z_idx]
        filled_2d = filled_mask[z_idx]
        missing_2d = support_2d & (~filled_2d)
        if not np.any(missing_2d):
            continue

        source_2d = filled_2d & support_2d
        if not np.any(source_2d):
            continue

        src_rc = np.argwhere(source_2d)
        query_rc = np.argwhere(missing_2d)
        if src_rc.shape[0] == 0 or query_rc.shape[0] == 0:
            continue

        src_coords = np.column_stack((x_axis[src_rc[:, 1]], y_axis[src_rc[:, 0]])).astype(np.float32, copy=False)
        src_vals = volume_zyx[z_idx][source_2d].astype(np.float32, copy=False)
        tree = cKDTree(src_coords)
        k = min(int(hole_fill_k), int(src_coords.shape[0]))

        filled_vals = np.zeros((query_rc.shape[0],), dtype=np.float32)
        for start in range(0, query_rc.shape[0], chunk_size):
            stop = min(start + chunk_size, query_rc.shape[0])
            query_chunk = query_rc[start:stop]
            query_coords = np.column_stack((x_axis[query_chunk[:, 1]], y_axis[query_chunk[:, 0]])).astype(np.float32, copy=False)
            dists, idxs = tree.query(query_coords, k=k, workers=-1)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            safe_d = np.where(dists < 1e-8, 1e-8, dists)
            weights = 1.0 / safe_d
            filled_vals[start:stop] = (weights * src_vals[idxs]).sum(axis=1) / weights.sum(axis=1)

        volume_zyx[z_idx][missing_2d] = filled_vals
        filled_mask[z_idx][missing_2d] = True
        hole_filled_voxels += int(query_rc.shape[0])

    return int(hole_filled_voxels)


def main():
    args = parse_args()
    if args.hole_fill_k <= 0:
        raise ValueError("--hole-fill-k must be >= 1")

    output_dir = make_dated_output_dir(args.output, args.ckpt)
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint/data load device: {LOAD_DEVICE}")

    baked_dataset, dataset_path, ckpt = load_dataset_from_ckpt(args.ckpt)
    print(f"Dataset loaded from: {dataset_path}")

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
            print("Sequence plane mask requested, but no valid front/back plane metadata was found. Falling back to the saved recons footprint only.")
        else:
            print("Using sequence plane mask:")
            print(f"  Source: {plane_mask_data.get('source', 'unknown')}")
            print(f"  Front plane point: {plane_mask_data['front_point']}")
            print(f"  Back plane point: {plane_mask_data['back_point']}")

    recons_files = autodetect_recons_files(dataset_path, args)
    print(f"Using saved recons ref: {recons_files.ref}")
    print(f"  data_recal: {recons_files.data_recal_mat}")
    print(f"  points:     {recons_files.recons_points_mat}")
    print(f"  sagplan:    {recons_files.recons_sagplan_mat}")

    recons_inputs = load_saved_recons_inputs(recons_files)
    data_recal = recons_inputs["data_recal"]
    frame_records = load_frame_records_from_infos(
        Path(baked_dataset.infos_json_path),
        baked_dataset,
        expected_frames=data_recal.shape[0],
    )
    local_points, depth_axis, sag_axis = build_local_pixel_geometry(baked_dataset)

    delta_x_cc = float(args.delta_x_cc) if args.delta_x_cc is not None else float(baked_dataset.roi_px_size_width_mm)
    delta_x_seqdyn = float(args.delta_x_seqdyn) if args.delta_x_seqdyn is not None else float(baked_dataset.roi_px_size_width_mm)
    coord_pts_img_ref_tr, coord_pts_img_seqdyn_tr = transform_coord(
        recons_inputs["coord_pts_img_ref"],
        recons_inputs["coord_pts_img_seqdyn"],
        delta_x_cc,
        delta_x_seqdyn,
        recons_inputs["idx_sag"],
        recons_inputs["P"],
        recons_inputs["tr_coord"],
    )
    X_transform = determine_transform_v2(coord_pts_img_ref_tr, coord_pts_img_seqdyn_tr, data_recal)
    xp, zp, offset_x, offset_z = data_transform_v2(data_recal, X_transform)
    mask_2d = build_recons_mask_2d(xp, zp)
    coord_mask_xz, idx_mask = coord_to_fill_fortran(mask_2d)

    print(f"data_recal shape (frame, sagittal, depth): {tuple(data_recal.shape)}")
    print(f"Saved idx_sag: {int(recons_inputs['idx_sag'])}")
    print(f"Recomputed X_transform: {X_transform.tolist()}")
    print(f"Recons common grid shape (x, z): {mask_2d.shape}")
    print(f"Recons support voxels per sagittal plane: {int(coord_mask_xz.shape[0]):,}")

    print("Stage 1/3: voxelizing known observed points onto the export_full_grid lattice...")
    volume_zyx, observed_mask = voxelize_observed_recal(
        data_recal,
        frame_records,
        local_points,
        point_min,
        point_max,
        spacing_xyz,
    )
    observed_voxels = int(np.count_nonzero(observed_mask))
    print(f"Observed voxels written directly to grid: {observed_voxels:,}")

    print("Stage 2/3: running saved recons3D interpolation and voxelizing intermediate slices...")
    recons_knn = precompute_recons_neighbours(coord_mask_xz, xp, zp, nb_pts=3)
    stage2_sum, stage2_count, stage2_mask, recons_added, contributing_sag_slices = voxelize_recons_intermediate(
        data_recal,
        recons_knn,
        frame_records,
        depth_axis,
        sag_axis,
        point_min,
        point_max,
        spacing_xyz,
        observed_mask,
        plane_mask_data,
    )
    stage2_voxels = int(np.count_nonzero(stage2_mask))
    fill_from_stage2 = (~observed_mask) & stage2_mask
    if np.any(fill_from_stage2):
        volume_zyx[fill_from_stage2] = stage2_sum[fill_from_stage2] / stage2_count[fill_from_stage2].astype(np.float32)
    filled_mask = observed_mask | fill_from_stage2
    support_mask = observed_mask | stage2_mask
    print(f"Recons intermediate samples voxelized: {recons_added:,}")
    print(f"Full-grid voxels filled by stage 2: {int(np.count_nonzero(fill_from_stage2)):,}")
    print(f"Sagittal slices contributing in stage 2: {contributing_sag_slices:,} / {data_recal.shape[1]:,}")

    hole_filled_voxels = 0
    if args.disable_hole_fill:
        print("Stage 3/3: hole filling disabled.")
    else:
        print("Stage 3/3: filling remaining blank full-grid voxels plane-wise...")
        hole_filled_voxels = fill_remaining_holes(
            volume_zyx,
            filled_mask,
            support_mask,
            x_axis,
            y_axis,
            args.hole_fill_k,
            args.chunk_size,
        )
        print(f"Residual full-grid voxels filled in stage 3: {hole_filled_voxels:,}")

    filled = int(np.count_nonzero(filled_mask))
    supported = int(np.count_nonzero(support_mask))
    total_voxels = int(volume_zyx.size)
    print(f"Supported voxels: {supported:,} / {total_voxels:,}")
    print(f"Filled voxels after all stages: {filled:,} / {total_voxels:,}")
    print(f"Intensity range: [{float(volume_zyx.min()):.2f}, {float(volume_zyx.max()):.2f}]")

    print("Saving MHD volume...")
    save_mhd(volume_zyx, output_dir, spacing_xyz)

    metadata = {
        "method": "saved_recons3d_then_voxelize_to_full_grid",
        "ckpt": str(args.ckpt),
        "dataset_pkl": str(dataset_path),
        "bounds_dataset_pkl": str(args.bounds_dataset) if args.bounds_dataset else str(dataset_path),
        "recons_ref": recons_files.ref,
        "data_recal_mat": str(recons_files.data_recal_mat),
        "recons_points_mat": str(recons_files.recons_points_mat),
        "recons_sagplan_mat": str(recons_files.recons_sagplan_mat),
        "grid_shape_zyx": list(volume_zyx.shape),
        "spacing_mm_xyz": spacing_xyz.tolist(),
        "base_spacing_mm_xyz": base_spacing_xyz.tolist(),
        "resolution_scale": float(args.resolution_scale),
        "point_min_mm": point_min.tolist(),
        "point_max_mm": point_max.tolist(),
        "data_recal_shape": list(data_recal.shape),
        "delta_x_cc_mm": float(delta_x_cc),
        "delta_x_seqdyn_mm": float(delta_x_seqdyn),
        "saved_idx_sag": int(recons_inputs["idx_sag"]),
        "x_transform": X_transform.tolist(),
        "xp_min_max": [float(np.min(xp)), float(np.max(xp))],
        "zp_min_max": [float(np.min(zp)), float(np.max(zp))],
        "recons_mask_shape_xz": list(mask_2d.shape),
        "recons_mask_voxels_per_plane": int(coord_mask_xz.shape[0]),
        "recons_offset_x": float(offset_x),
        "recons_offset_z": float(offset_z),
        "hole_fill_k": int(args.hole_fill_k),
        "hole_fill_enabled": bool(not args.disable_hole_fill),
        "use_sequence_plane_mask": bool(plane_mask_data is not None),
        "sequence_plane_mask_requested": bool(use_sequence_plane_mask),
        "sequence_plane_mask_source": None if plane_mask_data is None else plane_mask_data.get("source"),
        "observed_voxels": int(observed_voxels),
        "stage2_grid_voxels": int(stage2_voxels),
        "stage2_added_samples": int(recons_added),
        "stage2_filled_voxels": int(np.count_nonzero(fill_from_stage2)),
        "stage2_contributing_sagittal_slices": int(contributing_sag_slices),
        "hole_filled_voxels": int(hole_filled_voxels),
        "supported_voxels": int(supported),
        "filled_voxels": int(filled),
        "total_voxels": int(total_voxels),
        "mhd_grid_axis_order": ["h", "z", "w"],
        "mhd_grid_axis_mapping": {"h": "-x", "z": "-z", "w": "-y"},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Saved to: {output_dir}")
    print(f"  volume.raw / volume.mhd  — shape (h, z, w): {np.flip(volume_zyx, axis=2).shape}")


if __name__ == "__main__":
    main()
