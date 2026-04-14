from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import time

import dataset
import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.optimize import least_squares
from scipy.spatial import KDTree
import torch
import tqdm

from utils import get_base_points

# python /home/zchen/Code/NeUF/recons3d_exact_from_saved.py \
#   --recons-points-mat /home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/Reconstruction_3D/Patient0/data_3D_Patient0_J35_2_reconstruction_points.mat \
#   --recons-sagplan-mat /home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/Reconstruction_3D/Patient0/data_3D_Patient0_J35_2_sagplan_params.mat \
#   --data-recal-mat /home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/data_recal_Patient0_J35_2_d_0.5.mat \
#   --delta-x-cc 1 \
#   --delta-x-seqdyn 1 \
#   --output /home/zchen/Code/NeUF/exports/recons3d_exact_from_saved \
#   --save-debug-npy

LOAD_DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class FrameRecord:
    position: np.ndarray
    rotmat: np.ndarray
    frame_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact recons3D coordinate transform and 2.5D interpolation "
            "from saved reconstruction points / sagplan params / data_recal inputs."
        )
    )
    parser.add_argument("--recons-points-mat", type=Path, required=True, help="Path to data_3D_*_reconstruction_points.mat")
    parser.add_argument("--recons-sagplan-mat", type=Path, required=True, help="Path to data_3D_*_sagplan_params.mat")
    parser.add_argument("--data-recal-mat", type=Path, required=True, help="Path to data_recal_*.mat")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .mat path, or a base output directory for dated run folders.",
    )
    parser.add_argument("--delta-x-cc", type=float, required=True, help="Sagittal reference spacing in mm/px")
    parser.add_argument("--delta-x-seqdyn", type=float, required=True, help="Dynamic sequence spacing in mm/px")
    parser.add_argument(
        "--dataset-pkl",
        type=Path,
        default=None,
        help="Optional baked dataset .pkl used to recover world-space grid geometry for the exported .h5 bundle.",
    )
    parser.add_argument(
        "--save-debug-npy",
        action="store_true",
        help="Also save xp.npy, zp.npy, and X_transform.npy alongside the main outputs.",
    )
    return parser.parse_args()


def load_h5_or_mat_array(path: Path, key: str) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"{path} does not contain dataset '{key}'.")
            return np.array(f[key])
    except OSError:
        data = sio.loadmat(str(path))
        if key not in data:
            raise KeyError(f"{path} does not contain variable '{key}'.")
        return np.array(data[key])


def load_recons_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(str(path))
    if "coord_pts_img_ref" not in data or "coord_pts_img_seqdyn" not in data:
        raise KeyError(f"{path} must contain coord_pts_img_ref and coord_pts_img_seqdyn.")
    coord_pts_img_ref = np.asarray(data["coord_pts_img_ref"], dtype=np.float64)
    coord_pts_img_seqdyn = np.asarray(data["coord_pts_img_seqdyn"], dtype=np.float64)
    return coord_pts_img_ref, coord_pts_img_seqdyn


def infer_ref(points_path: Path) -> str:
    name = points_path.name
    prefix = "data_3D_"
    suffix = "_reconstruction_points.mat"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return points_path.stem


def make_dated_output_dir(base_output_dir: Path, run_name: str) -> Path:
    date_str = date.today().strftime("%d-%m-%Y")
    dated_root = base_output_dir / date_str
    dated_root.mkdir(parents=True, exist_ok=True)
    safe_run_name = Path(run_name).name or "exact_recons3d"

    num = 0
    while True:
        candidate = dated_root / f"{safe_run_name}_{num}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        num += 1


def resolve_output_layout(output_arg: Path, ref: str) -> dict[str, Path | str]:
    if output_arg.suffix.lower() == ".mat":
        output_dir = output_arg.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_arg.stem
        return {
            "mode": "explicit_mat",
            "output_dir": output_dir,
            "mat_path": output_arg,
            "mhd_path": output_arg.with_suffix(".mhd"),
            "h5_path": output_arg.with_suffix(".h5"),
            "metadata_path": output_arg.with_name(f"{stem}_metadata.json"),
            "debug_xp_path": output_arg.with_name(f"{stem}_xp.npy"),
            "debug_zp_path": output_arg.with_name(f"{stem}_zp.npy"),
            "debug_transform_path": output_arg.with_name(f"{stem}_X_transform.npy"),
            "mhd_base_name": stem,
        }

    output_dir = make_dated_output_dir(output_arg, ref)
    return {
        "mode": "dated_run_dir",
        "output_dir": output_dir,
        "mat_path": output_dir / "exact_recons3d.mat",
        "mhd_path": output_dir / "volume.mhd",
        "h5_path": output_dir / "recons_common_grid.h5",
        "metadata_path": output_dir / "metadata.json",
        "debug_xp_path": output_dir / "xp.npy",
        "debug_zp_path": output_dir / "zp.npy",
        "debug_transform_path": output_dir / "X_transform.npy",
        "mhd_base_name": "volume",
    }


def save_mhd_array(volume_for_raw, output_dir: Path, base_name: str, dim_sizes, spacing_sizes, element_type="MET_FLOAT"):
    raw_path = output_dir / f"{base_name}.raw"
    mhd_path = output_dir / f"{base_name}.mhd"

    if element_type == "MET_FLOAT":
        array_to_write = volume_for_raw.astype(np.float32, copy=False)
    else:
        array_to_write = volume_for_raw

    with raw_path.open("wb") as f:
        if array_to_write.flags.c_contiguous:
            array_to_write.tofile(f)
        else:
            for slice_idx in range(array_to_write.shape[0]):
                np.ascontiguousarray(array_to_write[slice_idx]).tofile(f)

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


def save_recons_common_grid_mhd(volume_x_sag_z, output_dir: Path, base_name: str):
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


def resolve_dataset_pkl(data_recal_mat: Path, dataset_pkl: Path | None) -> Path:
    candidates = []
    if dataset_pkl is not None:
        candidates.append(dataset_pkl)

    recal_dir = data_recal_mat.parent
    candidates.extend(
        [
            recal_dir / "us_recal_original" / "baked_dataset.pkl",
            recal_dir / "dataset.pkl",
            recal_dir / "us_recal_original" / "dataset.pkl",
        ]
    )

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate a dataset .pkl required for world-space recons grid export. "
        "Pass --dataset-pkl explicitly."
    )


def load_baked_dataset(dataset_pkl: Path):
    saved = torch.load(dataset_pkl, map_location=LOAD_DEVICE, weights_only=False)
    if "dataset" not in saved:
        raise KeyError(f"'dataset' key not found in {dataset_pkl}")
    return saved["dataset"]


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
    return depth_axis, sag_axis


def save_recons_common_grid_h5(
    h5_path: Path,
    data_3d_mask_uint8: np.ndarray,
    compact_bundle: dict,
    frame_records: list[FrameRecord],
    depth_axis_mm: np.ndarray,
    sag_axis_mm: np.ndarray,
    metadata: dict,
):
    frame_positions_mm = np.stack([rec.position for rec in frame_records], axis=0).astype(np.float32, copy=False)
    frame_rotmats = np.stack([rec.rotmat for rec in frame_records], axis=0).astype(np.float32, copy=False)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data_3d_mask_uint8", data=data_3d_mask_uint8.astype(np.uint8, copy=False))
        f.create_dataset("coord_mask_xz_zero_based", data=compact_bundle["coord_mask_xz_zero_based"].astype(np.int32, copy=False))
        f.create_dataset("active_sag_indices_zero_based", data=compact_bundle["active_sag_indices_zero_based"].astype(np.int32, copy=False))
        f.create_dataset("frame_positions_mm", data=frame_positions_mm)
        f.create_dataset("frame_rotmats", data=frame_rotmats)
        f.create_dataset("depth_axis_mm", data=depth_axis_mm.astype(np.float32, copy=False))
        f.create_dataset("sag_axis_mm", data=sag_axis_mm.astype(np.float32, copy=False))
        f.create_dataset("I_complet_1", data=compact_bundle["I_complet_1"].astype(np.int32, copy=False))
        f.create_dataset("d_complet_1", data=compact_bundle["d_complet_1"].astype(np.float32, copy=False))
        f.create_dataset("d_complet_1_1", data=compact_bundle["d_complet_1_1"].astype(np.float32, copy=False))
        f.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=True)


class _NullProgressDialog:
    def setWindowTitle(self, *_args, **_kwargs):
        return None

    def setWindowModality(self, *_args, **_kwargs):
        return None

    def setCancelButton(self, *_args, **_kwargs):
        return None

    def setMinimumDuration(self, *_args, **_kwargs):
        return None

    def show(self):
        return None

    def setValue(self, *_args, **_kwargs):
        return None

    def setLabelText(self, *_args, **_kwargs):
        return None


def _process_events():
    return None


def transform_coord(coord_pts_img_sag, coord_pts_img_seqdyn, delta_X_img_sag, delta_X_seqdyn, idx_sag, P, tr_coord):
    scale_factor = delta_X_img_sag / delta_X_seqdyn
    coord_pts_img_sag_scaled = coord_pts_img_sag * scale_factor
    coord_pts_img_sag_tr = np.column_stack((coord_pts_img_sag_scaled[:, 1], coord_pts_img_sag_scaled[:, 0]))

    pts_seq_yx = np.column_stack((coord_pts_img_seqdyn[:, 1], coord_pts_img_seqdyn[:, 0]))
    num_pts = len(pts_seq_yx)
    ones_col = np.full((num_pts, 1), idx_sag)
    pts_3d = np.vstack((pts_seq_yx[:, 0], ones_col.flatten(), pts_seq_yx[:, 1]))
    tr_coord = tr_coord.reshape(3, 1)
    pts_3d_transformed = P @ (pts_3d + tr_coord - 1)
    coord_pts_img_seqdyn_tr = pts_3d_transformed[[0, 2], :].T

    return coord_pts_img_sag_tr, coord_pts_img_seqdyn_tr


def determine_transform_v2(coord_pts_img_ref, coord_pts_img_seqdyn, data_recal, debug_mode, idx, main_dir):
    sz3 = data_recal.shape[2]
    b_inf = np.array([-30, 200, -0.08, -2, -80, (-np.pi / 2) / sz3, -np.pi / 3], dtype=np.float64)
    b_sup = np.array([80, 800, 0.08, 2, 30, (np.pi / 2) / sz3, np.pi / 3], dtype=np.float64)
    x0 = np.array([0, 350, 0, 0, 0, 0, 0], dtype=np.float64)
    print(f"Initial x0: {x0}")

    r = coord_pts_img_seqdyn[:, 0].astype(np.float64)
    t = coord_pts_img_seqdyn[:, 1].astype(np.float64)
    ref_x = coord_pts_img_ref[:, 0].astype(np.float64)
    ref_y = coord_pts_img_ref[:, 1].astype(np.float64)

    def fun(x):
        res_x = x[0] + x[2] * t + (r + x[4]) * np.cos(t * x[5] + x[6]) - ref_x
        res_y = x[1] + x[3] * t + (r + x[4]) * np.sin(t * x[5] + x[6]) - ref_y
        res = np.empty(res_x.size * 2, dtype=np.float64)
        res[0::2] = res_x
        res[1::2] = res_y
        return res

    diff_step = np.full_like(x0, 1e-6, dtype=np.float64)

    res = least_squares(
        fun,
        x0,
        bounds=(b_inf, b_sup),
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=2000,
        diff_step=diff_step,
        loss="linear",
    )

    X_transform = res.x
    print(f"Found parameters: {X_transform}")

    if debug_mode:
        main_path = Path(main_dir)
        ref_file_path = main_path / "Ref_Files" / "ReconstructionParameters_v2.mat"
        if ref_file_path.exists():
            mat_data = sio.loadmat(str(ref_file_path))
            params_patients = mat_data["params_patients_echo_v2"]
        else:
            raise FileNotFoundError()

        for i in range(7):
            params_patients[idx, i] = X_transform[i]

        sio.savemat(
            str(main_path / "Ref_Files" / "ReconstructionParameters_v2.mat"),
            {"params_patients_echo_v2": params_patients},
        )
        print("Parameters saved in .mat file.")

    return X_transform


def get_data_int_mask(data_recal, pts_aug, xp, zp, closing_size=3):
    _process_events()
    x_max = int(np.ceil(xp.max())) + 1
    z_max = int(np.ceil(zp.max())) + 1
    ny = data_recal.shape[1]

    mask = np.zeros((x_max, ny, z_max), dtype=bool)

    x_orig = np.round(xp).astype(int)
    z_orig = np.round(zp).astype(int)

    valid = (x_orig >= 0) & (x_orig < x_max) & (z_orig >= 0) & (z_orig < z_max)
    mask[x_orig[valid], :, z_orig[valid]] = True

    def interpolate_line(coords_start, coords_end, n_interp):
        t = np.linspace(0, 1, n_interp + 1)[:-1]
        t = t[:, None]

        start = coords_start[None, :]
        end = coords_end[None, :]

        interp = (1 - t) * start + t * end
        return interp.ravel()

    xp_max_start = xp[-1, :-1]
    xp_max_end = xp[-1, 1:]
    zp_max_start = zp[-1, :-1]
    zp_max_end = zp[-1, 1:]

    xp_max_interp = interpolate_line(xp_max_start, xp_max_end, pts_aug)
    zp_max_interp = interpolate_line(zp_max_start, zp_max_end, pts_aug)

    xp_min_start = xp[0, :-1]
    xp_min_end = xp[0, 1:]
    zp_min_start = zp[0, :-1]
    zp_min_end = zp[0, 1:]

    xp_min_interp = interpolate_line(xp_min_start, xp_min_end, pts_aug)
    zp_min_interp = interpolate_line(zp_min_start, zp_min_end, pts_aug)

    for x_interp, z_interp in [(xp_max_interp, zp_max_interp), (xp_min_interp, zp_min_interp)]:
        x_r = np.round(x_interp).astype(int)
        z_r = np.round(z_interp).astype(int)
        valid = (x_r >= 0) & (x_r < x_max) & (z_r >= 0) & (z_r < z_max)
        mask[x_r[valid], :, z_r[valid]] = True
        _process_events()

    structure = np.ones((closing_size, closing_size), dtype=bool)

    for iy in tqdm.tqdm(range(ny), desc="Mask closing", dynamic_ncols=True):
        slice_2d = mask[:, iy, :]
        filled = binary_fill_holes(slice_2d)
        closed = binary_closing(filled, structure=structure)
        mask[:, iy, :] = closed
        _process_events()

    return mask.astype(np.uint8)


def data_transform_v2(data_recal, X_transform):
    Cx, Ct, vx, vt, delta, omega, theta = X_transform

    # data_recal is indexed as (frame, sagittal, depth).
    # r follows image depth, while t follows the moving frame index.
    n_rows = data_recal.shape[2]
    n_cols = data_recal.shape[0]

    r = np.arange(1, n_rows + 1).reshape(-1, 1)
    t = np.arange(1, n_cols + 1).reshape(1, -1)

    x_moving = Cx + vx * t + (r + delta) * np.cos(t * omega + theta)
    y_moving = Ct + vt * t + (r + delta) * np.sin(t * omega + theta)

    offset_x = np.abs(np.minimum(np.min(x_moving), 0)) + 1
    offset_y = np.abs(np.minimum(np.min(y_moving), 0)) + 1

    x_moving += offset_x
    y_moving += offset_y

    return x_moving, y_moving, offset_x, offset_y


def coord_after_transform(data_recal, xp, zp):
    Nx, Nz = data_recal.shape
    if xp.shape != (Nx, Nz) or zp.shape != (Nx, Nz):
        raise ValueError(
            f"xp/zp shape mismatch with reference slice: "
            f"data={data_recal.shape}, xp={xp.shape}, zp={zp.shape}"
        )

    xp = xp.reshape(Nx, 1, Nz)
    zp = zp.reshape(Nx, 1, Nz)

    X = np.zeros((Nx, 1, Nz))
    Y = np.zeros((Nx, 1, Nz))
    Z = np.zeros((Nx, 1, Nz))

    X[:, 0, :] = xp[:, 0, :]
    Y[:, 0, :] = 1
    Z[:, 0, :] = zp[:, 0, :]

    # For xp/zp shaped as (depth, frame), Fortran order keeps one frame curve
    # as a contiguous block of depth samples.
    coord_transform = np.column_stack((
        X.ravel(order="F"),
        Y.ravel(order="F"),
        Z.ravel(order="F"),
    ))

    return coord_transform


def coord_to_fill(data_int_mask):
    idx_mask = np.flatnonzero(data_int_mask.ravel() > 0)

    X, Z = data_int_mask.shape

    x_coords = np.repeat(np.arange(1, X + 1)[:, None], Z, axis=1)
    z_coords = np.repeat(np.arange(1, Z + 1)[None, :], X, axis=0)
    y_coords = np.ones_like(x_coords)

    x_flat = x_coords.ravel()[idx_mask]
    y_flat = y_coords.ravel()[idx_mask]
    z_flat = z_coords.ravel()[idx_mask]

    coord_mask = np.column_stack((x_flat, y_flat, z_flat))

    return coord_mask, idx_mask


def data_interpolation_v2_25D_3(data_recal, nb_pts, multi_img, xp, zp):
    _process_events()
    wb = _NullProgressDialog()
    wb.setValue(5)
    _process_events()

    print("Calcul du masque d'interpolation (1/3)", flush=True)
    wb.setLabelText("Calcul du masque d'interpolation (1/3)")
    wb.setValue(20)
    _process_events()
    data_int_mask_full = get_data_int_mask(data_recal, pts_aug=9, xp=xp, zp=zp)
    _process_events()
    data_3D_mask = data_int_mask_full.astype(np.uint8)

    data_int_mask = data_int_mask_full[:, 0, :]

    coord_transform = coord_after_transform(data_recal[:, 0, :].T, xp, zp)
    coord_transform = coord_transform[:, [0, 2]]

    coord_mask, idx_mask = coord_to_fill(data_int_mask)
    coord_mask = coord_mask[:, [0, 2]]
    coord_mask_xz_zero_based = coord_mask.astype(np.int32, copy=False) - 1

    n_voxels = len(coord_mask)
    n_height = data_recal.shape[2]
    n_long = data_recal.shape[0]

    Idx = np.zeros((n_voxels, n_long, nb_pts), dtype=int)
    Dist = np.zeros((n_voxels, n_long, nb_pts), dtype=float)
    _process_events()
    print("Recherche des plus proches voisins (2/3)", flush=True)
    wb.setLabelText("Recherche des plus proches voisins (2/3)")
    wb.setValue(20)
    _process_events()
    for i in tqdm.tqdm(range(n_long), desc="KNN curves", dynamic_ncols=True):
        start = i * n_height
        end = start + n_height
        points_i = coord_transform[start:end, :]
        tree = KDTree(points_i)
        dist_i, idx_i = tree.query(coord_mask, k=nb_pts)
        Idx[:, i, :] = idx_i
        Dist[:, i, :] = dist_i
        wb.setValue(20 + int(30 * (i + 1) / n_long))
        _process_events()

    num_img_d_min = np.argmin(Dist[:, :, 0], axis=1)

    if multi_img == 1:
        n_channels = 4
    else:
        n_channels = 2

    I_complet = np.zeros((n_voxels, nb_pts, n_channels), dtype=int)
    d_complet = np.full((n_voxels, nb_pts, n_channels), np.inf)

    for i in range(n_voxels):
        mid = num_img_d_min[i]

        if 0 <= mid < n_long:
            I_complet[i, :, 0] = mid * n_height + Idx[i, mid, :]
            d_complet[i, :, 0] = Dist[i, mid, :]

        bool_right = False
        if mid > 0 and mid < n_long - 1:
            bool_right = Dist[i, mid + 1, 0] < Dist[i, mid - 1, 0]

        if bool_right or mid == 0:
            if mid + 1 < n_long:
                I_complet[i, :, 1] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                d_complet[i, :, 1] = Dist[i, mid + 1, :]

            if multi_img == 1:
                if mid == 0:
                    if mid + 2 < n_long:
                        I_complet[i, :, 2] = (mid + 2) * n_height + Idx[i, mid + 2, :]
                        d_complet[i, :, 2] = Dist[i, mid + 2, :]
                elif mid == n_long - 2:
                    if mid - 1 >= 0:
                        I_complet[i, :, 2] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                        d_complet[i, :, 2] = Dist[i, mid - 1, :]
                else:
                    if mid - 1 >= 0:
                        I_complet[i, :, 2] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                        d_complet[i, :, 2] = Dist[i, mid - 1, :]
                    if mid + 2 < n_long:
                        I_complet[i, :, 3] = (mid + 2) * n_height + Idx[i, mid + 2, :]
                        d_complet[i, :, 3] = Dist[i, mid + 2, :]

        else:
            if mid - 1 >= 0:
                I_complet[i, :, 1] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                d_complet[i, :, 1] = Dist[i, mid - 1, :]

            if multi_img == 1:
                if mid == 1:
                    if mid + 1 < n_long:
                        I_complet[i, :, 2] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                        d_complet[i, :, 2] = Dist[i, mid + 1, :]
                elif mid == n_long - 1:
                    if mid - 2 >= 0:
                        I_complet[i, :, 2] = (mid - 2) * n_height + Idx[i, mid - 2, :]
                        d_complet[i, :, 2] = Dist[i, mid - 2, :]
                else:
                    if mid + 1 < n_long:
                        I_complet[i, :, 2] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                        d_complet[i, :, 2] = Dist[i, mid + 1, :]
                    if mid - 2 >= 0:
                        I_complet[i, :, 3] = (mid - 2) * n_height + Idx[i, mid - 2, :]
                        d_complet[i, :, 3] = Dist[i, mid - 2, :]
        _process_events()

    max_idx = n_height * n_long - 1
    I_complet = np.clip(I_complet, 0, max_idx)

    I_complet_1 = I_complet[:, :, 0:2].reshape(n_voxels, nb_pts * 2)
    d_complet_1 = d_complet[:, :, 0:2].reshape(n_voxels, nb_pts * 2)
    d_complet_1_1 = np.sqrt(d_complet_1 ** 2 + 1)

    if multi_img == 1:
        if nb_pts == 3:
            n_pts_2 = 1
        elif nb_pts == 5:
            n_pts_2 = 3
        else:
            raise ValueError("nb_pts must be 3 or 5 for multi_img=1")
        I_complet_2 = I_complet[:, 0:n_pts_2, 2:4].reshape(n_voxels, n_pts_2 * 2)
        d_complet_2 = d_complet[:, 0:n_pts_2, 2:4].reshape(n_voxels, n_pts_2 * 2)

    x_max = int(np.ceil(np.max(xp))) + 1
    z_max = int(np.ceil(np.max(zp))) + 1
    data_int = np.zeros((x_max, data_recal.shape[1], z_max), dtype=float)

    print("Reconstruction et interpolation du volume (3/3)", flush=True)
    wb.setLabelText("Reconstruction et interpolation du volume (3/3)")

    active_sag_indices_zero_based = np.arange(2, data_recal.shape[1] - 2, dtype=np.int32)
    prog = 50
    for j in tqdm.tqdm(active_sag_indices_zero_based, desc="Sagittal interpolation", dynamic_ncols=True):
        accum = np.zeros(n_voxels)

        data_recal_j = data_recal[:, j, :]

        if nb_pts in [3, 5]:
            data_recal_m1 = data_recal[:, j - 1, :]
            data_recal_p1 = data_recal[:, j + 1, :]

        if nb_pts == 5:
            data_recal_m2 = data_recal[:, j - 2, :]
            data_recal_p2 = data_recal[:, j + 2, :]
            d_complet_1_2 = np.sqrt(d_complet_1 ** 2 + 4)

            if multi_img == 1:
                d_complet_2_1 = np.sqrt(d_complet_2 ** 2 + 1)

        for k in range(I_complet_1.shape[1]):
            idx_k = I_complet_1[:, k]
            w = 1 / d_complet_1[:, k]
            vals = data_recal_j.ravel()[idx_k]
            accum += w * vals

            if nb_pts == 3:
                w1 = 1 / d_complet_1_1[:, k]
                vals_m1 = data_recal_m1.ravel()[idx_k]
                vals_p1 = data_recal_p1.ravel()[idx_k]
                accum += w1 * (vals_m1 + vals_p1)
            elif nb_pts == 5:
                w1 = 1 / d_complet_1_1[:, k]
                vals_m1 = data_recal_m1.ravel()[idx_k]
                vals_p1 = data_recal_p1.ravel()[idx_k]
                accum += w1 * (vals_m1 + vals_p1)

                w2 = 1 / d_complet_1_2[:, k]
                vals_m2 = data_recal_m2.ravel()[idx_k]
                vals_p2 = data_recal_p2.ravel()[idx_k]
                accum += w2 * (vals_m2 + vals_p2)

        if multi_img == 1:
            for k in range(I_complet_2.shape[1]):
                idx_k = I_complet_2[:, k]
                w = 1 / d_complet_2[:, k]
                vals = data_recal_j.ravel()[idx_k]
                accum += w * vals

                if nb_pts == 5:
                    w1 = 1 / d_complet_2_1[:, k]
                    vals_m1 = data_recal_m1.ravel()[idx_k]
                    vals_p1 = data_recal_p1.ravel()[idx_k]
                    accum += w1 * (vals_m1 + vals_p1)

        sum_w = np.sum(1 / d_complet_1, axis=1)

        if nb_pts in [3, 5]:
            sum_w += 2 * np.sum(1 / d_complet_1_1, axis=1)

        if nb_pts == 5:
            sum_w += 2 * np.sum(1 / d_complet_1_2, axis=1)

        if multi_img == 1:
            sum_w += np.sum(1 / d_complet_2, axis=1)
            if nb_pts == 5:
                sum_w += 2 * np.sum(1 / d_complet_2_1, axis=1)

        sum_w[sum_w == 0] = 1
        accum /= sum_w

        data_int_temp = np.zeros((x_max, z_max))
        data_int_temp.ravel()[idx_mask] = accum

        data_int[:, j, :] = data_int_temp
        if j % 10 == 0:
            prog = 50 + int(45 * (j / data_recal.shape[1]))
            wb.setValue(min(prog, 98))
            _process_events()

    data_int = data_int.astype(np.uint8)
    wb.setValue(min(prog, 100))
    _process_events()

    compact_bundle = {
        "coord_mask_xz_zero_based": coord_mask_xz_zero_based,
        "active_sag_indices_zero_based": active_sag_indices_zero_based,
        "I_complet_1": I_complet_1.astype(np.int32, copy=False),
        "d_complet_1": d_complet_1.astype(np.float32, copy=False),
        "d_complet_1_1": d_complet_1_1.astype(np.float32, copy=False),
    }

    return data_int, data_3D_mask, compact_bundle


def main():
    start_time = time.perf_counter()
    args = parse_args()
    ref = infer_ref(args.recons_points_mat)
    output_layout = resolve_output_layout(args.output, ref)
    output_dir = output_layout["output_dir"]
    mat_path = output_layout["mat_path"]
    mhd_path = output_layout["mhd_path"]
    grid_h5_path = output_layout["h5_path"]
    metadata_path = output_layout["metadata_path"]
    debug_xp_path = output_layout["debug_xp_path"]
    debug_zp_path = output_layout["debug_zp_path"]
    debug_transform_path = output_layout["debug_transform_path"]
    mhd_base_name = output_layout["mhd_base_name"]

    print(f"Output directory: {output_dir}")

    coord_pts_img_ref, coord_pts_img_seqdyn = load_recons_points(args.recons_points_mat)
    P = load_h5_or_mat_array(args.recons_sagplan_mat, "P").astype(np.float64)
    tr_coord = load_h5_or_mat_array(args.recons_sagplan_mat, "tr_coord").astype(np.float64).reshape(-1)
    idx_sag = int(np.asarray(load_h5_or_mat_array(args.recons_sagplan_mat, "idx_sag")).reshape(-1)[0])
    data_recal = load_h5_or_mat_array(args.data_recal_mat, "data_recal")
    dataset_pkl = resolve_dataset_pkl(args.data_recal_mat, args.dataset_pkl)
    baked_dataset = load_baked_dataset(dataset_pkl)
    infos_json_path = Path(getattr(baked_dataset, "infos_json_path", ""))
    if not infos_json_path.exists():
        raise FileNotFoundError(f"infos.json not found: {infos_json_path}")
    frame_records = load_frame_records_from_infos(infos_json_path, baked_dataset, expected_frames=data_recal.shape[0])
    depth_axis, sag_axis = build_local_pixel_geometry(baked_dataset)

    coord_pts_img_ref_tr, coord_pts_img_seqdyn_tr = transform_coord(
        coord_pts_img_ref,
        coord_pts_img_seqdyn,
        float(args.delta_x_cc),
        float(args.delta_x_seqdyn),
        idx_sag,
        P,
        tr_coord,
    )

    X_transform = determine_transform_v2(
        coord_pts_img_ref_tr,
        coord_pts_img_seqdyn_tr,
        data_recal,
        False,
        0,
        str(output_dir),
    )
    xp, zp, offset_x, offset_z = data_transform_v2(data_recal, X_transform)
    data_3D, data_3D_mask, compact_bundle = data_interpolation_v2_25D_3(
        data_recal.astype(np.float32),
        3,
        0,
        xp,
        zp,
    )
    reconstruction_core_seconds = float(time.perf_counter() - start_time)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_recons_common_grid_mhd(data_3D, output_dir, str(mhd_base_name))
    save_dict = {
        "data_3D": data_3D,
        "data_3D_mask": data_3D_mask,
        "xp": xp,
        "zp": zp,
        "X_transform": np.asarray(X_transform, dtype=np.float64),
        "coord_pts_img_ref": coord_pts_img_ref,
        "coord_pts_img_seqdyn": coord_pts_img_seqdyn,
        "coord_pts_img_ref_tr": coord_pts_img_ref_tr,
        "coord_pts_img_seqdyn_tr": coord_pts_img_seqdyn_tr,
        "idx_sag": np.array([[idx_sag]], dtype=np.int32),
        "P": P,
        "tr_coord": tr_coord.reshape(1, -1),
        "metadata": {
            "ref": ref,
            "delta_X_cc": float(args.delta_x_cc),
            "delta_X_seqdyn": float(args.delta_x_seqdyn),
            "offset_x": float(offset_x),
            "offset_z": float(offset_z),
            "dataset_pkl": str(dataset_pkl),
            "infos_json_path": str(infos_json_path),
            "output_dir": str(output_dir),
            "output_metadata_json": str(metadata_path),
            "output_mhd": str(mhd_path),
            "output_grid_h5": str(grid_h5_path),
            "reconstruction_core_seconds": reconstruction_core_seconds,
            "recons_points_mat": str(args.recons_points_mat),
            "recons_sagplan_mat": str(args.recons_sagplan_mat),
            "data_recal_mat": str(args.data_recal_mat),
            "logic_source": "recons3D.py exact core functions",
        },
    }
    sio.savemat(str(mat_path), save_dict, do_compression=True)

    h5_metadata = {
        "ref": ref,
        "output_dir": str(output_dir),
        "data_recal_shape": list(np.asarray(data_recal).shape),
        "data_3d_shape": list(np.asarray(data_3D).shape),
        "data_3d_mask_shape": list(np.asarray(data_3D_mask).shape),
        "xp_shape": list(np.asarray(xp).shape),
        "zp_shape": list(np.asarray(zp).shape),
        "dataset_pkl": str(dataset_pkl),
        "infos_json_path": str(infos_json_path),
        "reconstruction_core_seconds": reconstruction_core_seconds,
        "source_script": "recons3d_exact_from_saved.py",
    }
    save_recons_common_grid_h5(
        grid_h5_path,
        data_3D_mask,
        compact_bundle,
        frame_records,
        depth_axis,
        sag_axis,
        h5_metadata,
    )

    if args.save_debug_npy:
        np.save(debug_xp_path, xp)
        np.save(debug_zp_path, zp)
        np.save(debug_transform_path, np.asarray(X_transform, dtype=np.float64))
    total_wall_seconds = float(time.perf_counter() - start_time)
    metadata_json = {
        "ref": ref,
        "output_layout_mode": output_layout["mode"],
        "output_dir": str(output_dir),
        "output_mat": str(mat_path),
        "output_metadata_json": str(metadata_path),
        "output_mhd": str(mhd_path),
        "output_grid_h5": str(grid_h5_path),
        "save_debug_npy": bool(args.save_debug_npy),
        "data_recal_shape": list(np.asarray(data_recal).shape),
        "xp_shape": list(xp.shape),
        "zp_shape": list(zp.shape),
        "data_3D_shape": list(data_3D.shape),
        "data_3D_mask_shape": list(data_3D_mask.shape),
        "idx_sag": idx_sag,
        "delta_X_cc": float(args.delta_x_cc),
        "delta_X_seqdyn": float(args.delta_x_seqdyn),
        "offset_x": float(offset_x),
        "offset_z": float(offset_z),
        "dataset_pkl": str(dataset_pkl),
        "infos_json_path": str(infos_json_path),
        "recons_points_mat": str(args.recons_points_mat),
        "recons_sagplan_mat": str(args.recons_sagplan_mat),
        "data_recal_mat": str(args.data_recal_mat),
        "reconstruction_core_seconds": reconstruction_core_seconds,
        "reconstruction_total_seconds": total_wall_seconds,
    }
    if args.save_debug_npy:
        metadata_json["debug_xp_npy"] = str(debug_xp_path)
        metadata_json["debug_zp_npy"] = str(debug_zp_path)
        metadata_json["debug_transform_npy"] = str(debug_transform_path)
    metadata_path.write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")

    print(f"Saved exact recons3D output to: {mat_path}")
    print(f"Saved common-grid MHD to: {mhd_path}")
    print(f"Saved common-grid H5 bundle to: {grid_h5_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"data_recal shape: {tuple(np.asarray(data_recal).shape)}")
    print(f"xp/zp shape: {tuple(xp.shape)}")
    print(f"data_3D shape: {tuple(data_3D.shape)}")
    print(f"Core reconstruction time: {reconstruction_core_seconds:.2f} s")
    print(f"Total reconstruction time: {total_wall_seconds:.2f} s")


if __name__ == "__main__":
    main()
