"""官方重建基线的数据、物理几何与固定划分。"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.spatial.transform import Rotation

from neuf.registration_comparison import load_volume

ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = ROOT / "logs/20261024_train01/registration/index_all/legacy_current/run_config/comparison_manifest.json"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不能序列化 {type(value)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")
    temporary.replace(path)


def inspect_inputs(output):
    """读取标定原表，避免把导出文件内固定的 1 mm 当成真实尺度。"""
    import torch
    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu_available": torch.cuda.is_available(), "reference_tables": {}}
    for name in ("nibabel", "tinycudann", "pydicom", "openpyxl"):
        result[name] = importlib.util.find_spec(name) is not None
    for path in sorted((ROOT / "code_seg_3D/Ref_Files").glob("*.mat")):
        contents = loadmat(path, simplify_cells=True)
        result["reference_tables"][path.name] = {}
        for key, value in contents.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            item["values"] = arr.tolist()
            result["reference_tables"][path.name][key] = item
    write_json(output, result)
    print(Path(output).read_text(), flush=True)


def inspect_reconstruction(output):
    """读取用户补充目录的变量表及小型参数，不加载巨大重建体。"""
    import h5py
    entries = []
    parent = ROOT / "code_seg_3D/Pre_traitement_echo_v2"
    for folder in ("Recalage", "Reconstruction_3D"):
        for path in sorted((parent / folder).rglob("*.mat")):
            entry = {"path": str(path.relative_to(ROOT)), "variables": {}}
            if h5py.is_hdf5(path):
                with h5py.File(path) as handle:
                    for name, value in handle.items():
                        if isinstance(value, h5py.Dataset):
                            item = {"shape": list(value.shape), "dtype": str(value.dtype)}
                            if value.size < 10000 and value.dtype.kind in "fiu":
                                item["values"] = np.asarray(value).tolist()
                            entry["variables"][name] = item
            else:
                for name, shape, kind in whosmat(path):
                    item = {"shape": list(shape), "dtype": kind}
                    if np.prod(shape) < 10000:
                        value = loadmat(path, variable_names=[name], simplify_cells=True)[name]
                        item["values"] = np.asarray(value).tolist()
                    entry["variables"][name] = item
            entries.append(entry)
    write_json(output, entries)
    print(Path(output).read_text(), flush=True)


def cases():
    """复用已逐体素核对的清单；两个内容相同的 MAT 共用一个训练结果。"""
    config = json.loads(INPUT_MANIFEST.read_text())
    return [c for c in config["cases"] if not c["case"].endswith("_matlab")]


def load_case(case_id, pixel_geometry=False):
    record = next(c for c in cases() if c["case"] == case_id)
    volume = load_volume(ROOT / record["path"], record["variable"])
    if volume.ndim != 3 or min(volume.shape) == 0:
        raise ValueError(f"无有效三维序列：{record}")
    images = np.ascontiguousarray(volume.transpose(2, 0, 1))
    n, h, w = images.shape
    rotations = np.repeat(np.eye(3)[None], n, axis=0)
    translations = np.zeros((n, 3))
    if record["cohort"] == "cropping":
        if not pixel_geometry:
            raise ValueError("Cropping 缺少可对应的完整标定，像素几何需要用户明确授权")
        dx = dy = thickness = 1.0
        translations[:, 2] = np.arange(n) - (n - 1) / 2
        unit, source = "pixel", "用户授权的等间距平行切片初始化；不是测量位姿或毫米标定"
    else:
        folder = ROOT / "data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original"
        infos = json.loads((folder / "infos.json").read_text())
        dx = float(infos["infos"]["px_size_cm"]["width"]) * 10
        dy = float(infos["infos"]["px_size_cm"]["height"]) * 10
        ox = oy = 0
        if case_id == "us_recal_original_cropped":
            roi = json.loads((folder / "infos_roi.json").read_text())["infos"]["ROI"]
            ox, oy = roi["x"], roi["y"]
            if (h, w) != (roi["height"], roi["width"]):
                raise ValueError("裁剪图尺寸与 ROI 不符")
        original_w = infos["infos"]["scan_dims_px"]["width"]
        local_center = np.array([(oy + h / 2) * dy, (ox + w / 2 - original_w / 2) * dx, 0])
        # 既有 NeUF 的局部坐标是 (深度,横向,0)，转换成右手 (横向,深度,法向)。
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        for i in range(n):
            frame = infos[str(i)]
            rotation = Rotation.from_quat([frame["w1"], frame["w2"], frame["w3"], frame["w0"]]).as_matrix()
            rotations[i] = rotation @ swap
            translations[i] = rotation @ local_center + [frame["x"], frame["y"], frame["z"]]
        # 现有资料没有实测 elevational PSF，用相邻帧中心距离作为显式记录的厚度假设。
        thickness = float(np.median(np.linalg.norm(np.diff(translations, axis=0), axis=1)))
        thickness = max(thickness, min(dx, dy))
        unit, source = "mm", "infos.json；裁剪图只应用 infos_roi.json 的 ROI，沿用原图位姿"
    if not np.allclose(np.linalg.det(rotations), 1):
        raise ValueError("切片坐标不是右手刚体变换")
    valid = np.arange(5, n - 1, 10)
    train = np.setdiff1d(np.arange(n), valid)
    selected = valid[np.linspace(0, len(valid) - 1, min(5, len(valid))).round().astype(int)]
    sampled = images[np.linspace(0, n - 1, min(16, n)).round().astype(int)]
    mask = np.max(sampled, axis=0) > 2
    if mask.sum() < 100:
        raise ValueError("扫描没有足够的非背景像素")
    corners = np.array([[-(w - 1) * dx / 2, -(h - 1) * dy / 2, 0],
                        [(w - 1) * dx / 2, -(h - 1) * dy / 2, 0],
                        [-(w - 1) * dx / 2, (h - 1) * dy / 2, 0],
                        [(w - 1) * dx / 2, (h - 1) * dy / 2, 0]])
    world = np.einsum("nij,kj->nki", rotations, corners) + translations[:, None]
    bbox = np.stack((world.min((0, 1)), world.max((0, 1))))
    bbox += np.array([[-1], [1]]) * max(dx, dy, thickness)
    signature = hashlib.sha256(images.tobytes()).hexdigest()
    metadata = dict(record, shape_nhw=[n, h, w], unit=unit, geometry_source=source,
                    dx=dx, dy=dy, assumed_thickness=thickness,
                    thickness_source="相邻帧中心距离与像素间距的较大者；未测量 PSF",
                    bbox=bbox.tolist(), train_indices=train.tolist(), validation_indices=valid.tolist(),
                    comparison_indices=selected.tolist(), input_sha256=signature,
                    normalization="uint8 / 255；共用显示 [0,1]", sector_threshold=2)
    return dict(images=images, rotations=rotations.astype(np.float32),
                translations=translations.astype(np.float32), mask=mask,
                metadata=metadata, train=train, valid=valid, selected=selected, bbox=bbox)


def local_grid(data, rows, cols):
    h, w = data["images"].shape[1:]
    meta = data["metadata"]
    return np.stack(((cols - (w - 1) / 2) * meta["dx"],
                     (rows - (h - 1) / 2) * meta["dy"], np.zeros_like(rows)), -1).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--inspect-reconstruction", type=Path)
    args = parser.parse_args()
    if args.inspect:
        inspect_inputs(args.inspect)
    else:
        inspect_reconstruction(args.inspect_reconstruction)
