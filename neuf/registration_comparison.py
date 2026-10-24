"""逐病例比较历史与当前配准实现，保留原算法的数值计算。"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat, whosmat
from scipy.ndimage import binary_erosion


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def inventory(repo: Path, output: Path) -> dict:
    """检查全部文件格式与维度，不将参考矢状图当作扫描序列。"""
    roots = [repo / "code_seg_3D/Pre_traitement_echo_v2/Cropping",
             repo / "data/cerebral_data/Pre_traitement_echo_v2/Recalage"]
    entries = []
    for root in roots:
        for path in sorted(root.rglob("*.mat")):
            record = {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size}
            try:
                if h5py.is_hdf5(path):
                    with h5py.File(path) as handle:
                        record["variables"] = {key: {"shape": list(value.shape),
                            "dtype": str(value.dtype), "matlab_class": str(value.attrs.get("MATLAB_class", ""))}
                            for key, value in handle.items() if isinstance(value, h5py.Dataset)}
                else:
                    record["variables"] = {name: {"shape": list(shape), "dtype": kind}
                                           for name, shape, kind in whosmat(path)}
            except Exception as error:
                record["error"] = repr(error)
            entries.append(record)
        for path in sorted(root.rglob("*.dcm")):
            record = {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size}
            try:
                import pydicom
                ds = pydicom.dcmread(path, stop_before_pixels=True,
                    specific_tags=["Rows", "Columns", "NumberOfFrames", "BitsAllocated", "SamplesPerPixel"])
                record["dicom"] = {name: str(getattr(ds, name, "")) for name in
                    ["Rows", "Columns", "NumberOfFrames", "BitsAllocated", "SamplesPerPixel"]}
            except Exception as error:
                record["error"] = repr(error)
            entries.append(record)
        folders = sorted({p.parent for p in root.rglob("us*.jpg") if re.fullmatch(r"us\d+\.jpg", p.name)})
        for folder in folders:
            files = sorted(folder.glob("us*.jpg"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
            with Image.open(files[0]) as image:
                size = list(image.size)
            entries.append({"path": str(folder.relative_to(repo)), "image_count": len(files),
                            "first_image_size_wh": size, "first": files[0].name, "last": files[-1].name})
    versions = {}
    for name in ["numpy", "scipy", "h5py", "pydicom", "Pillow", "matplotlib", "scikit-image", "nbformat", "nbclient"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"entries": entries, "versions": versions,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    write_json(output / "run_config/inventory.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


class HeadlessApplication:
    """历史实现无 GUI 时原本就跳过进度对话框。"""

    @staticmethod
    def instance():
        return None


class ConsoleProgress:
    """只替换新版 GUI 进度接口，不改动配准数值计算。"""

    def __init__(self, label="registration"):
        self.label, self.last = label, -1

    def phase(self, label, *bounds):
        self.last = -1

    def avancer(self, fraction):
        percent = min(100, int(fraction * 100))
        if percent // 25 > self.last // 25:
            print(f"{self.label}: {percent}%", flush=True)
            self.last = percent

    def annulee(self):
        return False

    def fermer(self):
        pass


def algorithm(source: Path, snapshot: Path):
    """消解完全相同的冲突副本，并仅移除 Qt/进度条导入。"""
    text = source.read_text()
    if text.startswith("<<<<<<<"):
        lines = text.splitlines()
        split = next(i for i, line in enumerate(lines) if line == "=======")
        end = next(i for i, line in enumerate(lines) if line.startswith(">>>>>>>"))
        first, second = "\n".join(lines[1:split]).strip(), "\n".join(lines[split+1:end]).strip()
        if first != second:
            raise ValueError("冲突两侧不同，不能自动选择算法")
        text = first
    tree = ast.parse(text)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
        (node.module.startswith("PySide6") or node.module == "tools.barre_progression"))]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(ast.unparse(tree) + "\n")
    namespace = {"QApplication": HeadlessApplication, "BarreEtape": ConsoleProgress}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["data_registration_v2"]


def load_volume(path: Path, variable: str | None = None) -> np.ndarray:
    """所有输入统一为 H×W×N；本项目 DICOM 须逆转导出时的翻转/转轴。"""
    if path.is_dir():
        frames = sorted(path.glob("us*.jpg"), key=lambda p: int(re.fullmatch(r"us(\d+)\.jpg", p.name).group(1)))
        indices = [int(re.fullmatch(r"us(\d+)\.jpg", p.name).group(1)) for p in frames]
        if indices != list(range(len(indices))):
            raise ValueError(f"帧编号不连续: {path}")
        return np.stack([np.asarray(Image.open(p).convert("L")) for p in frames], axis=2)
    if path.suffix == ".dcm":
        import pydicom
        pixels = pydicom.dcmread(path).pixel_array
        # 对应 tools/export_to_3d_dicom.py 的 transpose(0,2,1) 与 flip(axis=0)。
        return np.flip(pixels, axis=0).transpose(0, 2, 1).copy()
    if h5py.is_hdf5(path):
        with h5py.File(path) as handle:
            data = np.asarray(handle[variable])
        return data.transpose(tuple(reversed(range(data.ndim))))
    return loadmat(path, variable_names=[variable])[variable]


def prepare(repo: Path, output: Path) -> dict:
    info = json.loads((output / "run_config/inventory.json").read_text())
    entries = {item["path"]: item for item in info["entries"]}
    cases, excluded, verified = [], [], []
    crop = repo / "code_seg_3D/Pre_traitement_echo_v2/Cropping"
    for folder in sorted(crop.iterdir()):
        if not folder.is_dir():
            continue
        mat = folder / f"data_repos_{folder.name}.mat"
        dcm = folder / f"data_repos_{folder.name}.dcm"
        path, variable = (mat, "data_repos") if mat.exists() else (dcm, None)
        record = {"case": folder.name, "cohort": "cropping", "path": str(path.relative_to(repo)), "variable": variable}
        if mat.exists():
            shape = entries[str(mat.relative_to(repo))]["variables"]["data_repos"]["shape"]
            if len(shape) != 3 or min(shape) == 0:
                excluded.append(dict(record, reason=f"空扫描序列，shape={shape}"))
                continue
            if dcm.exists():
                original, exported = load_volume(mat, variable), load_volume(dcm)
                # 导出脚本先按全体最大值缩放；全量核对方向，避免把深度轴当帧轴。
                reference = (original.astype(np.float32) / max(int(original.max()), 1) * 255).astype(np.uint8)
                equal = original.shape == exported.shape and np.array_equal(reference, exported)
                verified.append({"case": folder.name, "shape": list(original.shape), "dicom_inverse_matches_export": bool(equal)})
                if not equal:
                    raise ValueError(f"DICOM 方向核对失败: {folder.name}")
        elif not dcm.exists():
            excluded.append(dict(record, reason="缺少扫描序列 MAT 与 DICOM"))
            continue
        cases.append(record)
    for relative, entry in entries.items():
        if not relative.startswith("data/cerebral_data/"):
            continue
        if "data_recal" in entry.get("variables", {}):
            cases.append({"case": Path(relative).stem, "cohort": "recalage_mat", "path": relative, "variable": "data_recal"})
        elif "image_count" in entry:
            cases.append({"case": Path(relative).name, "cohort": "recalage_jpeg", "path": relative, "variable": None})
    config = {"cases": cases, "excluded": excluded, "dicom_validation": verified,
              "d": 0.5, "current_step": 1, "current_roi_rigid": False,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": 0, "normalization": "不改变输入灰度；要求 uint8 0–255；JPEG 按原生解码值单列",
              "scope": "MAT/DICOM 同一序列只选一种主输入；所有 Recalage MAT 和两组 JPEG 分别运行；2D data_sag 为参考，不作为帧序列",
              "quality_limit": "无已知真实运动或重建 GT；相邻帧一致性、梯度与运行时间均不能单独证明重建质量提高"}
    write_json(output / "run_config/comparison_manifest.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    return config


def ncc(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = a.astype(np.float64), b.astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else None


def compare_metrics(source: np.ndarray, old: np.ndarray, new: np.ndarray, a: np.ndarray, b: np.ndarray):
    mask = binary_erosion(np.any(source > 0, axis=2), iterations=3)
    if not mask.any():
        raise ValueError("输入没有有效扇区")
    rows, different, total_error = [], 0, 0.0
    for frame in range(source.shape[2]):
        delta = new[:, :, frame].astype(np.int16) - old[:, :, frame].astype(np.int16)
        different += int(np.count_nonzero(delta))
        total_error += float(np.square(delta.astype(np.float64)).sum())
        row = {"frame": frame, "legacy_current_mae": float(np.abs(delta).mean()),
               "legacy_current_max_abs": int(np.abs(delta).max()),
               "transform_max_abs_difference": float(np.abs(a[frame] - b[frame]).max())}
        for name, volume in [("input", source), ("legacy", old), ("current", new)]:
            image = volume[:, :, frame]
            row[f"{name}_zero_fraction_fixed_sector"] = float((image[mask] == 0).mean())
            gx = np.diff(image.astype(np.float32), axis=1)[mask[:, :-1]]
            gy = np.diff(image.astype(np.float32), axis=0)[mask[:-1, :]]
            row[f"{name}_gradient_rms"] = float(np.sqrt((np.square(gx).mean() + np.square(gy).mean()) / 2))
            if frame:
                row[f"{name}_adjacent_ncc"] = ncc(volume[:, :, frame-1][mask], image[mask])
        rows.append(row)
    summary = {"voxel_count": int(old.size), "different_voxels": different,
               "pixel_identical": different == 0, "legacy_current_rmse": float(np.sqrt(total_error / old.size)),
               "legacy_current_max_abs": max(row["legacy_current_max_abs"] for row in rows),
               "transform_max_abs_difference": float(np.abs(a - b).max()),
               "transform_identical": bool(np.array_equal(a, b))}
    for name in ["input", "legacy", "current"]:
        for metric in ["adjacent_ncc", "gradient_rms", "zero_fraction_fixed_sector"]:
            values = [row[f"{name}_{metric}"] for row in rows if row.get(f"{name}_{metric}") is not None]
            summary[f"{name}_{metric}_mean"] = float(np.mean(values)) if values else None
    return summary, rows


def run_case(repo: Path, output: Path, record: dict, d: float) -> dict:
    case_dir = output.parents[2] / record["cohort"] / f"index_{record['case']}" / "legacy_current"
    case_dir.mkdir(parents=True, exist_ok=True)
    result = dict(record, output=str(case_dir), status="running")
    write_json(case_dir / "metrics/status.json", result)
    try:
        np.random.seed(0)
        source = np.ascontiguousarray(load_volume(repo / record["path"], record["variable"]))
        if source.ndim != 3 or min(source.shape) < 2 or source.dtype != np.uint8:
            raise ValueError(f"要求非空 uint8 H×W×N，实际 {source.shape}, {source.dtype}")
        if source.max() <= 1:
            raise ValueError("扫描序列强度为空或为 0–1，不能套用原算法阈值")
        result.update(shape=list(source.shape), input_min=int(source.min()), input_max=int(source.max()),
                      input_sha256=hashlib.sha256(source.tobytes()).hexdigest())
        variants, transforms = {}, {}
        for name in ["legacy", "current"]:
            function = algorithm(output / f"run_config/source/{name}_original.py", case_dir / f"run_config/{name}_headless.py")
            started = time.perf_counter()
            print(f"START {record['case']} {name}: shape={source.shape}, d={d}", flush=True)
            if name == "legacy":
                registered, matrices = function(d, source)
            else:
                registered, matrices = function(d, source, step=1, roi_rigid=False,
                                                 barre=ConsoleProgress(record["case"]))
            result[f"{name}_registration_seconds"] = time.perf_counter() - started
            if registered is None or registered.shape != source.shape or len(matrices) != source.shape[2]:
                raise ValueError(f"{name} 返回的体数据/矩阵数量不完整")
            variants[name] = registered
            transforms[name] = np.stack([matrices[i] for i in range(source.shape[2])])
            target = case_dir / f"predictions/{name}/data_recal.mat"
            target.parent.mkdir(parents=True, exist_ok=True)
            savemat(target, {"data_recal": registered, "t_form": transforms[name], "d": d,
                            "source_path": record["path"], "algorithm": name}, do_compression=True)
            # 读取实际输出，核对可独立复用，既不改写输入，也不依赖内存对象。
            saved = loadmat(target, variable_names=["data_recal", "t_form"])
            if not np.array_equal(saved["data_recal"], registered) or not np.array_equal(saved["t_form"], transforms[name]):
                raise ValueError(f"{name} 保存后读取不一致")
            del saved
            print(f"DONE {record['case']} {name}: {result[f'{name}_registration_seconds']:.2f}s", flush=True)
        summary, rows = compare_metrics(source, variants["legacy"], variants["current"], transforms["legacy"], transforms["current"])
        result.update(summary, status="complete")
        frame_ids = sorted({0, source.shape[2]//4, source.shape[2]//2, 3*source.shape[2]//4, source.shape[2]-1})
        columns = sorted({source.shape[1]//4, source.shape[1]//2, 3*source.shape[1]//4})
        write_json(case_dir / "run_config/comparison_manifest.json", dict(record, shape=list(source.shape),
            frame_indices=frame_ids, sagittal_columns=columns, gray_range=[0, 255], d=d, current_step=1,
            roi_rigid=False, transform_convention="t_form 是原函数的中心旋转参数矩阵，不等同于三维探头位姿"))
        samples = {}
        for name, volume in [("input", source), *variants.items()]:
            samples[f"{name}_frames"] = volume[:, :, frame_ids]
            samples[f"{name}_sagittal"] = volume[:, columns, :]
        np.savez_compressed(case_dir / "metrics/comparison_samples.npz", **samples,
                            legacy_transforms=transforms["legacy"], current_transforms=transforms["current"])
        write_csv(case_dir / "metrics/per_frame.csv", rows)
    except Exception as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        print(f"FAILED {record['case']}: {result['error']}", flush=True)
    write_json(case_dir / "metrics/status.json", result)
    return result


def run_all(repo: Path, output: Path, workers: int) -> None:
    config = prepare(repo, output)
    started = time.perf_counter()
    rows = []
    write_json(output / "metrics/status.json", {"status": "running", "planned": len(config["cases"]), "completed": 0})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_case, repo, output, record, config["d"]): record for record in config["cases"]}
        for future in as_completed(pending):
            record = pending[future]
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append(dict(record, status="failed", error=repr(error)))
            rows.sort(key=lambda row: (row["cohort"], row["case"]))
            write_csv(output / "metrics/summary.csv", rows)
            write_json(output / "metrics/results.json", rows)
            write_json(output / "metrics/status.json", {"status": "running", "planned": len(config["cases"]),
                       "finished": len(rows), "elapsed_seconds": time.perf_counter()-started})
            print(f"CASES {len(rows)}/{len(config['cases'])}: {record['case']} {rows[-1]['status']}", flush=True)
    state = {"status": "complete" if all(row["status"] == "complete" for row in rows) else "partial_failure",
             "planned": len(config["cases"]), "completed": sum(row["status"] == "complete" for row in rows),
             "excluded": config["excluded"], "elapsed_seconds": time.perf_counter()-started,
             "checkpoint_milestones": "不适用：仅运行经典配准，无训练"}
    write_json(output / "metrics/status.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if state["status"] != "complete":
        raise RuntimeError("部分病例失败，详见逐病例 status.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["inventory", "compare"], default="inventory")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "inventory":
        inventory(args.repo, args.output)
    else:
        run_all(args.repo, args.output, args.workers)


if __name__ == "__main__":
    main()
