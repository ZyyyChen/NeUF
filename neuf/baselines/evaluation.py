"""固定留出观测的定量/图像对比与统一世界网格导出。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
import torch

from .data import ROOT, local_grid, write_json


def predict(model, name, config, xyz):
    from .workflow import query
    values = []
    with torch.no_grad():
        for start in range(0, len(xyz), 16384):
            points = torch.as_tensor(xyz[start:start + 16384], dtype=torch.float32, device="cuda")
            values.append(query(model, name, config, points).detach().cpu().numpy())
    return np.concatenate(values)


def evaluate(model, name, config, data, output, step, smoke=False):
    model.eval()
    folder = output / "plots" / f"step_{step:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    stride = 8 if smoke else 1
    h, w = data["images"].shape[1:]
    rows, cols = np.mgrid[0:h:stride, 0:w:stride]
    local = local_grid(data, rows, cols).reshape(-1, 3)
    mask = binary_erosion(data["mask"], iterations=3)[::stride, ::stride]
    indices = data["valid"] if step == config["planned_steps"] and not smoke else data["selected"]
    metrics, panels, snapshots = [], [], []
    for frame in indices:
        world = local @ data["rotations"][frame].T + data["translations"][frame]
        result = predict(model, name, config, world).reshape(rows.shape)
        observed = data["images"][frame, ::stride, ::stride].astype(np.float32) / 255
        before = data["train"][data["train"] < frame][-1]
        after = data["train"][data["train"] > frame][0]
        alpha = (frame - before) / (after - before)
        baseline = ((1 - alpha) * data["images"][before, ::stride, ::stride].astype(np.float32)
                    + alpha * data["images"][after, ::stride, ::stride]) / 255
        for label, image in ((name, result), ("temporal_linear", baseline)):
            mse = float(np.square(image[mask] - observed[mask]).mean())
            _, ssim_map = structural_similarity(observed, image, data_range=1, full=True)
            grad_y, grad_x = np.gradient(image)
            target_y, target_x = np.gradient(observed)
            metrics.append(dict(step=step, frame=int(frame), model=label,
                                mae=float(np.abs(image[mask] - observed[mask]).mean()),
                                psnr=float(-10 * np.log10(max(mse, 1e-12))),
                                ssim=float(ssim_map[mask].mean()),
                                gradient_mae=float((np.abs(grad_y - target_y) + np.abs(grad_x - target_x))[mask].mean()),
                                mask_pixels=int(mask.sum()), stride=stride))
        if frame in data["selected"]:
            panels.append((int(frame), observed, baseline, result))
            snapshots.append(result)
    path = output / "metrics" / f"validation_step_{step:06d}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    fig, axes = plt.subplots(len(panels), 4, figsize=(12, len(panels) * 2.4), squeeze=False)
    for row, (frame, observed, baseline, result) in enumerate(panels):
        for ax, image, title in zip(axes[row], (observed, baseline, result, np.abs(result - observed)),
                                    (f"Input held-out {frame}", "Temporal linear", name, "Absolute error")):
            ax.imshow(image, cmap="magma" if title == "Absolute error" else "gray", vmin=0, vmax=0.3 if title == "Absolute error" else 1)
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle(f"{config['input']['case']} | step {step} | display [0,1] | input is observed US, no 3D GT")
    fig.tight_layout()
    fig.savefig(folder / "fixed_heldout.png", dpi=120)
    plt.close(fig)
    (output / "predictions").mkdir(exist_ok=True)
    np.savez_compressed(output / "predictions" / f"fixed_step_{step:06d}.npz", result=np.stack(snapshots),
                        observed=np.stack([p[1] for p in panels]), baseline=np.stack([p[2] for p in panels]),
                        indices=data["selected"], mask=mask)
    selected = [m for m in metrics if m["model"] == name]
    means = {key: float(np.mean([m[key] for m in selected])) for key in ("mae", "psnr", "ssim", "gradient_mae")}
    fixed = [m for m in selected if m["frame"] in data["selected"]]
    fixed_means = {key: float(np.mean([m[key] for m in fixed])) for key in means}
    curve_path = output / "metrics/validation_curve.csv"
    exists = curve_path.exists()
    with curve_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "count", *means])
        if not exists:
            writer.writeheader()
        writer.writerow(dict(step=step, count=len(fixed), **fixed_means))
    curve = np.genfromtxt(curve_path, delimiter=",", names=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, metric in zip(axes, ("mae", "psnr", "ssim")):
        ax.plot(np.atleast_1d(curve["step"]), np.atleast_1d(curve[metric]), "o-")
        ax.set(xlabel="Completed updates", ylabel=metric)
    fig.tight_layout()
    fig.savefig(folder / "metrics.png", dpi=120)
    plt.close(fig)
    write_json(output / "metrics" / f"evaluation_step_{step:06d}.json", dict(step=step, evaluated_frames=len(selected), metrics=means, fixed_metrics=fixed_means,
               comparison_indices=data["selected"].tolist(), mask="原序列固定扇区内腐蚀3像素", stride=stride,
               normalization="uint8 / 255，数值指标不裁剪预测，图片共用 [0,1]"))
    print(f"Evaluation step={step}: {means}", flush=True)


def export_volume(model, name, config, data, output):
    import nibabel as nib
    # 所有模型复用初始数据包围盒，长边最多256点，避免巨大空网格。
    bbox = data["bbox"]
    spacing = max(0.75, float(np.max(bbox[1] - bbox[0])) / 255)
    axes = [np.arange(bbox[0, i], bbox[1, i] + spacing * 0.5, spacing) for i in range(3)]
    zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    xyz = np.stack((xx, yy, zz), -1).reshape(-1, 3).astype(np.float32)
    model.eval()
    volume = predict(model, name, config, xyz).reshape(xx.shape).astype(np.float32)
    folder = output / "predictions"
    volume.tofile(folder / "volume_float.raw")
    affine = np.diag([spacing, spacing, spacing, 1])
    affine[:3, 3] = bbox[0]
    nii = nib.Nifti1Image(volume.transpose(2, 1, 0), affine)
    if data["metadata"]["unit"] == "mm":
        nii.header.set_xyzt_units("mm")
    nib.save(nii, folder / "volume_float.nii.gz")
    (folder / "volume_float.mhd").write_text(
        f"ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
        f"TransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = {' '.join(map(str,bbox[0]))}\n"
        f"ElementSpacing = {spacing} {spacing} {spacing}\nDimSize = {len(axes[0])} {len(axes[1])} {len(axes[2])}\n"
        "ElementType = MET_FLOAT\nElementDataFile = volume_float.raw\n")
    write_json(folder / "volume_geometry.json", dict(unit=data["metadata"]["unit"], spacing=spacing,
               bbox=bbox.tolist(), shape_zyx=list(volume.shape), support="初始扫描包围盒，包围盒外不导出；非实测区域可能外推"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image in zip(ax, (volume[len(volume)//2], volume[:, volume.shape[1]//2], volume[:, :, volume.shape[2]//2])):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    fig.suptitle(f"{name} | common initial world grid | spacing={spacing:.3f} {data['metadata']['unit']} | no 3D GT")
    fig.tight_layout()
    fig.savefig(output / "plots/volume_sections.png", dpi=140)
    plt.close(fig)


def report(run_id):
    root = ROOT / "logs" / run_id
    folder = root / "baselines/index_all/comparison"
    rows = []
    for path in sorted(root.glob("*/index_*/*/metrics/training_summary.json")):
        if "_smoke/" in str(path):
            continue
        result = json.loads(path.read_text())
        output = path.parent.parent
        config = json.loads((output / "run_config/config.json").read_text())
        step = result["completed_total_steps"]
        evaluation = output / "metrics" / f"evaluation_step_{step:06d}.json"
        metric = json.loads(evaluation.read_text())["metrics"] if evaluation.exists() else {}
        baseline = {}
        validation = output / "metrics" / f"validation_step_{step:06d}.csv"
        if validation.exists():
            with validation.open() as stream:
                values = [r for r in csv.DictReader(stream) if r["model"] == "temporal_linear"]
            baseline = {f"baseline_{key}": float(np.mean([float(r[key]) for r in values])) for key in ("mae", "psnr", "ssim")}
        rows.append(dict(case=config["input"]["case"], model=config["model_name"], status=result["status"],
                         steps=step, unit=result["unit"], wall_sec=result["session_wall_sec"],
                         **metric, **baseline, result_dir=str(output)))
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "metrics/results.json", rows)
    if rows:
        with (folder / "metrics/summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    text = "# ImplicitVol / NeSVoR 重建比较\n\n项目目标与测试原则：关注实际结构、几何、对比度与散斑；测试与 checkpoint 重载只验证实现，不能证明质量提高。\n\n"
    text += "留出观测不是三维 GT；比较仅限同病例、同划分、相同帧与掩膜。PSNR/SSIM 越高越好，MAE 越低越好；两模型架构与损失不同，等更新数不代表等计算量。\n\n"
    text += "|病例|模型|状态|步数|MAE↓|PSNR↑|SSIM↑|线性插值SSIM|耗时秒|单位|\n|---|---|---|---:|---:|---:|---:|---:|---:|---|\n"
    for row in rows:
        text += f"|{row['case']}|{row['model']}|{row['status']}|{row['steps']}|{row.get('mae', '未产生')}|{row.get('psnr', '未产生')}|{row.get('ssim', '未产生')}|{row.get('baseline_ssim','未产生')}|{row['wall_sec']:.2f}|{row['unit']}|\n"
    text += "\nPatient106 扫描为空；Recalage 两份相同 MAT 共用结果，不计作两个患者。Cropping 像素初始化须用户授权，结果不具有已验证的毫米尺度。\n"
    from .data import cases
    coverage = []
    for record in cases():
        for name in ("ImplicitVol", "NeSVoR"):
            done = next((row for row in rows if row["case"] == record["case"] and row["model"] == name), None)
            output = root / record["cohort"] / ("index_" + record["case"]) / name
            progress = output / "metrics/progress.json"
            steps = done["steps"] if done else json.loads(progress.read_text())["step"] if progress.exists() else 0
            status = done["status"] if done else "running" if (output / "run_config/config.json").exists() else "未训练：缺少核实的标定"
            checkpoints = {}
            for milestone in (5000, 7500, 10000):
                checkpoint = output / "checkpoints" / f"step_{milestone:06d}.pt"
                checkpoints[str(milestone)] = str(checkpoint) if checkpoint.is_file() else "未达到" if steps < milestone else "缺失"
            coverage.append(dict(case=record["case"], model=name, status=status, steps=steps, checkpoints=checkpoints))
    write_json(folder / "metrics/coverage.json", coverage)
    text += "\n## 独立 checkpoint 与覆盖范围\n\n"
    for entry in coverage:
        text += f"- {entry['case']} / {entry['model']}：{entry['status']}，{entry['steps']}/10000 步。\n"
        for milestone, path in entry["checkpoints"].items():
            text += f"  - step {milestone}：{path}\n"
    (folder / "plots").mkdir(exist_ok=True)
    for case in sorted({row["case"] for row in rows}):
        pair = [next((row for row in rows if row["case"] == case and row["model"] == name and row["status"] == "complete"), None)
                for name in ("ImplicitVol", "NeSVoR")]
        if not all(pair):
            continue
        samples = [np.load(Path(row["result_dir"]) / "predictions/fixed_step_010000.npz") for row in pair]
        if not all(np.array_equal(samples[0][key], samples[1][key]) for key in ("observed", "indices", "mask")):
            raise ValueError(f"两个模型的固定评价条件不一致：{case}")
        count = len(samples[0]["indices"])
        fig, axes = plt.subplots(count, 6, figsize=(17, count * 2.4), squeeze=False)
        for index, frame in enumerate(samples[0]["indices"]):
            observed, baseline = samples[0]["observed"][index], samples[0]["baseline"][index]
            a, b = samples[0]["result"][index], samples[1]["result"][index]
            for axis, image, title, error in zip(axes[index], (observed, baseline, a, b, abs(a-observed), abs(b-observed)),
                    (f"Input {frame}", "Temporal linear", "ImplicitVol", "NeSVoR", "ImplicitVol error", "NeSVoR error"),
                    (False, False, False, False, True, True)):
                axis.imshow(image, cmap="magma" if error else "gray", vmin=0, vmax=.3 if error else 1)
                axis.set_title(title)
                axis.axis("off")
        fig.suptitle(f"{case} | 10000 updates | fixed held-out frames | shared [0,1] | no 3D GT")
        fig.tight_layout()
        target = folder / "plots" / f"{case}_comparison.png"
        fig.savefig(target, dpi=120)
        plt.close(fig)
        text += f"\n- [{case} 固定图像对比]({target})\n"
    recovery = folder / "run_config/recovered_geometry.json"
    if recovery.exists():
        info = json.loads(recovery.read_text())
        text += (f"\n补充标定核对：Patient103 七参数模型标注 RMS={info.get('landmark_rms_pixels')} 像素，"
                 f"体积尺寸误差={info.get('extent_error_pixels')} 像素；原始 Cropping 到已配准图的方向核对"
                 f"状态={info['status']}。不能把二维 t_form 当作完整三维位姿。Patient103/104/105 的 delta_X 元数据未明确注明单位。\n")
    (folder / "metrics/comparison_report.md").write_text(text)
    print(text, flush=True)
