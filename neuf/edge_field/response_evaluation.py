"""单通道 response 的切片、支持区域距离及三维体可视化。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .data import write_json
from .evaluation import image_metrics, render, write_csv


def response_metrics(prediction, teacher, mask, spacing):
    result = image_metrics(prediction, teacher, mask)
    result["mae"] = float(np.abs(prediction-teacher)[mask].mean())
    foreground, background = mask & (teacher > .15), mask & (teacher <= .15)
    for name, region in (("foreground", foreground), ("background", background)):
        result[f"{name}_mae"] = float(np.abs(prediction-teacher)[region].mean()) if region.any() else None
    # 这是阈值化 response 支持区域的距离，不是人工标注的解剖边界误差。
    for threshold in (.15, .3, .5):
        p, t = (prediction > threshold) & mask, (teacher > threshold) & mask
        prefix = f"support_{threshold:g}"
        overlap = int((p & t).sum())
        result[f"{prefix}_precision"] = overlap/int(p.sum()) if p.any() else None
        result[f"{prefix}_recall"] = overlap/int(t.sum()) if t.any() else None
        result[f"{prefix}_distance_mm"] = float(.5 * (
            distance_transform_edt(~p, sampling=spacing)[t].mean()
            + distance_transform_edt(~t, sampling=spacing)[p].mean())) if p.any() and t.any() else None
    return result


@torch.no_grad()
def evaluate_alignment(data, reference, poses, output, step, *, smoke=False):
    """固定初始支持域；验证/测试只由训练帧插值，自身位姿与 edge 不参与优化。"""
    from .losses import local_correlation

    reference.set_scale(0.)
    matrices = poses.matrices().detach()
    display = set(data.comparison + torch.where(data.frame_ids == 121)[0].cpu().tolist())
    split_map = {int(i): split for split, indices in data.splits.items() for i in indices}
    indices = sorted(display) if smoke else list(range(len(data.frame_ids)))
    rows = []
    for index in indices:
        frame = int(data.frame_ids[index])
        row = dict(frame_id=frame, split=split_map[index], step=step, supported=bool(reference.valid_frames[index]))
        if not row["supported"]:
            row.update(valid_pixels=0, mae=None, correlation_loss=None, coverage_loss=None)
            rows.append(row)
            continue
        frames = torch.tensor([index], device=data.edges.device)
        stride = 3
        target = data.edges[index, ::stride, ::stride]
        predicted, mask, coverage = reference.sample(frames, data.local[::stride, ::stride].reshape(1, -1, 3), matrices)
        predicted, mask, coverage = [v.reshape_as(target) for v in (predicted, mask, coverage)]
        row["valid_pixels"] = int(mask.sum())
        if row["valid_pixels"] < 100:
            row.update(supported=False, mae=None, correlation_loss=None, coverage_loss=None)
            rows.append(row)
            continue
        row.update(response_metrics(predicted.cpu().numpy(), target.cpu().numpy(), mask.cpu().numpy(),
                                    tuple(s*stride for s in data.spacing)))
        row.update(correlation_loss=float(local_correlation(predicted[None, None], target[None, None], mask[None, None])),
                   coverage_loss=float((1-coverage)[mask].mean()))
        rows.append(row)
        if index in display:
            predicted, mask, _ = reference.sample(frames, data.local.reshape(1, -1, 3), matrices)
            np.savez_compressed(output/"predictions"/f"frame_{frame:04d}_step_{step:06d}.npz",
                                target=data.edges[index].cpu().numpy(),
                                reference=predicted.reshape(data.height, data.width).cpu().numpy(),
                                mask=mask.reshape(data.height, data.width).cpu().numpy())
    # 无双侧支持的帧保留空指标，不把它们当作零误差混进平均数。
    keys = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(output/"metrics"/f"alignment_{step:06d}.csv", [{k: row.get(k) for k in keys} for row in rows])
    return rows


@torch.no_grad()
def summarize_alignment(data, reference, poses, output, before, after, history, step):
    summaries = []
    metric_keys = ("mae", "correlation_loss", "coverage_loss", "ssim", "support_0.3_distance_mm")
    for split in ("training", "validation", "test"):
        paired = [(a, b) for a, b in zip(before, after) if a["split"] == split and a["supported"] and b["supported"]]
        if not paired:
            continue
        summary = dict(split=split, count=len(paired), metric_stride=3, aggregation="equal weight per supported frame")
        for key in metric_keys:
            valid = [(a[key], b[key]) for a, b in paired if a.get(key) is not None and b.get(key) is not None]
            summary[f"{key}_before"] = float(np.mean([v[0] for v in valid])) if valid else None
            summary[f"{key}_after"] = float(np.mean([v[1] for v in valid])) if valid else None
        summaries.append(summary)
    write_csv(output/"metrics"/f"summary_{step:06d}.csv", summaries)
    validation = next((r for r in summaries if r["split"] == "validation"), None)
    improved = bool(validation and validation["count"] == len(data.splits["validation"])
                    and validation["mae_after"] < validation["mae_before"]
                    and validation["correlation_loss_after"] < validation["correlation_loss_before"]
                    and validation["coverage_loss_after"] < 1e-3)
    write_json(output/"metrics/comparison_summary.json", dict(
        step=step, results=summaries, heldout_consistency_improved=improved,
        recommended_pose_source="corrected candidate; pose accuracy unverified" if improved else "initial poses",
        input_poses_modified=False,
        interpretation="edge consistency only; real pose accuracy unknown",
        geometry="slice planes and normals fixed; only in-plane rigid corrections",
        source_policy="initial supports fixed; references contain training frames only, never the target itself",
        limitations=["nearby-section continuity is an assumption", "no out-of-plane correction",
                     "response is not anatomical ground truth", "unsupported frames remain at initial pose"]))
    directory = output/"plots"/f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    display_ids = [*data.metadata["comparison_indices"], 121]
    c, r, w, h = data.metadata["comparison_roi_xywh"]
    for frame in display_ids:
        initial_path = output/"predictions"/f"frame_{frame:04d}_step_000000.npz"
        final_path = output/"predictions"/f"frame_{frame:04d}_step_{step:06d}.npz"
        if not initial_path.exists() or not final_path.exists():
            continue
        with np.load(initial_path) as initial, np.load(final_path) as final:
            target, mask = initial["target"], initial["mask"]
            assert np.array_equal(mask, final["mask"])
            panels = (target, initial["reference"], final["reference"],
                      np.abs(final["reference"]-target)-np.abs(initial["reference"]-target))
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
            for col, (title, panel) in enumerate(zip(("NLSTV target", "Other slices: before", "Other slices: after", "Error change: blue = lower"), panels)):
                image = np.where(mask, panel, np.nan)
                settings = dict(cmap="coolwarm", vmin=-.3, vmax=.3) if col == 3 else dict(cmap="gray", vmin=0, vmax=1)
                shown = axes[0, col].imshow(image, **settings)
                axes[1, col].imshow(image[r:r+h, c:c+w], **settings)
                axes[0, col].set_title(title)
                axes[1, col].set_title(f"ROI x={c}, y={r}, {w}x{h}")
                if col == 3:
                    fig.colorbar(shown, ax=list(axes[:, col]), shrink=.5)
                for ax in axes[:, col]:
                    ax.axis("off")
            fig.suptitle(f"cerebral frame {frame}; EdgeAlign step {step}; native pixels; fixed initial support")
            fig.savefig(directory/f"frame_{frame:04d}.png", dpi=130)
            plt.close(fig)
    matrices = poses.matrices().cpu().numpy()
    initial = data.initial.cpu().numpy()
    corrections = poses.corrections().cpu().numpy()
    records = []
    for i, frame in enumerate(data.frame_ids.cpu().tolist()):
        row = dict(frame_id=frame, optimized=bool(reference.active[i]),
                   center_axial_correction_mm=float(corrections[i, 0]), center_lateral_correction_mm=float(corrections[i, 1]),
                   inplane_rotation_deg=float(np.degrees(corrections[i, 2])),
                   probe_translation_change_mm=float(np.linalg.norm(matrices[i, :3, 3]-initial[i, :3, 3])))
        row.update({f"initial_{axis}_mm": float(initial[i, j, 3]) for j, axis in enumerate("xyz")})
        row.update({f"corrected_{axis}_mm": float(matrices[i, j, 3]) for j, axis in enumerate("xyz")})
        row.update({f"r{j}{k}": float(matrices[i, j, k]) for j in range(3) for k in range(3)})
        records.append(row)
    write_csv(output/"metrics/pose_changes.csv", records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key in ("shape", "response"):
        values = np.array([r[key] for r in history])
        window = min(25, len(values))
        axes[0, 0].plot(np.arange(window, len(values)+1), np.convolve(values, np.ones(window)/window, mode="valid"), label=key)
    axes[0, 0].set(xlabel="Optimization step", ylabel="Loss")
    ids = data.frame_ids.cpu().numpy()
    axes[0, 1].plot(ids, corrections[:, 0], label="Axial")
    axes[0, 1].plot(ids, corrections[:, 1], label="Lateral")
    axes[0, 1].set(xlabel="Original frame", ylabel="Image-center correction [mm]")
    axes[1, 0].plot(ids, np.degrees(corrections[:, 2]), label="In-plane rotation")
    axes[1, 0].set(xlabel="Original frame", ylabel="Correction [degree]")
    axes[1, 1].plot(initial[:, 0, 3], initial[:, 1, 3], label="Initial")
    axes[1, 1].plot(matrices[:, 0, 3], matrices[:, 1, 3], label="Corrected")
    axes[1, 1].set(xlabel="Probe world x [mm]", ylabel="Probe world y [mm]")
    for ax in axes.flat:
        ax.legend()
    fig.suptitle("Edge-only in-plane pose alignment; corrections are not ground-truth errors")
    fig.savefig(directory/"pose_alignment.png", dpi=140)
    plt.close(fig)


def evaluate_response(model, poses, data, output, step, progress, *, final=False):
    output = Path(output)
    directory = output / "plots" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    matrices = poses.matrices().detach()
    train_preview = int(data.splits["training"][len(data.splits["training"])//2])
    display = set(data.comparison + [train_preview])
    indices = [(split, int(index)) for split in (("validation", "test") if final else ("validation",))
               for index in data.splits[split].cpu().tolist()]
    indices.append(("training_diagnostic", train_preview))
    rows = []
    valid, interior = data.mask.cpu().numpy(), data.interior.cpu().numpy()
    for split, index in indices:
        # 原生像素网格；验证和测试的 teacher 不用于位姿拟合或更新结构场。
        predicted = render(model, matrices[index], data.local, progress)[..., 0]
        teacher = data.edges[index].cpu().numpy()
        frame_id = int(data.frame_ids[index])
        rows.append(dict(split=split, frame_id=frame_id, step=step, grid="native",
                         **response_metrics(predicted, teacher, interior, data.spacing)))
        if index not in display:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        c, r, w, h = data.metadata["comparison_roi_xywh"]
        panels = (teacher, predicted, np.abs(teacher-predicted))
        for col, (title, panel) in enumerate(zip(("NLSTV teacher", "Reconstructed response", "Absolute error"), panels)):
            axes[0, col].imshow(np.where(valid, panel, np.nan), cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(title)
            axes[1, col].imshow(np.where(valid, panel, np.nan)[r:r+h, c:c+w], cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title(f"Fixed ROI x={c}, y={r}, {w}x{h}")
            for ax in axes[:, col]:
                ax.axis("off")
        fig.suptitle(f"{output.name}; cerebral {frame_id}; {split}; step {step}; native [0,1]")
        fig.savefig(directory / f"frame_{frame_id:04d}.png", dpi=140)
        plt.close(fig)
        if final:
            np.savez_compressed(output / "predictions" / f"frame_{frame_id:04d}.npz",
                                edge=predicted, teacher=teacher, mask=valid)
    write_csv(output / "metrics" / f"images_{step:06d}.csv", rows)
    return rows


def compare_responses(data, run_root, variants):
    root = Path(run_root)
    output = root / "comparison"
    for folder in ("plots", "metrics", "run_config"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    write_json(output/"run_config/comparison_manifest.json", dict(
        **data.metadata, response_only=True, metric_grid="native",
        support_distance="symmetric mean distance between thresholded response supports; not anatomical ground truth"))
    for index in data.comparison:
        frame = int(data.frame_ids[index])
        teacher = data.edges[index].cpu().numpy()
        images = {"Teacher": teacher}
        for name in variants:
            with np.load(root/name/"predictions"/f"frame_{frame:04d}.npz") as saved:
                images[name] = saved["edge"]
        fig, axes = plt.subplots(2, len(images), figsize=(5*len(images), 8), squeeze=False, layout="constrained")
        c, r, w, h = data.metadata["comparison_roi_xywh"]
        mask = data.mask.cpu().numpy()
        for col, (name, response) in enumerate(images.items()):
            response = np.where(mask, response, np.nan)
            axes[0, col].imshow(response, cmap="gray", vmin=0, vmax=1)
            axes[1, col].imshow(response[r:r+h, c:c+w], cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(name)
            axes[1, col].set_title(f"Fixed ROI x={c}, y={r}, {w}x{h}")
            for ax in axes[:, col]:
                ax.axis("off")
        fig.suptitle(f"cerebral {frame}; response only; native pixels; fixed [0,1]; heldout poses fixed")
        fig.savefig(output/"plots"/f"response_{frame:04d}.png", dpi=150)
        plt.close(fig)
    volumes = [np.load(root/name/"predictions/edge_volume.npy", mmap_mode="r") for name in variants]
    supports = [np.load(root/name/"predictions/volume_support.npy", mmap_mode="r") for name in variants]
    common_support = np.logical_and.reduce(supports)
    fig, axes = plt.subplots(len(variants), 6, figsize=(21, 4*len(variants)), squeeze=False, layout="constrained")
    spacing = float(__import__("json").loads((root/variants[0]/"predictions/volume_metadata.json").read_text())["spacing_xyz_mm"][0])
    for row, (name, volume) in enumerate(zip(variants, volumes)):
        shown = np.where(common_support, volume, 0)
        for axis, labels in enumerate((("x", "y"), ("x", "z"), ("y", "z"))):
            section = np.take(shown, shown.shape[axis]//2, axis=axis)
            projection = shown.max(axis=axis)
            for col, panel, title in ((axis, section, "central plane"), (axis+3, projection, "maximum projection")):
                ax = axes[row, col]
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1, origin="lower",
                          extent=[0, panel.shape[1]*spacing, 0, panel.shape[0]*spacing])
                ax.set(title=f"{name}: {title}", xlabel=f"{labels[0]} from volume origin [mm]", ylabel=f"{labels[1]} from volume origin [mm]")
    fig.suptitle("3D response; fixed world planes; common observation support; [0,1]")
    fig.savefig(output/"plots/edge_volume_comparison.png", dpi=150)
    plt.close(fig)
    aggregates, corrections = [], {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    for name in variants:
        path = sorted((root/name/"metrics").glob("images_*.csv"))[-1]
        with path.open() as handle:
            records = list(csv.DictReader(handle))
        for split in ("validation", "test", "training_diagnostic"):
            subset = [r for r in records if r["split"] == split]
            keys = [k for k in records[0] if k not in ("split", "frame_id", "step", "grid")]
            row = dict(model=name, split=split, count=len(subset), grid="native")
            for key in keys:
                values = [float(r[key]) for r in subset if r[key] != ""]
                row[key] = float(np.mean(values)) if values else None
                row[f"{key}_valid_count"] = len(values)
            aggregates.append(row)
        with (root/name/"metrics/pose_changes.csv").open() as handle:
            records = list(csv.DictReader(handle))
        train_ids = set(data.metadata["splits"]["training"])
        records = [r for r in records if int(r["frame_id"]) in train_ids]
        corrections[name] = {}
        for ax, key in zip(axes, ("translation_change_mm", "rotation_change_deg")):
            values = [float(r[key]) for r in records]
            corrections[name][key] = float(np.mean(values))
            ax.plot([int(r["frame_id"]) for r in records], values, label=name)
            ax.set(xlabel="Original training frame", ylabel=key)
            ax.legend()
    fig.suptitle("Pose changes relative to input; no pose ground truth")
    fig.savefig(output/"plots/pose_corrections.png", dpi=140)
    plt.close(fig)
    write_csv(output/"metrics/response_summary.csv", aggregates)
    write_json(output/"metrics/comparison_summary.json", dict(
        status="complete", variants=variants, response_metrics=aggregates, mean_training_pose_corrections=corrections,
        supervision="fixed traditional NLSTV response only; no B-mode losses or confidence",
        limitations=["single seed", "teacher response is not anatomical ground truth", "real pose accuracy unknown",
                     "validation and test poses fixed", "volume support approximate; outside support unobserved",
                     "voxel spacing is export sampling, not demonstrated spatial resolution"]))
