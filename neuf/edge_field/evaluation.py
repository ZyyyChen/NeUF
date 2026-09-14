from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

from .data import write_json


@torch.no_grad()
def query(model, xyz, progress=None, chunk=65536):
    flat = xyz.reshape(-1, 3)
    channels = 1 if model.response_only else 2
    return torch.cat([model(part, progress).cpu() for part in flat.split(chunk)]).numpy().reshape(*xyz.shape[:-1], channels)


@torch.no_grad()
def render(model, matrix, local, progress=1.):
    xyz = local @ matrix[:3, :3].T + matrix[:3, 3]
    return query(model, xyz, progress)


def image_metrics(prediction, target, mask):
    error = prediction - target
    mse = float(np.mean(error[mask] ** 2))
    _, ssim = structural_similarity(target, prediction, data_range=1., full=True)
    gp = np.hypot(*np.gradient(gaussian_filter(prediction, 1)))
    gt = np.hypot(*np.gradient(gaussian_filter(target, 1)))
    return dict(mse=mse, psnr=-10 * math.log10(max(mse, 1e-12)),
                ssim=float(ssim[mask].mean()), gradient_rms_ratio=float(
                    np.sqrt(np.mean(gp[mask] ** 2) / (np.mean(gt[mask] ** 2) + 1e-12))),
                contrast_std_ratio=float(prediction[mask].std() / (target[mask].std() + 1e-8)))


def select_profiles(image, mask, count=3):
    """只根据原图选候选，不依据任何模型输出；亮带不强行作为阶跃。"""
    gy, gx = np.gradient(gaussian_filter(image, 1.2))
    magnitude = np.hypot(gy, gx)
    allowed = mask.copy()
    allowed[:16] = allowed[-16:] = False
    allowed[:, :16] = allowed[:, -16:] = False
    score = magnitude * allowed * (magnitude == maximum_filter(magnitude, size=11))
    candidates = np.argsort(score.ravel())[-100:][::-1]
    selected = []
    for flat in candidates:
        row, col = np.unravel_index(flat, image.shape)
        if score[row, col] <= 0 or any(np.linalg.norm(np.array([row, col]) - p["center"]) < 30 for p in selected):
            continue
        normal = np.array([gy[row, col], gx[row, col]]) / (magnitude[row, col] + 1e-12)
        spec = dict(center=[int(row), int(col)], normal=normal.tolist())
        profile = sample_profile(image, spec)
        if measure_profile(profile)["valid"]:
            selected.append(spec)
        if len(selected) >= count:
            break
    return selected


def sample_profile(image, spec):
    t = np.linspace(-12, 12, 97)
    normal = np.array(spec["normal"])
    tangent = np.array([-normal[1], normal[0]])
    points = np.array(spec["center"])[:, None, None] + normal[:, None, None] * t[None, :, None]
    points = points + tangent[:, None, None] * np.arange(-2, 3)[None, None, :]
    return map_coordinates(image, points, order=1, mode="nearest").mean(-1)


def measure_profile(values):
    t = np.linspace(-12, 12, len(values))
    low, high = float(values[:16].mean()), float(values[-16:].mean())
    contrast = high - low
    invalid = dict(valid=False, width_10_90_px=None, center_px=None, contrast=contrast,
                   overshoot=None, reason="unstable_plateaus_or_step_fit")
    if abs(contrast) < .025:
        return invalid
    drift = max(np.ptp(np.polyval(np.polyfit(t[:16], values[:16], 1), t[:16])),
                np.ptp(np.polyval(np.polyfit(t[-16:], values[-16:], 1), t[-16:])))
    if drift / abs(contrast) > .3:
        return invalid
    sigmoid = lambda x, a, b, center, scale: a + b / (1 + np.exp(np.clip(-(x-center)/scale, -60, 60)))
    try:
        fit, _ = curve_fit(sigmoid, t, values, p0=(low, contrast, 0, 1),
                           bounds=([-1, -2, -5, .1], [2, 2, 5, 8]), maxfev=2000)
    except (ValueError, RuntimeError):
        return invalid
    residual = np.sqrt(np.mean((sigmoid(t, *fit) - values) ** 2)) / abs(contrast)
    if residual > .2:
        return invalid
    half_width = np.log(9) * fit[3]
    # 10–90% 过渡必须完全落在两侧平台窗口之间，禁止用缓坡外推边缘宽度。
    if fit[2] - half_width < -8 or fit[2] + half_width > 8:
        return dict(invalid, reason="transition_outside_observed_plateau_interval")
    lo, hi = sorted((low, high))
    over = max(0., float(values.max() - hi), float(lo - values.min())) / abs(contrast)
    return dict(valid=True, width_10_90_px=float(2 * np.log(9) * fit[3]), center_px=float(fit[2]),
                contrast=contrast, overshoot=over, reason=None)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, poses, data, output, step, progress, *, final=False, stride=4):
    output = Path(output)
    plot_dir = output / "plots" / f"step_{step:06d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    matrices = poses.matrices().detach()
    rows = []
    names = ("validation", "test") if final else ("validation",)
    mask = data.interior[::stride, ::stride].cpu().numpy()
    for split in names:
        for index in data.splits[split].cpu().tolist():
            prediction = render(model, matrices[index], data.local[::stride, ::stride], progress)[..., 0]
            target = data.images[index, ::stride, ::stride].cpu().numpy()
            rows.append(dict(split=split, frame_id=int(data.frame_ids[index]), step=step,
                             **image_metrics(prediction, target, mask)))
    write_csv(output / "metrics" / f"images_{step:06d}.csv", rows)
    saved = []
    training_preview = int(data.splits["training"][len(data.splits["training"])//2])
    for index in [*data.comparison, training_preview]:
        # 固定比较图使用原始分辨率，验证位姿始终不训练。
        prediction = render(model, matrices[index], data.local, progress)
        target = data.images[index].cpu().numpy()
        teacher = data.edges[index].cpu().numpy()
        valid = data.mask.cpu().numpy()
        frame = int(data.frame_ids[index])
        fig, axes = plt.subplots(1, 5, figsize=(19, 4), layout="constrained")
        panels = (target, prediction[..., 0], np.abs(prediction[..., 0] - target), teacher, prediction[..., 1])
        titles = ("Observed B-mode", "Reconstructed gray", "Absolute error [0,.15]", "NLSTV response", "Predicted response")
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(np.where(valid, panel, np.nan), cmap="gray", vmin=0, vmax=.15 if title.startswith("Absolute") else 1)
            ax.set_title(title)
            ax.axis("off")
        split_label = "training diagnostic" if index == training_preview else "validation (unfitted pose)"
        fig.suptitle(f"{output.name}; cerebral frame {frame}; {split_label}; step {step}; native pixels; fixed [0,1]")
        fig.savefig(plot_dir / f"frame_{frame:04d}.png", dpi=140)
        plt.close(fig)
        if final:
            np.savez_compressed(output / "predictions" / f"frame_{frame:04d}.npz",
                                gray=prediction[..., 0], edge=prediction[..., 1], observed=target,
                                teacher=teacher, mask=valid)
        saved.append(dict(frame_id=frame, split=split_label, step=step, **image_metrics(prediction[..., 0], target, data.interior.cpu().numpy())))
    write_csv(output / "metrics" / f"comparison_{step:06d}.csv", saved)
    return rows


def export_volume(model, data, output, spacing=.75, corrected_poses=None):
    """固定世界包围盒；float32 NPY 顺序 z/y/x，MHD origin/spacing 顺序 x/y/z。"""
    bounds = data.bounds.cpu().numpy()
    axes = [np.arange(bounds[0, j], bounds[1, j] + spacing * .5, spacing, dtype=np.float32) for j in range(3)]
    if np.prod([len(a) for a in axes]) > 20_000_000:
        raise ValueError("导出体超过 2000 万体素，请明确增大 spacing")
    # 统一以原始训练采集点估计支持区域，所有模型使用同一个显示掩膜。
    local = data.local[::8, ::8][data.interior[::8, ::8]].cpu().numpy()
    points = []
    matrices = data.initial if corrected_poses is None else corrected_poses.detach()
    for matrix in matrices[data.splits["training"]].cpu().numpy():
        points.append(local @ matrix[:3, :3].T + matrix[:3, 3])
    tree = cKDTree(np.concatenate(points))
    volume, support = [], []
    yy, xx = np.meshgrid(axes[1], axes[0], indexing="ij")
    for z in axes[2]:
        xyz = np.stack((xx, yy, np.full_like(xx, z)), -1)
        volume.append(query(model, torch.tensor(xyz, device=data.images.device))[..., 0])
        distance = tree.query(xyz.reshape(-1, 3), workers=2)[0].reshape(xx.shape)
        support.append(distance <= 1.5)
    volume, support = np.stack(volume), np.stack(support)
    prediction_dir = Path(output) / "predictions"
    basename = "edge_volume" if model.response_only else "volume_float"
    np.save(prediction_dir / f"{basename}.npy", volume)
    np.save(prediction_dir / "volume_support.npy", support)
    volume.astype("<f4").tofile(prediction_dir / f"{basename}.raw")
    (prediction_dir / f"{basename}.mhd").write_text(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
        + f"DimSize = {len(axes[0])} {len(axes[1])} {len(axes[2])}\n"
        + f"ElementSpacing = {spacing} {spacing} {spacing}\nOffset = {' '.join(map(str, bounds[0]))}\n"
        + f"ElementType = MET_FLOAT\nElementDataFile = {basename}.raw\n")
    write_json(prediction_dir / "volume_metadata.json", dict(spacing_xyz_mm=[spacing]*3, origin_xyz_mm=bounds[0].tolist(),
               shape_zyx=list(volume.shape), quantity="NLSTV structural response" if model.response_only else "B-mode intensity",
               support=f"within 1.5mm of {'corrected' if corrected_poses is not None else 'original'} training points subsampled every 8px; approximate",
               sagittal="world x constant; not claimed anatomical sagittal without orientation labels"))
    return volume, support


def compare(data, run_root, variants):
    output = Path(run_root) / "comparison"
    for folder in ("metrics", "plots", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    rows, all_profiles = [], {}
    for index in data.comparison:
        frame = int(data.frame_ids[index])
        original = data.images[index].cpu().numpy()
        candidates = select_profiles(original, data.interior.cpu().numpy())
        all_profiles[str(frame)] = candidates
        records = {name: np.load(Path(run_root) / name / "predictions" / f"frame_{frame:04d}.npz")["gray"] for name in variants}
        fig, axes = plt.subplots(2, 1+len(variants), figsize=(16, 8), layout="constrained")
        for col, (name, image) in enumerate({"Observed": original, **records}.items()):
            axes[0, col].imshow(np.where(data.mask.cpu().numpy(), image, np.nan), cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(name)
            # 延续根 notebook 的固定 ROI，不根据模型输出移动。
            c, r, w, h = data.metadata["comparison_roi_xywh"]
            axes[1, col].imshow(image[r:r+h, c:c+w], cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title(f"Fixed ROI x={c}, y={r}, {w}x{h}px")
            for ax in axes[:, col]:
                ax.axis("off")
        fig.suptitle(f"cerebral {frame}; equal configured field updates; fixed [0,1]")
        fig.savefig(output / "plots" / f"comparison_{frame:04d}.png", dpi=160)
        plt.close(fig)
        if not candidates:
            continue
        fig, axes = plt.subplots(1, len(candidates), figsize=(5*len(candidates), 4), squeeze=False, layout="constrained")
        for number, spec in enumerate(candidates):
            ref = measure_profile(sample_profile(original, spec))
            for name, image in {"Observed": original, **records}.items():
                profile = sample_profile(image, spec)
                measured = measure_profile(profile)
                rows.append(dict(frame_id=frame, candidate=number, model=name, **measured,
                                 width_ratio=measured["width_10_90_px"]/ref["width_10_90_px"] if measured["valid"] else None,
                                 center_shift_px=measured["center_px"]-ref["center_px"] if measured["valid"] else None,
                                 contrast_ratio=measured["contrast"]/ref["contrast"]))
                axes[0, number].plot(np.linspace(-12, 12, len(profile)), profile, label=name)
            axes[0, number].set(xlabel="Normal distance [px]", ylabel="Gray [0,1]", title=f"Candidate {number}")
            axes[0, number].legend()
        fig.savefig(output / "plots" / f"profiles_{frame:04d}.png", dpi=140)
        plt.close(fig)
    write_csv(output / "metrics" / "edge_profiles.csv", rows)
    write_json(output / "run_config" / "profile_selection.json", all_profiles)
    fig, axes = plt.subplots(1, len(variants), figsize=(5*len(variants), 5), squeeze=False, layout="constrained")
    for col, name in enumerate(variants):
        volume = np.load(Path(run_root) / name / "predictions" / "volume_float.npy", mmap_mode="r")
        support = np.load(Path(run_root) / name / "predictions" / "volume_support.npy", mmap_mode="r")
        x = volume.shape[2] // 2
        axes[0, col].imshow(np.where(support[:, :, x], volume[:, :, x], np.nan), cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0, col].set(title=f"{name}: fixed world-x plane", xlabel="y voxel", ylabel="z voxel")
    fig.savefig(output / "plots" / "sagittal_comparison.png", dpi=150)
    plt.close(fig)
    aggregates = []
    pose_summary = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    for name in variants:
        directory = Path(run_root) / name
        latest = sorted((directory / "metrics").glob("images_*.csv"))[-1]
        with latest.open() as handle:
            image_rows = list(csv.DictReader(handle))
        for split in ("validation", "test"):
            subset = [r for r in image_rows if r["split"] == split]
            aggregates.append(dict(model=name, split=split, count=len(subset),
                **{key: float(np.mean([float(r[key]) for r in subset]))
                   for key in ("mse", "psnr", "ssim", "gradient_rms_ratio", "contrast_std_ratio")}))
        with (directory / "metrics/pose_changes.csv").open() as handle:
            changes = list(csv.DictReader(handle))
        train_ids = set(data.metadata["splits"]["training"])
        changes = [r for r in changes if int(r["frame_id"]) in train_ids]
        for ax, key in zip(axes, ("translation_change_mm", "rotation_change_deg")):
            ax.plot([int(r["frame_id"]) for r in changes], [float(r[key]) for r in changes], label=name)
            ax.set(xlabel="Original training frame", ylabel=key)
            ax.legend()
        pose_summary[name] = {key: float(np.mean([float(r[key]) for r in changes]))
                              for key in ("translation_change_mm", "rotation_change_deg")}
    fig.suptitle("Pose corrections relative to input; not errors against ground truth")
    fig.savefig(output / "plots" / "pose_corrections.png", dpi=140)
    plt.close(fig)
    write_csv(output / "metrics" / "image_summary.csv", aggregates)
    write_json(output / "metrics" / "comparison_summary.json", dict(status="complete", variants=variants,
               image_metrics=aggregates, mean_training_pose_corrections=pose_summary,
               selected_profiles=sum(map(len, all_profiles.values())), quality_claim="尚未验证；需联合审查图像、几何和固定指标",
               limitations=["single seed", "observed B-mode is not clean ground truth", "real tracking poses have no independent ground truth",
                            "profile candidates are intensity edges, not annotated anatomy", "validation/test poses are fixed and never fitted"]))
