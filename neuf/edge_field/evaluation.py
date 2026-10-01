from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter, sobel
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity
from skimage.feature import canny

from .data import write_json
from .losses import blur, edge_preserve_weights, gradients


EDGE_PRESERVE_VARIANTS = (
    "HashObservedEdgeGatedSharp", "HashObservedEdgePreserve", "HashObservedEdgePreserve3D",
)


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


def edge_guided_metrics(prediction, target, predicted_edge, teacher_edge, mask):
    """描述边缘保真；teacher response 不是人工解剖边界真值。"""
    py, px = np.gradient(prediction)
    ty, tx = np.gradient(target)
    weight = 1 + 2 * teacher_edge
    gradient_error = np.abs(px - tx) + np.abs(py - ty)
    selected_prediction = predicted_edge[mask]
    selected_teacher = teacher_edge[mask]
    correlation = (
        float(np.corrcoef(selected_prediction, selected_teacher)[0, 1])
        if selected_prediction.std() > 1e-8 and selected_teacher.std() > 1e-8
        else 0.0
    )
    return {
        "weighted_gradient_mae": float(np.mean((weight * gradient_error)[mask])),
        "edge_response_mae": float(np.mean(np.abs(selected_prediction - selected_teacher))),
        "edge_response_correlation": correlation,
    }


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


def evaluate(model, poses, data, output, step, progress, *, final=False, stride=4,
             save_training_preview=False):
    output = Path(output)
    plot_dir = output / "plots" / f"step_{step:06d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    matrices = poses.matrices().detach()
    rows = []
    names = ("validation", "test") if final else ("validation",)
    mask = data.interior[::stride, ::stride].cpu().numpy()
    targets = data.images
    for split in names:
        for index in data.splits[split].cpu().tolist():
            prediction = render(model, matrices[index], data.local[::stride, ::stride], progress)
            target = targets[index, ::stride, ::stride].cpu().numpy()
            row = dict(split=split, frame_id=int(data.frame_ids[index]), step=step,
                       target_kind="observed", **image_metrics(prediction[..., 0], target, mask))
            teacher = data.edges[index, ::stride, ::stride].cpu().numpy()
            row.update(edge_guided_metrics(
                prediction[..., 0], target, prediction[..., 1], teacher, mask,
            ))
            rows.append(row)
    write_csv(output / "metrics" / f"images_{step:06d}.csv", rows)
    saved = []
    training_preview = int(data.splits["training"][len(data.splits["training"])//2])
    split_lookup = {int(index): name for name, indices in data.splits.items()
                    for index in indices.cpu().tolist()}
    for index in dict.fromkeys([*data.comparison, training_preview]):
        # 比较图使用原始分辨率；共享轨迹只从训练图拟合，再推算验证帧角度。
        prediction = render(model, matrices[index], data.local, progress)
        target = targets[index].cpu().numpy()
        observed = data.images[index].cpu().numpy()
        teacher = data.edges[index].cpu().numpy()
        valid = data.mask.cpu().numpy()
        frame = int(data.frame_ids[index])
        panels = (target, prediction[..., 0], np.abs(prediction[..., 0] - target), teacher, prediction[..., 1])
        titles = ("Observed B-mode", "Reconstructed gray", "Absolute error [0,.15]", "NLSTV response", "Predicted response")
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), layout="constrained")
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(np.where(valid, panel, np.nan), cmap="gray", vmin=0, vmax=.15 if title.startswith("Absolute") else 1)
            ax.set_title(title)
            ax.axis("off")
        split_name = split_lookup[index]
        pose_label = ("shared trajectory; image not fitted" if getattr(poses, "mode", "fixed") != "fixed"
                      else "unfitted pose")
        split_label = ("training diagnostic" if split_name == "training"
                       else f"{split_name} ({pose_label})")
        fig.suptitle(f"{output.name}; cerebral frame {frame}; {split_label}; step {step}; native pixels; fixed [0,1]")
        fig.savefig(plot_dir / f"frame_{frame:04d}.png", dpi=140)
        plt.close(fig)
        if final or (save_training_preview and index == training_preview):
            payload = dict(gray=prediction[..., 0], edge=prediction[..., 1], observed=observed,
                           target=target, teacher=teacher, mask=valid, target_kind="observed")
            if final:
                np.savez_compressed(output / "predictions" / f"frame_{frame:04d}.npz", **payload)
            if save_training_preview and index == training_preview:
                np.savez_compressed(
                    output / "predictions" / f"frame_{frame:04d}_step_{step:06d}.npz", **payload)
        saved_row = dict(frame_id=frame, split=split_label, step=step, target_kind="observed",
                         **image_metrics(prediction[..., 0], target, data.interior.cpu().numpy()))
        saved_row.update(edge_guided_metrics(
            prediction[..., 0], target, prediction[..., 1], teacher,
            data.interior.cpu().numpy(),
        ))
        saved.append(saved_row)
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
    volume, edge_volume, support = [], [], []
    yy, xx = np.meshgrid(axes[1], axes[0], indexing="ij")
    for z in axes[2]:
        xyz = np.stack((xx, yy, np.full_like(xx, z)), -1)
        prediction = query(model, torch.tensor(xyz, device=data.images.device))
        volume.append(prediction[..., 0])
        if not model.response_only:
            edge_volume.append(prediction[..., 1])
        distance = tree.query(xyz.reshape(-1, 3), workers=2)[0].reshape(xx.shape)
        support.append(distance <= 1.5)
    volume, support = np.stack(volume), np.stack(support)
    edge_volume = np.stack(edge_volume) if edge_volume else None
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
    if edge_volume is not None:
        np.save(prediction_dir / "edge_volume_float.npy", edge_volume)
        edge_volume.astype("<f4").tofile(prediction_dir / "edge_volume_float.raw")
        (prediction_dir / "edge_volume_float.mhd").write_text(
            "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
            + f"DimSize = {len(axes[0])} {len(axes[1])} {len(axes[2])}\n"
            + f"ElementSpacing = {spacing} {spacing} {spacing}\nOffset = {' '.join(map(str, bounds[0]))}\n"
            + "ElementType = MET_FLOAT\nElementDataFile = edge_volume_float.raw\n")
    write_json(prediction_dir / "volume_metadata.json", dict(spacing_xyz_mm=[spacing]*3, origin_xyz_mm=bounds[0].tolist(),
               shape_zyx=list(volume.shape), quantity="NLSTV structural response" if model.response_only else "B-mode intensity",
               support=f"within 1.5mm of {'corrected' if corrected_poses is not None else 'original'} training points subsampled every 8px; approximate",
               auxiliary_edge_volume="edge_volume_float.npy/.mhd/.raw; supervised only for edge-guided variants",
               sagittal="world x constant; not claimed anatomical sagittal without orientation labels"))
    return volume, support


def compare(data, run_root, variants):
    output = Path(run_root) / "comparison"
    for folder in ("metrics", "plots", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    rows, all_profiles = [], {}
    preserve_comparison = set(EDGE_PRESERVE_VARIANTS) <= set(variants)
    shared_trajectory = any(
        json.loads((Path(run_root) / name / "run_config/config.json").read_text()).get("pose_mode", "fixed") != "fixed"
        for name in variants
    )
    heldout_description = ("heldout images not fitted; angles inferred from shared training trajectory"
                           if shared_trajectory else "validation/test poses are fixed and never fitted")
    indices = list(data.comparison)
    if preserve_comparison:
        if data.frame_ids[indices].cpu().tolist() != [236, 226, 216, 206]:
            raise ValueError("分区保边对照必须保留固定验证帧及顺序 236/226/216/206")
        indices.insert(0, data.frame_ids.cpu().tolist().index(119))
    for index in indices:
        frame = int(data.frame_ids[index])
        observed = data.images[index].cpu().numpy()
        original = observed
        candidates = select_profiles(original, data.interior.cpu().numpy())
        all_profiles[str(frame)] = candidates
        records = {name: np.load(Path(run_root) / name / "predictions" / f"frame_{frame:04d}.npz")["gray"] for name in variants}
        panels = {"Observed": observed, **records}
        fig, axes = plt.subplots(3 if preserve_comparison else 2, len(panels),
                                 figsize=(4*len(panels), 12 if preserve_comparison else 8), layout="constrained")
        for col, (name, image) in enumerate(panels.items()):
            axes[0, col].imshow(np.where(data.mask.cpu().numpy(), image, np.nan), cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(name)
            # 延续根 notebook 的固定 ROI，不根据模型输出移动。
            c, r, w, h = ([350, 190, 250, 270] if preserve_comparison and frame == 119
                          else data.metadata["comparison_roi_xywh"])
            axes[1, col].imshow(image[r:r+h, c:c+w], cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title(f"Fixed ROI x={c}, y={r}, {w}x{h}px")
            if preserve_comparison:
                error = np.abs(image - observed)
                axes[2, col].imshow(error[r:r+h, c:c+w], cmap="magma", vmin=0, vmax=.15)
                axes[2, col].set_title("ROI absolute error [0,.15]")
            for ax in axes[:, col]:
                ax.axis("off")
        label = "training diagnostic" if frame == 119 else f"validation; {heldout_description}"
        caption = f"; {label}; native acquisition plane (coronal view)" if preserve_comparison else ""
        fig.suptitle(f"cerebral {frame}; equal configured field updates; fixed [0,1]{caption}")
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
    profile_summary = []
    for name in ("Observed", *variants):
        subset = [row for row in rows if row["model"] == name]
        valid = [row for row in subset if row["valid"]]
        profile_summary.append(dict(
            model=name, count=len(subset), valid_count=len(valid),
            mean_width_ratio=float(np.mean([row["width_ratio"] for row in valid])) if valid else None,
            mean_abs_center_shift_px=float(np.mean([abs(row["center_shift_px"]) for row in valid])) if valid else None,
            mean_contrast_ratio=float(np.mean([row["contrast_ratio"] for row in subset])) if subset else None,
            mean_overshoot=float(np.mean([row["overshoot"] for row in valid])) if valid else None,
        ))
    write_csv(output / "metrics" / "profile_summary.csv", profile_summary)
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
            metric_keys = [
                "mse", "psnr", "ssim", "gradient_rms_ratio", "contrast_std_ratio",
                "weighted_gradient_mae", "edge_response_mae", "edge_response_correlation",
            ]
            aggregates.append(dict(model=name, split=split, count=len(subset), target_kind="observed",
                **{key: float(np.mean([float(r[key]) for r in subset])) for key in metric_keys}))
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
               target_kind="observed",
               image_metrics=aggregates, mean_training_pose_corrections=pose_summary,
               selected_profiles=sum(map(len, all_profiles.values())), quality_claim="尚未验证；需联合审查图像、几何和固定指标",
               limitations=["single seed", "observed B-mode is not clean ground truth", "real tracking poses have no independent ground truth",
                            "profile candidates are intensity edges, not annotated anatomy", heldout_description]))
    if preserve_comparison:
        compare_edge_preserve(data, run_root, EDGE_PRESERVE_VARIANTS, rows)


def _region_statistics(image, mask):
    """高频下降同时对照灰度均值和方差；固定 sigma=1 原生像素，不判定解剖降噪。"""
    count = int(mask.sum())
    if not count:
        return dict(pixel_count=0, mean=None, variance=None, high_frequency_energy=None)
    values = image[mask]
    high_frequency = image - gaussian_filter(image, 1.)
    return dict(pixel_count=count, mean=float(values.mean()), variance=float(values.var()),
                high_frequency_energy=float(np.mean(high_frequency[mask] ** 2)))


def _preserve_volume_comparison(run_root, variants, output):
    """共用原始观测支持和毫米网格；空间差分仅描述变化，真实结构也会产生差分。"""
    root = Path(run_root)
    reference = root / variants[0] / "predictions"
    metadata = json.loads((reference / "volume_metadata.json").read_text())
    support = np.load(reference / "volume_support.npy").astype(bool)
    spacing = np.asarray(metadata["spacing_xyz_mm"])
    origin = np.asarray(metadata["origin_xyz_mm"])
    x_indices = [int(round((support.shape[2] - 1) * fraction)) for fraction in (.25, .5, .75)]
    rows = []
    fig, axes = plt.subplots(len(x_indices), len(variants), figsize=(5*len(variants), 12),
                             squeeze=False, layout="constrained")
    for column, name in enumerate(variants):
        directory = root / name / "predictions"
        current_metadata = json.loads((directory / "volume_metadata.json").read_text())
        for key in ("spacing_xyz_mm", "origin_xyz_mm", "shape_zyx"):
            if current_metadata[key] != metadata[key]:
                raise ValueError(f"分区保边比较体网格不同: {name}, {key}")
        if not np.array_equal(np.load(directory / "volume_support.npy"), support):
            raise ValueError(f"分区保边比较必须使用相同的原始观测支持域: {name}")
        volume = np.load(directory / "volume_float.npy", mmap_mode="r")
        values = volume[support]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"分区保边体数据的固定支持域为空或包含非有限数值: {name}")
        for axis, direction in enumerate(("z", "y", "x")):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            paired = support[tuple(lower)] & support[tuple(upper)]
            distance = float(spacing[2-axis])
            differences = np.diff(volume, axis=axis)[paired] / distance
            triple = [slice(None)] * 3
            triple[axis] = slice(2, None)
            middle = [slice(None)] * 3
            middle[axis] = slice(1, -1)
            lower[axis] = slice(None, -2)
            triplets = support[tuple(lower)] & support[tuple(middle)] & support[tuple(triple)]
            curvature = np.diff(volume, n=2, axis=axis)[triplets] / distance**2
            rows.append(dict(model=name, direction=direction, support_voxels=int(support.sum()),
                             mean=float(values.mean()), variance=float(values.var()),
                             paired_voxels=int(paired.sum()), triplet_voxels=int(triplets.sum()),
                             mean_abs_gradient_per_mm=float(np.mean(np.abs(differences))) if differences.size else None,
                             rms_gradient_per_mm=float(np.sqrt(np.mean(differences**2))) if differences.size else None,
                             mean_abs_curvature_per_mm2=float(np.mean(np.abs(curvature))) if curvature.size else None))
        for row, x in enumerate(x_indices):
            axes[row, column].imshow(np.where(support[:, :, x], volume[:, :, x], np.nan),
                                     cmap="gray", vmin=0, vmax=1, origin="lower",
                                     extent=(origin[1]-.5*spacing[1], origin[1]+(support.shape[1]-.5)*spacing[1],
                                             origin[2]-.5*spacing[2], origin[2]+(support.shape[0]-.5)*spacing[2]))
            axes[row, column].set(title=f"{name}\nworld x={origin[0]+x*spacing[0]:.3f} mm", xlabel="world y [mm]", ylabel="world z [mm]")
    fig.suptitle("Fixed world-x planes; shared observation support; [0,1]; anatomical sagittal orientation unverified")
    fig.savefig(output / "plots" / "world_x_preserve_comparison.png", dpi=150)
    plt.close(fig)
    write_csv(output / "metrics" / "preserve_volume_continuity.csv", rows)
    return dict(x_indices=x_indices, world_x_mm=[float(origin[0]+x*spacing[0]) for x in x_indices],
                spacing_xyz_mm=spacing.tolist(), origin_xyz_mm=origin.tolist(), shape_zyx=list(support.shape),
                support=metadata["support"], metrics=rows)


def compare_edge_preserve(data, run_root, variants=EDGE_PRESERVE_VARIANTS, profile_rows=None):
    """补充分区保边对照；只读已有预测，保留原图基准和原有比较文件接口。"""
    output = Path(run_root) / "comparison"
    for folder in ("metrics", "plots", "run_config", "predictions"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    if profile_rows is None:
        with (output / "metrics" / "edge_profiles.csv").open() as handle:
            profile_rows = list(csv.DictReader(handle))
        for row in profile_rows:
            row["valid"] = row["valid"] == "True"
            for key in ("width_ratio", "center_shift_px", "contrast_ratio", "overshoot"):
                row[key] = float(row[key]) if row[key] else None
    frames = [119, 236, 226, 216, 206]
    indices = data.frame_ids.cpu().tolist()
    interior = data.interior.cpu().numpy()
    rows, selections = [], {}
    fig, axes = plt.subplots(1, len(frames), figsize=(4*len(frames), 4), layout="constrained")
    for column, frame in enumerate(frames):
        index = indices.index(frame)
        observed = data.images[index].cpu().numpy()
        teacher = data.edges[index].cpu().numpy()
        roi = [350, 190, 250, 270] if frame == 119 else data.metadata["comparison_roi_xywh"]
        x, y, width, height = roi
        selected = np.zeros_like(interior)
        selected[y:y+height, x:x+width] = True
        # 区域只由原始 teacher 和固定 ROI 决定，绝不按模型预测移动或筛选。
        background = interior & selected & (teacher <= .15)
        np.savez_compressed(output / "predictions" / f"background_mask_{frame:04d}.npz", mask=background)
        selections[str(frame)] = dict(roi_xywh=roi, background_pixels=int(background.sum()),
                                       selection="interior AND fixed ROI AND original NLSTV response <= 0.15")
        crop = np.s_[y:y+height, x:x+width]
        axes[column].imshow(observed[crop], cmap="gray", vmin=0, vmax=1)
        axes[column].imshow(np.ma.masked_where(~background[crop], background[crop]),
                            cmap="autumn", vmin=0, vmax=1, alpha=.25)
        axes[column].set_title(f"Frame {frame}; fixed ROI\nlow-response pixels: {int(background.sum())}")
        axes[column].axis("off")
        records = {"Observed": observed}
        for name in variants:
            with np.load(Path(run_root) / name / "predictions" / f"frame_{frame:04d}.npz") as prediction:
                if not np.array_equal(prediction["observed"], observed):
                    raise ValueError(f"分区保边对照原图不一致: {name}, frame {frame}")
                records[name] = prediction["gray"].copy()
        reference = _region_statistics(observed, background)
        for name, image in records.items():
            stats = _region_statistics(image, background)
            rows.append(dict(frame_id=frame, split="training diagnostic" if frame == 119 else "validation",
                             model=name, grid="native", **image_metrics(image, observed, interior),
                             **{f"background_{key}": value for key, value in stats.items()},
                             background_mean_shift=stats["mean"]-reference["mean"] if stats["pixel_count"] else None,
                             background_variance_ratio=stats["variance"]/max(reference["variance"], 1e-12) if stats["pixel_count"] else None,
                             background_high_frequency_ratio=stats["high_frequency_energy"]/max(reference["high_frequency_energy"], 1e-12) if stats["pixel_count"] else None))
    fig.suptitle("Fixed source-only low-response regions (yellow overlay); not clean anatomical background")
    fig.savefig(output / "plots" / "preserve_background_regions.png", dpi=150)
    plt.close(fig)
    write_csv(output / "metrics" / "preserve_image_metrics.csv", rows)
    names = ("Observed", *variants)
    valid_sets = [{(str(row["frame_id"]), str(row["candidate"])) for row in profile_rows
                   if row["model"] == name and row["valid"]} for name in names]
    common = set.intersection(*valid_sets)
    summaries = []
    for name in names:
        for split, selected_frames in (("training diagnostic", {119}), ("validation", set(frames[1:]))):
            subset = [row for row in profile_rows if row["model"] == name and int(row["frame_id"]) in selected_frames]
            paired = [row for row in subset if (str(row["frame_id"]), str(row["candidate"])) in common]
            summaries.append(dict(model=name, split=split, selected_count=len(subset),
                                  valid_count=sum(row["valid"] for row in subset), common_valid_count=len(paired),
                                  mean_width_ratio=float(np.mean([row["width_ratio"] for row in paired])) if paired else None,
                                  mean_abs_center_shift_px=float(np.mean([abs(row["center_shift_px"]) for row in paired])) if paired else None,
                                  mean_contrast_ratio=float(np.mean([row["contrast_ratio"] for row in paired])) if paired else None,
                                  mean_overshoot=float(np.mean([row["overshoot"] for row in paired])) if paired else None))
    write_csv(output / "metrics" / "preserve_profile_summary.csv", summaries)
    volume = _preserve_volume_comparison(run_root, variants, output)
    write_json(output / "run_config" / "preserve_selection.json", dict(frames=frames, regions=selections,
               background_definition="fixed low-NLSTV-response region; not verified homogeneous anatomy",
               high_frequency_definition="mean((image - Gaussian(image, sigma=1 native pixel))^2) in fixed region"))
    write_json(output / "metrics" / "edge_preserve_summary.json", dict(
        status="complete", variants=list(variants), frames=frames, image_metrics=rows, profile_summary=summaries,
        volume=volume, quality_claim="尚未验证；需联合审查原图、边缘位置/对比和三维结构，不能以更平滑自动判优",
        figures=[item for frame in frames for item in (f"comparison_{frame:04d}.png", f"profiles_{frame:04d}.png")
                 if (output / "plots" / item).exists()] + ["preserve_background_regions.png", "world_x_preserve_comparison.png"],
        limitations=["observed B-mode is not clean anatomical ground truth",
                     "low teacher response can still contain weak anatomical structure or speckle",
                     "lower high-frequency energy requires preserved mean, variance, edge position and contrast",
                     "spatial differences include real structure; lower continuity statistics alone do not establish improvement",
                     "fixed world-x reslices are not verified anatomical sagittal planes; exported spacing is not measured resolution",
                     "frame 119 is a training diagnostic; validation planes retain fixed input poses; one seed"]))


def _fixed_profile_selection(selection_path, candidate_keys=None):
    """保留历史候选编号与顺序；显式给出的子集不按本次结果重新筛选。"""
    source = Path(selection_path).read_bytes()
    selection = json.loads(source)
    by_frame = ({int(selection["frame_id"]): selection["candidates"]} if "frame_id" in selection
                else {int(frame): candidates for frame, candidates in selection.items()})
    original_keys = [(frame, number) for frame, candidates in by_frame.items()
                     for number in range(len(candidates))]
    keys = original_keys if candidate_keys is None else [tuple(map(int, key)) for key in candidate_keys]
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(original_keys):
        raise ValueError("固定剖面键必须非空、不重复且全部来自历史 selection")
    manifest = dict(source_path=str(Path(selection_path).resolve()),
                    source_sha256=hashlib.sha256(source).hexdigest(), original_selection=selection,
                    original_candidate_count=len(original_keys), selected_candidate_keys=keys,
                    excluded_candidate_keys=[key for key in original_keys if key not in keys],
                    selection_rule="all historical candidates" if candidate_keys is None else "explicit frozen historical subset; no reselection using current predictions")
    return by_frame, keys, manifest


def _profile_band_points(spec):
    """与 sample_profile 完全相同的 97×5 个双线性采样坐标，顺序为 row/col。"""
    normal = np.asarray(spec["normal"])
    tangent = np.array([-normal[1], normal[0]])
    return (np.asarray(spec["center"])[:, None, None]
            + normal[:, None, None] * np.linspace(-12, 12, 97)[None, :, None]
            + tangent[:, None, None] * np.arange(-2, 3)[None, None, :])


def _profile_band_valid(mask, points):
    inside = ((points[0] >= 0) & (points[0] <= mask.shape[0]-1)
              & (points[1] >= 0) & (points[1] <= mask.shape[1]-1))
    coverage = map_coordinates(mask.astype(np.float32), points, order=1, mode="constant", cval=0.)
    return bool(inside.all() and (coverage >= 1-1e-6).all())


def _draw_fixed_profiles(axis, roi, candidates=()):
    """坐标保持原生像素；编号沿用历史文件，不把 ROI 边界当采样有效域。"""
    x, y, width, height = roi
    axis.add_patch(Rectangle((x-.5, y-.5), width, height, fill=False, edgecolor="cyan", linewidth=1.2))
    for number, spec in candidates:
        center, normal = np.asarray(spec["center"]), np.asarray(spec["normal"])
        ends = center[:, None] + normal[:, None] * np.array([-12., 12.])
        color = plt.get_cmap("tab10")(number % 10)
        axis.plot(ends[1], ends[0], color=color, linewidth=1.5)
        axis.scatter(center[1], center[0], color=[color], s=12)
        axis.text(center[1]+3, center[0]-3, str(number), color=color, fontsize=9,
                  bbox=dict(facecolor="black", alpha=.5, pad=.5, edgecolor="none"))


def diagnose_preserve_profiles(data, source_root, selection_path, output, *, prefix="internal",
                               candidate_keys=None, roi_xywh=None):
    """重测固定内部剖面；失败样本完整保留，聚合仅用完整有效采样带及共同有效拟合。"""
    source_root, output = Path(source_root), Path(output)
    for folder in ("metrics", "plots", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    by_frame, keys, manifest = _fixed_profile_selection(selection_path, candidate_keys)
    frame_ids = list(dict.fromkeys(frame for frame, _ in keys))
    ids = data.frame_ids.cpu().tolist()
    mask, interior = data.mask.cpu().numpy(), data.interior.cpu().numpy()
    training_ids = set(data.frame_ids[data.splits["training"]].cpu().tolist())
    names = ("Observed", *EDGE_PRESERVE_VARIANTS)
    labels = ("Observed", "Sharp baseline", "Preserve", "Preserve + 3D")
    rows, figures, rois = [], [], {}
    for frame in frame_ids:
        index = ids.index(frame)
        observed = data.images[index].cpu().numpy()
        records = {"Observed": observed}
        for name in EDGE_PRESERVE_VARIANTS:
            with np.load(source_root / name / "predictions" / f"frame_{frame:04d}.npz") as prediction:
                if not np.array_equal(prediction["observed"], observed):
                    raise ValueError(f"内部剖面对照原图不一致: {name}, frame {frame}")
                records[name] = prediction["gray"].copy()
        roi = list(roi_xywh if roi_xywh is not None else
                   manifest["original_selection"].get("roi_xywh", data.metadata["comparison_roi_xywh"]))
        rois[str(frame)] = roi
        x, y, width, height = roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y:y+height, x:x+width] = True
        candidates = [(number, by_frame[frame][number]) for key_frame, number in keys if key_frame == frame]
        split = "training diagnostic" if frame in training_ids else "validation"
        columns = min(4, len(candidates))
        fig, axes = plt.subplots(math.ceil(len(candidates)/columns), columns,
                                 figsize=(5*columns, 4*math.ceil(len(candidates)/columns)),
                                 squeeze=False, sharex=True, sharey=True, layout="constrained")
        for panel, (number, spec) in enumerate(candidates):
            points = _profile_band_points(spec)
            band_valid = _profile_band_valid(interior, points)
            sector_valid = _profile_band_valid(mask, points)
            roi_valid = _profile_band_valid(roi_mask, points)
            profiles = {name: sample_profile(image, spec) for name, image in records.items()}
            measured = {name: measure_profile(profile) for name, profile in profiles.items()}
            common = band_valid and all(result["valid"] for result in measured.values())
            reference = measured["Observed"]
            axis = axes.flat[panel]
            for name, label in zip(names, labels):
                result = measured[name]
                paired_fit = result["valid"] and reference["valid"]
                rows.append(dict(frame_id=frame, split=split, candidate=number, model=name,
                                 center_row=int(spec["center"][0]), center_col=int(spec["center"][1]),
                                 sampleband_valid=band_valid, sampleband_in_sector=sector_valid,
                                 sampleband_in_roi=roi_valid,
                                 sampleband_reason=None if band_valid else "sample band or bilinear support leaves training interior",
                                 fit_valid=result["valid"], common_valid=common, **result,
                                 width_ratio=result["width_10_90_px"]/reference["width_10_90_px"] if paired_fit else None,
                                 center_shift_px=result["center_px"]-reference["center_px"] if paired_fit else None,
                                 contrast_ratio=result["contrast"]/reference["contrast"] if abs(reference["contrast"]) > 1e-12 else None))
                axis.plot(np.linspace(-12, 12, 97), profiles[name], label=label)
            axis.set(xlabel="Normal distance [native px]", ylabel="Gray [0,1]", ylim=(0, 1),
                     title=f"Candidate {number}; band={band_valid}; common fit={common}")
        for axis in list(axes.flat)[len(candidates):]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Frame {frame}; {split}; fixed historical profiles; invalid fits retained")
        filename = f"{prefix}_profiles_{frame:04d}.png"
        fig.savefig(output / "plots" / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
        for axis in axes:
            axis.imshow(np.where(mask, observed, np.nan), cmap="gray", vmin=0, vmax=1)
            _draw_fixed_profiles(axis, roi, candidates)
            axis.set(xlabel="Column [native px]", ylabel="Row [native px]")
        axes[0].set_title("Observed; fixed centers and +/-12 px normals")
        axes[1].set(xlim=(x-.5, x+width-.5), ylim=(y+height-.5, y-.5), title=f"Historical ROI {roi}")
        fig.suptitle(f"Frame {frame}; {split}; fixed [0,1]")
        filename = f"{prefix}_profile_locations_{frame:04d}.png"
        fig.savefig(output / "plots" / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)

        fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 5), layout="constrained")
        for axis, name, label in zip(axes, names, labels):
            axis.imshow(records[name][y:y+height, x:x+width], cmap="gray", vmin=0, vmax=1,
                        extent=(x-.5, x+width-.5, y+height-.5, y-.5))
            axis.set(title=label, xlabel="Column [native px]", ylabel="Row [native px]")
        fig.suptitle(f"Frame {frame}; {split}; fixed historical ROI {roi}; [0,1]")
        filename = f"{prefix}_roi_comparison_{frame:04d}.png"
        fig.savefig(output / "plots" / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)
    summaries = []
    for name in names:
        subset = [row for row in rows if row["model"] == name]
        common = [row for row in subset if row["common_valid"]]
        widths = [row["width_ratio"] for row in common]
        summaries.append(dict(model=name, candidate_count=len(subset),
                              sampleband_valid_count=sum(row["sampleband_valid"] for row in subset),
                              fit_valid_count=sum(row["fit_valid"] for row in subset), common_valid_count=len(common),
                              mean_width_px=float(np.mean([row["width_10_90_px"] for row in common])) if common else None,
                              median_width_px=float(np.median([row["width_10_90_px"] for row in common])) if common else None,
                              mean_width_ratio=float(np.mean(widths)) if common else None,
                              median_width_ratio=float(np.median(widths)) if common else None,
                              mean_abs_width_ratio_deviation=float(np.mean(np.abs(np.asarray(widths)-1))) if common else None,
                              mean_abs_center_shift_px=float(np.mean([abs(row["center_shift_px"]) for row in common])) if common else None,
                              mean_contrast_ratio=float(np.mean([row["contrast_ratio"] for row in common])) if common else None,
                              mean_abs_contrast_ratio_deviation=float(np.mean([abs(row["contrast_ratio"]-1) for row in common])) if common else None,
                              mean_overshoot=float(np.mean([row["overshoot"] for row in common])) if common else None))
    write_csv(output / "metrics" / f"{prefix}_profiles.csv", rows)
    write_csv(output / "metrics" / f"{prefix}_profile_summary.csv", summaries)
    manifest.update(roi_by_frame=rois, source_root=str(source_root.resolve()),
                    sampling="97 normal samples in [-12,12] px; mean over 5 tangent offsets [-2,-1,0,1,2]; bilinear",
                    complete_band="all bilinear sample coverage in data.interior; sector/ROI membership also reported; ROI membership not required",
                    aggregation="same candidates with complete interior band and valid Observed plus all three model fits; every failure retained in CSV")
    write_json(output / "run_config" / f"{prefix}_profile_selection.json", manifest)
    return dict(status="complete", prefix=prefix, frame_ids=frame_ids, selection=manifest,
                model_summaries=summaries, candidate_rows=rows, figures=figures,
                limitations=["fixed historical candidates are not an unbiased sample of anatomical boundaries",
                             "frame 119 is a training diagnostic; observed B-mode is not clean anatomical truth",
                             "mean ratio near 1 can hide mixed over/undersmoothing; inspect mean absolute deviation and individual profiles"])


@torch.no_grad()
def diagnose_preserve_weights(data, source_root, output, selection_path, *, roi_xywh=(350, 190, 250, 270),
                              extra_selection_path=None, extra_candidate_keys=None):
    """复用训练 softweight；全图资格图不包含随机 patch 边界及实际采样频率。"""
    source_root, output = Path(source_root), Path(output)
    for folder in ("metrics", "plots", "predictions", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    by_frame, keys, selection = _fixed_profile_selection(selection_path)
    extra_selection = None
    if extra_selection_path is not None:
        extra_frames, extra_keys, extra_selection = _fixed_profile_selection(extra_selection_path, extra_candidate_keys)
        if set(by_frame) & set(extra_frames):
            raise ValueError("两个历史剖面来源的帧不能重叠，以免候选编号歧义")
        by_frame.update(extra_frames)
        keys.extend(extra_keys)
    elif extra_candidate_keys is not None:
        raise ValueError("extra_candidate_keys 必须同时提供 extra_selection_path")
    ids = data.frame_ids.cpu().tolist()
    input_mask = data.interior
    mask = input_mask.cpu().numpy()
    rows, center_rows, figures = [], [], []
    x, y, width, height = roi_xywh
    roi_mask = np.zeros_like(mask)
    roi_mask[y:y+height, x:x+width] = True
    for frame in (119, 236, 226, 216, 206):
        index = ids.index(frame)
        target = data.images[index][None, None]
        response = data.edges[index][None, None]
        observed = target[0, 0].cpu().numpy()
        teacher = response[0, 0].cpu().numpy()
        with np.load(source_root / EDGE_PRESERVE_VARIANTS[0] / "predictions" / f"frame_{frame:04d}.npz") as prediction:
            if not np.array_equal(prediction["observed"], observed):
                raise ValueError(f"权重诊断原图与正式实验不一致: frame {frame}")
            if not np.array_equal(prediction["teacher"], teacher):
                raise ValueError(f"权重诊断 teacher 与正式实验不一致: frame {frame}")
        # 返回的权重对齐中心差分网格；外圈一像素没有梯度中心，补零回原生尺寸。
        weight = torch.nn.functional.pad(edge_preserve_weights(target, response, input_mask[None, None]), (1, 1, 1, 1))[0, 0].cpu().numpy()
        dx, dy = gradients(blur(target, 1))
        magnitude = torch.nn.functional.pad((dx.square()+dy.square()).sqrt(), (1, 1, 1, 1))[0, 0].cpu().numpy()
        response_mask = mask & (teacher > .15)
        selected = weight > 0
        rejected = response_mask & ~selected
        candidates = [(number, by_frame[frame][number]) for key_frame, number in keys if key_frame == frame]
        current_centers = []
        for number, spec in candidates:
            row, col = map(int, spec["center"])
            record = dict(frame_id=frame, candidate=number, center_row=row, center_col=col,
                          response=float(teacher[row, col]), blurred_gradient=float(magnitude[row, col]),
                          weight=float(weight[row, col]), selected=bool(selected[row, col]),
                          within_input_mask=bool(mask[row, col]), within_roi=bool(roi_mask[row, col]))
            current_centers.append(record)
            center_rows.append(record)
        row = dict(frame_id=frame, split="training diagnostic" if frame == 119 else "validation eligibility only",
                   profile_center_count=len(current_centers), profile_center_selected_count=sum(item["selected"] for item in current_centers),
                   profile_center_selected_fraction=sum(item["selected"] for item in current_centers)/len(current_centers) if current_centers else None,
                   profile_center_weights=json.dumps(current_centers, ensure_ascii=False))
        for label, region in (("full", mask), ("roi", mask & roi_mask)):
            count = int(region.sum())
            response_count = int((response_mask & region).sum())
            selected_count = int((selected & region).sum())
            row.update({f"{label}_pixels": count, f"{label}_response_pixels": response_count,
                        f"{label}_selected_pixels": selected_count,
                        f"{label}_response_fraction": response_count/count if count else None,
                        f"{label}_selected_fraction": selected_count/count if count else None,
                        f"{label}_selected_of_response": selected_count/response_count if response_count else None,
                        f"{label}_mean_softweight": float(weight[region].mean()) if count else None})
        rows.append(row)
        np.savez_compressed(output / "predictions" / f"edge_weights_{frame:04d}.npz", weight=weight,
                            selected=selected, rejected_response=rejected, response_mask=response_mask,
                            teacher=teacher, blurred_gradient=magnitude, input_mask=mask, roi_mask=roi_mask)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        panels = (weight, np.where(mask, teacher, 0), selected.astype(np.uint8)+2*rejected)
        titles = ("Observed + full-image softweight", "Original NLSTV response", "Response > .15: accepted green / rejected red")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            for view in range(2):
                axis = axes[view, column]
                axis.imshow(np.where(mask, observed, np.nan), cmap="gray", vmin=0, vmax=1)
                shown = axis.imshow(np.ma.masked_where(panel <= 0, panel),
                                    cmap=ListedColormap(["limegreen", "orangered"]) if column == 2 else "magma",
                                    vmin=1 if column == 2 else 0, vmax=2 if column == 2 else 1, alpha=.8)
                _draw_fixed_profiles(axis, roi_xywh, candidates)
                axis.set(title=title if view == 0 else f"Historical internal ROI {list(roi_xywh)}",
                         xlabel="Column [native px]", ylabel="Row [native px]")
                if view:
                    axis.set(xlim=(x-.5, x+width-.5), ylim=(y+height-.5, y-.5))
            if column < 2:
                fig.colorbar(shown, ax=axes[:, column], label="Fixed NLSTV response / softweight [0,1]", shrink=.7)
        fig.suptitle(f"Frame {frame}; {row['split']}; full-image eligibility; random-patch boundary effects not modelled")
        filename = f"edge_weights_{frame:04d}.png"
        fig.savefig(output / "plots" / filename, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)
    write_csv(output / "metrics" / "edge_weight_coverage.csv", rows)
    write_csv(output / "metrics" / "edge_weight_profile_centers.csv", center_rows)
    manifest = dict(source_root=str(source_root.resolve()), profile_selection=selection, roi_xywh=list(roi_xywh),
                    extra_profile_selection=extra_selection,
                    formula="E * (E > 0.15) * (norm(central_gradient(Gaussian(observed,sigma=1))) > 0.01) * (avg_pool9(data.interior) > 0.999)",
                    implementation="losses.edge_preserve_weights; same helper used by training gradient loss",
                    input_mask="data.interior, matching data.patches()['mask']; only native outer ring padded with zero",
                    coverage_denominator="full/ROI fractions use data.interior pixels; selected_of_response uses E>0.15 within the same domain",
                    limitations=["full-image eligibility omits random-patch boundary exclusion and training sampling frequency",
                                 "validation-frame maps diagnose eligibility only; validation frames were never optimized",
                                 "high response and gradient thresholds select intensity structure, not annotated anatomy"])
    write_json(output / "run_config" / "edge_weight_definition.json", manifest)
    return dict(status="complete", definition=manifest, coverage=rows, profile_centers=center_rows, figures=figures)


def _sagittal_metric_mask(mask, pitch_mm):
    # 完整保留 Gaussian 的 4σ 支持及 SSIM 半径，排除渲染域外补零的影响。
    radius = int(math.ceil(4 * .5 / pitch_mm)) + 3
    return minimum_filter(mask, size=2*radius+1, mode="constant", cval=0).astype(bool)


def _sagittal_structure_metrics(prediction, reference, mask, pitch_mm):
    """两侧同样按 0.5 mm 平滑，跨采集平面仅报告结构相关和辅助 SSIM。"""
    valid = _sagittal_metric_mask(mask, pitch_mm)
    reference_smooth = gaussian_filter(reference, .5/pitch_mm)
    prediction_smooth = gaussian_filter(prediction, .5/pitch_mm)
    reference_edge = np.hypot(*np.gradient(reference_smooth))
    prediction_edge = np.hypot(*np.gradient(prediction_smooth))
    metrics = dict(pixel_count=int(mask.sum()), metric_pixel_count=int(valid.sum()), edge_ncc=None, ssim=None)
    if valid.any():
        p, r = prediction_edge[valid], reference_edge[valid]
        metrics["edge_ncc"] = float(np.corrcoef(p, r)[0, 1]) if min(p.std(), r.std()) > 1e-8 else 0.
        _, similarity = structural_similarity(reference_smooth, prediction_smooth, data_range=1., full=True)
        metrics["ssim"] = float(similarity[valid].mean())
    scale = max(float(np.quantile(reference_edge[mask], .99)), 1e-6) if mask.any() else 1.
    return metrics, np.abs(prediction_edge-reference_edge)/scale


def _sagittal_support(lines, sagittal, data, world_yz):
    """沿训练帧交线的有效深度采样估计支持；距离阈值与已有体导出一致。"""
    training_ids = data.frame_ids[data.splits["training"]].cpu().numpy()
    source_valid = sagittal.source_support.detach().cpu().numpy().astype(bool)
    selected = source_valid & np.isin(lines["frame_ids"], training_ids)[:, None]
    native_xy = lines["xy"][selected]
    inverse = np.linalg.inv(sagittal.linear.detach().cpu().numpy())
    points = (native_xy-sagittal.offset.detach().cpu().numpy()) @ inverse
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        return np.zeros(world_yz.shape[:-1], dtype=bool)
    distance = cKDTree(points).query(world_yz.reshape(-1, 2), workers=2)[0]
    return (distance <= 1.5).reshape(world_yz.shape[:-1])


def _sagittal_grid(data, sagittal):
    """显示网格由原生参考图及冻结标定唯一决定，不依据训练输出选 ROI。"""
    if not hasattr(sagittal, "_evaluation_grid"):
        reference = sagittal.reference.detach().cpu().numpy()
        rows, columns = np.indices(reference.shape)
        native_xy = np.stack((columns, rows), -1)
        # 标定使用行向量：native_xy = world_yz @ linear + offset。
        world_yz = ((native_xy-sagittal.offset.detach().cpu().numpy())
                    @ np.linalg.inv(sagittal.linear.detach().cpu().numpy()))
        initial_lines = sagittal.evaluate(data.initial, 1.)
        fixed = _sagittal_support(initial_lines, sagittal, data, world_yz)
        fixed &= sagittal.reference_mask.detach().cpu().numpy().astype(bool)
        bounds = data.bounds.detach().cpu().numpy()
        fixed &= ((world_yz >= bounds[0, 1:]) & (world_yz <= bounds[1, 1:])).all(-1)
        if not bounds[0, 0] <= sagittal.world_x_mm <= bounds[1, 0]:
            raise ValueError("冻结 sagittal 平面位于重建世界包围盒之外")
        if not _sagittal_metric_mask(fixed, sagittal.pitch_mm).any():
            raise ValueError("初始 sagittal 共同支持不足，不能生成有效结构对照")
        sagittal._evaluation_grid = reference, world_yz, fixed
    return sagittal._evaluation_grid


@torch.no_grad()
def evaluate_sweep(model, poses, data, sagittal, output, step, progress):
    """固定原生 sagittal 网格评估；交线指标固定使用最终尺度，便于跨步比较。"""
    output = Path(output)
    plot_dir = output / "plots" / f"step_{step:06d}"
    for folder in (plot_dir, output / "metrics", output / "predictions", output / "run_config"):
        folder.mkdir(parents=True, exist_ok=True)
    matrices = poses.matrices().detach()
    lines = sagittal.evaluate(matrices, 1.)
    reference, world_yz, fixed = _sagittal_grid(data, sagittal)
    current = _sagittal_support(lines, sagittal, data, world_yz)
    xyz = np.concatenate((np.full((*world_yz.shape[:-1], 1), sagittal.world_x_mm), world_yz), -1)
    gray = np.zeros_like(reference, dtype=np.float32)
    gray[fixed] = query(model, torch.as_tensor(xyz[fixed], device=data.initial.device, dtype=data.initial.dtype), progress)[..., 0]
    metrics, edge_difference = _sagittal_structure_metrics(gray, reference, fixed, sagittal.pitch_mm)
    mode = getattr(poses, "mode", "fixed")
    metrics.update(model=output.name, pose_mode=mode, step=int(step),
                   fixed_fov_coverage=float((current & fixed).sum()/fixed.sum()))
    ids = data.frame_ids.cpu().numpy()
    initial = torch.atan2(data.initial[:, 1, 0], -data.initial[:, 2, 0])
    differences = initial.diff()
    initial_angles = torch.cat((initial[:1], initial[:1]
                                + torch.atan2(differences.sin(), differences.cos()).cumsum(0))).cpu().numpy()
    angles = poses.angles().detach().cpu().numpy() if hasattr(poses, "angles") else initial_angles.copy()
    increments = np.diff(angles)
    trajectory_rows = [dict(frame_id=int(frame), initial_angle_deg=float(np.degrees(initial_angles[i])),
                            angle_deg=float(np.degrees(angles[i])),
                            angle_offset_deg=float(np.degrees(angles[i]-initial_angles[i])),
                            increment_deg=float(np.degrees(increments[i-1])) if i else None,
                            speed_deg_per_frame=float(np.degrees(increments[i-1])/(ids[i]-ids[i-1])) if i else None)
                       for i, frame in enumerate(ids)]
    write_csv(output / "metrics" / f"trajectory_{step:06d}.csv", trajectory_rows)
    line_rows = []
    for i, frame in enumerate(lines["frame_ids"]):
        row = dict(frame_id=int(frame), split=str(lines["split"][i]), window_count=int(lines["window_count"][i]))
        for key in ("loss", "edge_ncc", "ssim", "coverage"):
            value = float(lines[key][i])
            row[key] = value if math.isfinite(value) else None
        line_rows.append(row)
    write_csv(output / "metrics" / f"sagittal_lines_{step:06d}.csv", line_rows)
    summaries = []
    for split in ("training", "validation", "test"):
        selected = [row for row in line_rows if row["split"] == split]
        summary = dict(split=split, frame_count=len(selected),
                       supported_frames=sum(row["window_count"] > 0 for row in selected))
        for key in ("edge_ncc", "ssim", "coverage"):
            values = [row[key] for row in selected if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
        summaries.append(summary)
    np.savez_compressed(output / "predictions" / f"sweep_step_{step:06d}.npz",
                        gray=gray, reference=reference, fixed_support=fixed, current_support=current,
                        frame_ids=ids, initial_angles=initial_angles, angles=angles, increments=increments,
                        **{f"line_{key}": value for key, value in lines.items()})
    manifest = dict(world_x_mm=float(sagittal.world_x_mm), native_shape=list(reference.shape),
                    native_mapping="native_xy = world_yz @ linear + offset; frozen calibration",
                    linear=sagittal.linear.detach().cpu().tolist(), offset=sagittal.offset.detach().cpu().tolist(),
                    fixed_fov="initial source-sector-safe training intersection samples within 1.5 mm AND frozen reference ROI AND original bounds; no edge-window selection",
                    current_coverage="fraction of fixed FOV within 1.5 mm of corrected training intersection samples",
                    metric_domain="fixed FOV square-eroded ceil(4*sigma_px)+3 pixels; both images Gaussian sigma=0.5mm; edge magnitude NCC and smoothed-gray 7px SSIM",
                    reference_pitch_mm=float(sagittal.pitch_mm), metric_sigma_mm=.5,
                    intensity_scale="reference and reconstruction [0,1]; no model-specific normalization",
                    line_metric_scale="progress=1 at every evaluation; no per-step scale changes",
                    heldout_frames=data.frame_ids[data.comparison].cpu().tolist(),
                    limitations=["independent ultrasound views differ in speckle; no cross-plane MSE reported",
                                 "support is an approximate sampling-distance criterion, not a resolution estimate",
                                 "sagittal used for trajectory supervision; alignment metrics are not independent anatomical accuracy"])
    write_json(output / "run_config" / "sweep_comparison_manifest.json", manifest)
    write_json(output / "metrics" / f"sweep_step_{step:06d}.json",
               dict(**metrics, intersection_metrics=summaries, quality_claim="尚未验证；需审查固定视域图像与独立留出帧"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for axis, panel, title in zip(axes, (reference, gray, edge_difference),
                                  ("Independent sagittal reference", "Reconstructed sagittal", "Edge difference / reference p99")):
        display_mask = _sagittal_metric_mask(fixed, sagittal.pitch_mm) if axis is axes[2] else fixed
        axis.imshow(np.where(display_mask, panel, np.nan), cmap="magma" if axis is axes[2] else "gray", vmin=0, vmax=1)
        axis.set(title=title, xlabel="Native column [px]", ylabel="Native row [px]")
    fig.suptitle(f"{output.name}; step {step}; x={sagittal.world_x_mm:.6f} mm; fixed initial FOV; coverage={metrics['fixed_fov_coverage']:.3f}")
    fig.savefig(plot_dir / "sagittal_reference_comparison.png", dpi=140)
    plt.close(fig)
    return metrics


def compare_sweeps(data, sagittal, run_root, variants):
    """同初始域报告主指标，同共同覆盖域显示；不重新挑选图像、深度或裁剪 ROI。"""
    output = Path(run_root) / "comparison"
    for folder in ("plots", "metrics", "predictions", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    records, rows = {}, []
    for name in variants:
        directory = Path(run_root) / name
        path = sorted((directory / "predictions").glob("sweep_step_*.npz"))[-1]
        with np.load(path) as saved:
            records[name] = {key: saved[key].copy() for key in saved.files}
        rows.append(json.loads((directory / "metrics" / f"{path.stem}.json").read_text()))
    if len({row["step"] for row in rows}) != 1:
        raise ValueError("轨迹比较要求各分支具有相同 field update 数")
    first = records[variants[0]]
    reference, fixed = first["reference"], first["fixed_support"]
    for record in records.values():
        for key in ("reference", "fixed_support", "frame_ids", "initial_angles", "line_frame_ids", "line_depth_mm", "line_fixed_support"):
            if not np.array_equal(record[key], first[key]):
                raise ValueError(f"轨迹比较的固定输入不一致: {key}")
    common = fixed & np.logical_and.reduce([record["current_support"] for record in records.values()])
    np.savez_compressed(output / "predictions" / "sweep_support.npz", fixed_support=fixed, common_support=common,
                        **{f"coverage_{name}": record["current_support"] for name, record in records.items()})
    flat_rows = []
    for name, row in zip(variants, rows):
        common_metrics, _ = _sagittal_structure_metrics(records[name]["gray"], reference, common, sagittal.pitch_mm)
        flat_rows.append({key: row[key] for key in ("model", "pose_mode", "step", "pixel_count", "metric_pixel_count", "edge_ncc", "ssim", "fixed_fov_coverage")}
                         | {f"common_{key}": value for key, value in common_metrics.items()})
    write_csv(output / "metrics" / "sweep_summary.csv", flat_rows)
    fig, axes = plt.subplots(1, len(variants)+1, figsize=(5*(len(variants)+1), 6), squeeze=False, layout="constrained")
    for axis, (name, panel) in zip(axes[0], [("Independent sagittal", reference), *[(name, records[name]["gray"]) for name in variants]]):
        axis.imshow(np.where(common, panel, np.nan), cmap="gray", vmin=0, vmax=1)
        axis.set(title=name, xlabel="Native column [px]", ylabel="Native row [px]")
    fig.suptitle("Frozen native sagittal plane; shared current coverage within fixed initial FOV; [0,1]")
    fig.savefig(output / "plots" / "sweep_sagittal_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    ids = first["frame_ids"]
    axes[0].plot(ids, np.degrees(first["initial_angles"]), color="black", linestyle="--", label="Input trajectory")
    for name, record in records.items():
        axes[0].plot(ids, np.degrees(record["angles"]), label=name)
        axes[1].plot(ids, np.degrees(record["angles"]-record["initial_angles"]), label=name)
        axes[2].plot(ids[1:], np.degrees(record["increments"])/np.diff(ids), label=name)
    for axis, label in zip(axes, ("Angle [deg]", "Angle offset [deg]", "Angular speed [deg/frame]")):
        axis.set(ylabel=label)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Original frame ID")
    fig.suptitle("Shared sweep trajectory; heldout angles interpolated from training corrections")
    fig.savefig(output / "plots" / "sweep_trajectory_comparison.png", dpi=150)
    plt.close(fig)
    comparison_ids = data.frame_ids[data.comparison].cpu().tolist()
    fig, axes = plt.subplots(2, len(comparison_ids), figsize=(5*len(comparison_ids), 7), squeeze=False, layout="constrained")
    for column, frame in enumerate(comparison_ids):
        indices = np.flatnonzero(first["line_frame_ids"] == frame)
        if len(indices) != 1:
            raise ValueError(f"固定交线比较帧 {frame} 缺失")
        index = indices[0]
        valid = first["line_fixed_support"][index].astype(bool)
        depth = first["line_depth_mm"]
        for axis, quantity in zip(axes[:, column], ("profile", "edge")):
            axis.plot(depth, np.where(valid, first[f"line_source_{quantity}"][index], np.nan), color="black", label="Heldout sweep image")
            for name, record in records.items():
                axis.plot(depth, np.where(valid, record[f"line_reference_{quantity}"][index], np.nan), label=f"Sagittal: {name}")
            axis.set(title=f"Frame {frame}; {quantity}", xlabel="Probe depth [mm]")
        axes[0, column].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed heldout frames and source depths; reference sampled along each inferred intersection")
    fig.savefig(output / "plots" / "sweep_heldout_profiles.png", dpi=150)
    plt.close(fig)
    summary = dict(variants=list(variants), image_metrics=flat_rows,
                   intersection_metrics={name: row["intersection_metrics"] for name, row in zip(variants, rows)},
                   fixed_fov_pixels=int(fixed.sum()), common_coverage_pixels=int(common.sum()),
                   common_coverage_fraction=float(common.sum()/fixed.sum()), heldout_comparison_ids=comparison_ids,
                   selection="frozen native grid, initial FOV and heldout IDs; no output-based crop or profile selection",
                   metric_interpretation="primary metrics use fixed initial FOV; common_* metrics use the shared current coverage shown",
                   quality_claim="尚未验证；结合重建切片、结构指标、覆盖率及源帧拟合检查",
                   limitations=["sagittal is supervision and not an independent anatomical ground truth",
                                "heldout sweep intensities are not fitted; their angles follow the learned trajectory",
                                "one seed; equal field updates do not imply equal runtime"])
    write_json(output / "metrics" / "sweep_summary.json", summary)
    return summary


def _guided_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _guided_reslice(data, matrices, guidance, xyz):
    """只用同一 training split、原始分辨率及固定列；不借用留出帧填洞。"""
    selected = _guided_numpy(data.splits["training"])
    column = guidance.column
    valid = _guided_numpy(data.mask)[:, column].astype(bool)
    local = _guided_numpy(data.local)[:, column][valid]
    matrices = _guided_numpy(matrices)[selected]
    points = local[None] @ matrices[:, :3, :3].transpose(0, 2, 1) + matrices[:, None, :3, 3]
    if np.max(np.abs(points[..., 0] - xyz[0, 0, 0])) > 1e-3:
        raise ValueError("固定源图列与 sagittal 世界平面不共面")
    values = _guided_numpy(data.images[data.splits["training"], :, column])[:, valid]
    result = griddata(points[..., 1:].reshape(-1, 2), values.reshape(-1), xyz[..., 1:], method="linear")
    return result.astype(np.float32), np.isfinite(result)


def _guided_safe_support(support, pitch_mm):
    # 两个尺度共享分母。0.5mm Gaussian 的 4σ，再留 Canny 4px Gaussian/梯度邻域。
    radius = int(math.ceil(4 * .5 / pitch_mm)) + 6
    return minimum_filter(support, size=2 * radius + 1, mode="constant", cval=0).astype(bool)


def _guided_edge_thresholds(reference, fixed, roi, pitch_mm):
    """双阈值只由固定参考图定义，跨模型/步数保持绝对梯度阈值一致。"""
    mask = _guided_safe_support(fixed, pitch_mm) & roi
    thresholds = []
    for sigma_mm in (0., .5):
        smoothed = gaussian_filter(reference, sigma_mm / pitch_mm) if sigma_mm else reference
        canny_input = gaussian_filter(smoothed, 1.)
        strength = np.hypot(sobel(canny_input, axis=0), sobel(canny_input, axis=1))
        positive = strength[mask & (strength > 1e-8)]
        high = float(np.quantile(positive, .8)) if positive.size else 1.
        thresholds.append((.4 * high, high))
    return np.asarray(thresholds, dtype=np.float64)


def _guided_correlation(first, second):
    if first.size < 2 or min(first.std(), second.std()) <= 1e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _guided_edge_metrics(predicted, reference, pitch_mm):
    """两边的候选都先限制在该 ROI 内，禁止匹配 ROI 外边缘获取高分。"""
    p, r = np.argwhere(predicted), np.argwhere(reference)
    metrics = dict(predicted_edge_pixels=len(p), reference_edge_pixels=len(r),
                   edge_mean_distance_mm=None, edge_p95_distance_mm=None)
    if len(p) and len(r):
        p_to_r = cKDTree(r).query(p, workers=2)[0] * pitch_mm
        r_to_p = cKDTree(p).query(r, workers=2)[0] * pitch_mm
        # 两个方向等权；p95 同时报告最差方向，避免密集预测淹没参考漏检。
        metrics.update(edge_mean_distance_mm=float((p_to_r.mean() + r_to_p.mean()) / 2),
                       edge_p95_distance_mm=float(max(np.quantile(p_to_r, .95), np.quantile(r_to_p, .95))))
    else:
        p_to_r = np.full(len(p), np.inf)
        r_to_p = np.full(len(r), np.inf)
    for tolerance, suffix in ((.5, "0p5mm"), (1., "1mm")):
        precision = float(np.mean(p_to_r <= tolerance)) if len(p) else None
        recall = float(np.mean(r_to_p <= tolerance)) if len(r) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall > 0
              else 0. if len(p) or len(r) else None)
        metrics.update({f"edge_precision_{suffix}": precision, f"edge_recall_{suffix}": recall,
                        f"edge_f1_{suffix}": f1})
    return metrics


_GUIDED_QUALITY_DEFINITIONS = dict(
    regions="仅用参考图 Gaussian sigma=0.5mm 后梯度（强度/mm）选区；固定安全 ROI 内 >=80% 分位且梯度>1e-8 为边缘候选，<=25% 分位为平坦候选，包含并列值；两个尺度复用相同区域",
    clarity="固定参考边缘候选区内，当前图像尺度下梯度 RMS 与参考 RMS 之比；不代表测得的空间分辨率",
    flat_highpass="原生灰度减 Gaussian sigma=0.5mm，在固定参考平坦候选区计算 RMS；各尺度行重复同一原生指标，缺失上下文贡献零并单独报告覆盖率",
    interpretation="平坦区高频减少可能来自适度降噪、细节损失或覆盖缺失；须联合边缘重合/距离、梯度、灰度 MAE、对比度和覆盖率判断，不按高频越低越好排序；跨方向超声外观可不同",
)


def _guided_metric_rows(gray, reference, fixed, current, roi, core, pitch_mm, thresholds, *, domain="fixed"):
    """保留固定分母；缺失灰度补零，缺失边缘算漏检且不计填零边界。"""
    fixed_safe = _guided_safe_support(fixed, pitch_mm)
    current_safe = _guided_safe_support(current, pitch_mm)
    filled = np.where(current & np.isfinite(gray), gray, 0).astype(np.float32)
    # 区域只由参考图定义，两个评价尺度复用；低梯度仅表示平坦候选，并非无噪声真值。
    reference_lowpass = gaussian_filter(reference, .5 / pitch_mm)
    structure_gradient = np.hypot(*np.gradient(reference_lowpass)) / pitch_mm
    selection = fixed_safe & roi
    flat_threshold, edge_threshold = (np.quantile(structure_gradient[selection], (.25, .8))
                                      if selection.any() else (None, None))
    edge_region = (selection & (structure_gradient >= edge_threshold) & (structure_gradient > 1e-8)
                   if selection.any() else selection.copy())
    flat_region = (selection & (structure_gradient <= flat_threshold)
                   if selection.any() else selection.copy())
    # 高频代理始终作用于原生灰度，避免不同 scale 改变“平坦区高频”的定义与滤波上下文。
    p_highpass = np.where(current_safe, filled - gaussian_filter(filled, .5 / pitch_mm), 0)
    r_highpass = reference - reference_lowpass
    regions = dict(all=roi, core=core & roi, extension=roi & ~core)
    rows, edge_panels = [], {}
    for scale_index, (scale, sigma_mm) in enumerate((("native", 0.), ("smooth_0p5mm", .5))):
        prediction = gaussian_filter(filled, sigma_mm / pitch_mm) if sigma_mm else filled
        target = gaussian_filter(reference, sigma_mm / pitch_mm) if sigma_mm else reference
        _, similarity = structural_similarity(target, prediction, data_range=1., win_size=7, full=True)
        # 缺失区填零的跳变不能充当预测边缘，边缘 NCC 同样采用安全上下文 gate。
        p_gradient = np.where(current_safe, np.hypot(*np.gradient(prediction)), 0)
        r_gradient = np.hypot(*np.gradient(target))
        low, high = thresholds[scale_index]
        reference_edges = canny(target, sigma=1., low_threshold=low, high_threshold=high, mask=fixed)
        predicted_edges = canny(prediction, sigma=1., low_threshold=low, high_threshold=high,
                                mask=current) & current_safe
        edge_panels[scale] = (reference_edges & fixed_safe & roi, predicted_edges & fixed_safe & roi)
        for region, region_mask in regions.items():
            selected = region_mask & fixed_safe
            count = int(selected.sum())
            requested = int(region_mask.sum())
            edge_selected, flat_selected = selected & edge_region, selected & flat_region
            edge_count, flat_count = int(edge_selected.sum()), int(flat_selected.sum())
            p_edge = float(np.sqrt(np.mean(p_gradient[edge_selected] ** 2)) / pitch_mm) if edge_count else None
            r_edge = float(np.sqrt(np.mean(r_gradient[edge_selected] ** 2)) / pitch_mm) if edge_count else None
            p_flat = float(np.sqrt(np.mean(p_highpass[flat_selected] ** 2))) if flat_count else None
            r_flat = float(np.sqrt(np.mean(r_highpass[flat_selected] ** 2))) if flat_count else None
            reference_std = float(target[selected].std()) if count else None
            row = dict(domain=domain, region=region, scale=scale, sigma_mm=sigma_mm,
                       roi_pixels=requested, metric_pixels=count,
                       reference_support_fraction=float(count / requested) if requested else None,
                       coverage=float((selected & current).sum() / count) if count else None,
                       edge_context_coverage=float((selected & current_safe).sum() / count) if count else None,
                       gray_mae=float(np.abs(prediction[selected] - target[selected]).mean()) if count else None,
                       contrast_std_ratio=(float(prediction[selected].std()) / reference_std
                                           if reference_std is not None and reference_std > 1e-8 else None),
                       ssim=float(similarity[selected].mean()) if count else None,
                       gray_ncc=_guided_correlation(prediction[selected], target[selected]),
                       edge_ncc=_guided_correlation(p_gradient[selected], r_gradient[selected]),
                       reference_edge_region_pixels=edge_count, reference_flat_region_pixels=flat_count,
                       reference_edge_threshold_per_mm=float(edge_threshold) if edge_threshold is not None else None,
                       reference_flat_threshold_per_mm=float(flat_threshold) if flat_threshold is not None else None,
                       edge_region_context_coverage=float((edge_selected & current_safe).sum() / edge_count) if edge_count else None,
                       edge_gradient_rms_per_mm=p_edge, reference_edge_gradient_rms_per_mm=r_edge,
                       edge_gradient_rms_ratio=p_edge / r_edge if r_edge is not None and r_edge > 1e-8 else None,
                       flat_region_context_coverage=float((flat_selected & current_safe).sum() / flat_count) if flat_count else None,
                       flat_native_highpass_rms=p_flat, reference_flat_native_highpass_rms=r_flat,
                       flat_native_highpass_rms_ratio=p_flat / r_flat if r_flat is not None and r_flat > 1e-8 else None,
                       flat_native_highpass_sigma_mm=.5)
            row.update(_guided_edge_metrics(predicted_edges & selected, reference_edges & selected, pitch_mm))
            rows.append(row)
    return rows, edge_panels


def _guided_plot(gray_by_name, reference, fixed, current_by_name, roi, core, pitch_mm, thresholds, output, *, crop):
    """固定原生方向和 [0,1]；白为边缘重叠，青为参考，紫为预测。"""
    names = list(gray_by_name)
    fig, axes = plt.subplots(3, len(names) + 1, figsize=(5 * (len(names) + 1), 12),
                             squeeze=False, layout="constrained")
    display = fixed
    axes[0, 0].imshow(np.where(display, reference, np.nan), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Sagittal reference (supervision)")
    axes[1, 0].imshow(np.where(display, reference, np.nan), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Fixed ROI: annotation hull + physical expansion")
    axes[2, 0].imshow(np.where(display, reference, np.nan), cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("Edges: cyan ref / magenta result / white overlap")
    for column, name in enumerate(names, 1):
        current = current_by_name[name]
        gray = np.where(current, gray_by_name[name], 0)
        axes[0, column].imshow(np.where(display, gray, np.nan), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(name)
        axes[1, column].imshow(np.where(display & roi, np.abs(gray - reference), np.nan),
                               cmap="magma", vmin=0, vmax=1)
        axes[1, column].set_title("Absolute gray difference [0,1]")
        safe = _guided_safe_support(fixed, pitch_mm) & roi
        current_safe = _guided_safe_support(current, pitch_mm)
        low, high = thresholds[1]
        ref_edge = canny(gaussian_filter(reference, .5 / pitch_mm), sigma=1.,
                         low_threshold=low, high_threshold=high, mask=fixed) & safe
        predicted_edge = canny(gaussian_filter(gray, .5 / pitch_mm), sigma=1.,
                               low_threshold=low, high_threshold=high, mask=current) & current_safe & safe
        overlay = np.repeat((np.where(display, reference, 0) * .45)[..., None], 3, axis=-1)
        overlay[ref_edge] = (0., 1., 1.)
        overlay[predicted_edge] = (1., 0., 1.)
        overlay[ref_edge & predicted_edge] = (1., 1., 1.)
        axes[2, column].imshow(overlay, vmin=0, vmax=1)
        axes[2, column].set_title("Edges after 0.5 mm smoothing")
    for axis in axes.flat:
        if roi.any() and not roi.all():
            axis.contour(roi, levels=[.5], colors=["yellow"], linewidths=.6)
        if core.any() and not core.all():
            axis.contour(core, levels=[.5], colors=["lime"], linewidths=.6)
        axis.set(xlabel="Native sagittal column [px]", ylabel="Native sagittal row [px]")
        if crop and roi.any():
            rr, cc = np.nonzero(roi)
            axis.set_xlim(max(0, cc.min() - 5), min(roi.shape[1] - 1, cc.max() + 5))
            axis.set_ylim(min(roi.shape[0] - 1, rr.max() + 5), max(0, rr.min() - 5))
    fig.suptitle("cerebral / index_all; same training frames, native grid, fixed ROI/support and [0,1]; missing result = 0")
    fig.savefig(output, dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate_guided_sagittal(model, poses, data, guidance, output, step, progress, *, raw_only=False):
    """全点 ROI 的统一评分；model=None 时评估相同训练帧的线性重采样。"""
    output = Path(output)
    plot_dir = output / "plots" / f"step_{step:06d}"
    for folder in (plot_dir, output / "metrics", output / "predictions", output / "run_config"):
        folder.mkdir(parents=True, exist_ok=True)
    reference = _guided_numpy(guidance.reference).astype(np.float32)
    xyz = _guided_numpy(guidance.xyz_grid).astype(np.float32)
    roi, core = _guided_numpy(guidance.roi).astype(bool), _guided_numpy(guidance.core_roi).astype(bool)
    reference_mask = _guided_numpy(guidance.reference_mask).astype(bool)
    pitch = float(guidance.pitch_mm)
    if not hasattr(guidance, "_guided_evaluation_support"):
        _, initial_support = _guided_reslice(data, data.initial, guidance, xyz)
        fixed = initial_support & reference_mask
        if not (roi & _guided_safe_support(fixed, pitch)).any():
            raise ValueError("固定初始训练支持与标注 ROI 没有足够的滤波上下文")
        thresholds = _guided_edge_thresholds(reference, fixed, roi, pitch)
        guidance._guided_evaluation_support = fixed, thresholds
    fixed, thresholds = guidance._guided_evaluation_support
    raw_gray, source_support = _guided_reslice(data, poses.matrices().detach(), guidance, xyz)
    current = source_support & reference_mask
    model_bounds = None
    if model is None or raw_only:
        gray = np.where(current, raw_gray, 0).astype(np.float32)
        method = "training_frames_linear_griddata"
    else:
        if model.response_only:
            raise ValueError("sagittal 灰度评价需要具有灰度输出的模型")
        if hasattr(model, "evaluation_bounds"):
            model_bounds = _guided_numpy(model.evaluation_bounds)
        elif model.hash_encoder is not None:
            model_bounds = np.stack((_guided_numpy(model.hash_encoder.bound_min), _guided_numpy(model.hash_encoder.bound_max)))
        else:
            center, extent = _guided_numpy(model.center), _guided_numpy(model.extent)
            model_bounds = np.stack((center - extent / 2, center + extent / 2))
        bounds_mask = ((xyz >= model_bounds[0]) & (xyz <= model_bounds[1])).all(-1)
        current &= bounds_mask
        gray = np.zeros_like(reference)
        # 查询整个模型有效网格，为 ROI 边缘的滤波保留真实上下文，禁止只渲染 ROI。
        if bounds_mask.any():
            gray[bounds_mask] = query(model, torch.as_tensor(xyz[bounds_mask], device=data.initial.device,
                                                            dtype=data.initial.dtype), progress)[..., 0]
        if not np.isfinite(gray).all():
            raise ValueError("sagittal 模型查询产生非有限灰度")
        gray[~current] = 0
        method = "neural_field_direct_world_grid"
    rows, _ = _guided_metric_rows(gray, reference, fixed, current, roi, core, pitch, thresholds)
    common = fixed & current
    shared_rows, _ = _guided_metric_rows(gray, reference, common, current, roi, core, pitch, thresholds,
                                       domain="own_current_support_auxiliary")
    for row in rows + shared_rows:
        row.update(model=output.name, step=int(step), method=method)
    uses_guidance = hasattr(poses, "fractional_world")
    landmarks = guidance.landmark_rows(poses) if uses_guidance else []
    frame_ids = _guided_numpy(data.frame_ids[data.splits["training"]])
    basename = f"guided_sagittal_step_{step:06d}"
    prediction_path = output / "predictions" / f"{basename}.npz"
    np.savez_compressed(prediction_path, gray=gray, reference=reference, fixed_support=fixed,
                        current_support=current, source_support=source_support, reference_mask=reference_mask,
                        roi=roi, core_roi=core, xyz_grid=xyz, training_frame_ids=frame_ids,
                        pitch_mm=pitch, step=int(step), edge_thresholds=thresholds,
                        model_bounds=np.asarray([]) if model_bounds is None else model_bounds)
    primary = next(row for row in rows if row["region"] == "all" and row["scale"] == "smooth_0p5mm")
    summary = dict(**primary, rows=rows + shared_rows, landmarks=landmarks,
                   quality_diagnostics=_GUIDED_QUALITY_DEFINITIONS,
                   prediction_path=str(prediction_path),
                   guidance_role="training" if uses_guidance else "evaluation_only",
                   quality_claim="sagittal 为引导方法的训练参考；此处评估拟合程度，不代表留出解剖精度")
    write_csv(output / "metrics" / f"{basename}.csv", rows + shared_rows)
    if landmarks:
        write_csv(output / "metrics" / f"landmarks_step_{step:06d}.csv", landmarks)
    write_json(output / "metrics" / f"{basename}.json", summary)
    manifest = dict(guidance=guidance.metadata, training_frame_ids=frame_ids.tolist(),
                    source_image_shape=list(data.images.shape[1:]), source_column=int(guidance.column),
                    fixed_support="finite initial training-frame linear griddata AND physical reference sector; independent of field bounds",
                    current_support="finite corrected training-frame griddata AND physical reference sector AND field bounds (for NeRF)",
                    main_domain="ROI intersect eroded fixed support; ROI itself is not eroded; same denominator for all methods",
                    fill_policy="missing predictions are zero on fixed main domain; never silently drop unsupported pixels",
                    edge_policy="reference-only absolute Canny thresholds; sigma=1 native pixel after stated gray smoothing; predicted edges gated by eroded current context; both matching sets restricted to each ROI",
                    edge_thresholds_native_then_smoothed=thresholds.tolist(),
                    edge_distance="mean: equally weighted mean of two directed distances; p95: maximum of two directed 95th percentiles",
                    quality_diagnostics=_GUIDED_QUALITY_DEFINITIONS,
                    metric_context_radius_px=int(math.ceil(4 * .5 / pitch)) + 6,
                    undefined_metrics="null for empty regions, correlation with constant images, ratios with reference denominator <=1e-8, and distance with empty edge set",
                    intensity_scale="shared [0,1], no model-specific normalization", pitch_mm=pitch,
                    regions="all=hull+expansion, core=hull, extension=all-core",
                    guidance_role="training" if uses_guidance else "evaluation_only",
                    limitation="sagittal and approximate landmarks guide the sagittal methods; these fit metrics do not establish heldout anatomical accuracy")
    write_json(output / "run_config" / "guided_sagittal_manifest.json", manifest)
    for crop, suffix in ((False, "full"), (True, "roi")):
        _guided_plot({output.name: gray}, reference, fixed, {output.name: current}, roi, core, pitch, thresholds,
                     plot_dir / f"guided_sagittal_{suffix}.png", crop=crop)
    return summary


def compare_guided_sagittal(paths, output):
    """paths={显示名:统一评价 NPZ}；重新计算固定主域和跨方法共同覆盖辅助域。"""
    if not paths:
        raise ValueError("至少需要一个已经按全点 ROI 重新查询的 sagittal 结果")
    output = Path(output)
    for folder in ("metrics", "plots", "predictions", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    records = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as saved:
            records[name] = {key: saved[key].copy() for key in saved.files}
    first = next(iter(records.values()))
    for name, record in records.items():
        for key in ("reference", "fixed_support", "roi", "core_roi", "xyz_grid", "training_frame_ids", "pitch_mm", "edge_thresholds"):
            if not np.array_equal(record[key], first[key]):
                raise ValueError(f"统一 sagittal 比较的固定输入不一致: {name}, {key}")
    reference, fixed, roi, core = (first[key] for key in ("reference", "fixed_support", "roi", "core_roi"))
    pitch, thresholds = float(first["pitch_mm"]), first["edge_thresholds"]
    common = fixed & np.logical_and.reduce([record["current_support"] for record in records.values()])
    rows = []
    for name, record in records.items():
        for domain, support in (("fixed", fixed), ("all_models_common_support_auxiliary", common)):
            metrics, _ = _guided_metric_rows(record["gray"], reference, support, record["current_support"],
                                            roi, core, pitch, thresholds, domain=domain)
            rows.extend(dict(model=name, step=int(record["step"]), **row) for row in metrics)
    write_csv(output / "metrics" / "guided_sagittal_comparison.csv", rows)
    primary = [row for row in rows if row["domain"] == "fixed" and row["region"] == "all"
               and row["scale"] == "smooth_0p5mm"]
    summary = dict(rows=rows, primary_rows=primary, inputs={name: str(path) for name, path in paths.items()},
                   common_support_pixels=int((common & roi).sum()),
                   quality_diagnostics=_GUIDED_QUALITY_DEFINITIONS,
                   interpretation="fixed rows use identical initial support; shared-support rows are auxiliary; all sagittal metrics measure supervised fit",
                   quality_claim="尚未验证；统一固定域图像、边缘重合/距离、清晰度、灰度保真和覆盖率需共同审查；平坦区高频降低不能单独证明降噪成功")
    write_json(output / "metrics" / "guided_sagittal_comparison.json", summary)
    np.savez_compressed(output / "predictions" / "guided_sagittal_support.npz", fixed_support=fixed,
                        common_support=common, roi=roi, core_roi=core,
                        **{f"current_{name}": record["current_support"] for name, record in records.items()})
    write_json(output / "run_config" / "comparison_manifest.json",
               dict(comparison_indices=["native_sagittal"], inputs=summary["inputs"],
                    normalization="shared [0,1] without per-model normalization", pitch_mm=pitch,
                    training_frame_ids=first["training_frame_ids"].tolist(),
                    crop="fixed annotation hull plus configured physical expansion", display_support="fixed initial support",
                    metrics="same definitions as each guided_sagittal_manifest.json; full and ROI figures use fixed support"))
    for crop, suffix in ((False, "full"), (True, "roi")):
        _guided_plot({name: record["gray"] for name, record in records.items()}, reference, fixed,
                     {name: record["current_support"] for name, record in records.items()}, roi, core, pitch,
                     thresholds, output / "plots" / f"guided_sagittal_comparison_{suffix}.png", crop=crop)
    return summary
