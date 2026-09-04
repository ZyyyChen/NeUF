"""在统一凸阵网格上比较 NeUF 与 UltraNeRF 的整集重建质量。"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from skimage.metrics import structural_similarity
import torch
from tqdm import tqdm

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.phase1_data import FrameRef, build_phase1_split, metric_mask
from neuf.slice_renderer import SliceRenderer


@dataclass(frozen=True)
class ComparisonConfig:
    workspace_root: Path
    dataset_path: Path
    ultra_dataset_dir: Path
    ultra_config_path: Path
    checkpoints: dict[str, Path]
    output_dir: Path
    base_seed: int = 20260901
    roi_size: int = 24
    roi_pairs_per_frame: int = 3
    histogram_bins: int = 64


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = f"{array.dtype.str}|{array.shape}|".encode("utf-8")
    return hashlib.sha256(descriptor + array.tobytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _partition_tensor(dataset: Dataset, ref: FrameRef, name: str) -> torch.Tensor:
    source_name = name if ref.source == "training_pool" else f"{name}_valid"
    tensor = getattr(dataset, source_name)
    return tensor[ref.slice_info.start : ref.slice_info.end]


def _native_neuf_image(dataset: Dataset, ref: FrameRef, name: str = "pixels") -> np.ndarray:
    return (
        _partition_tensor(dataset, ref, name)
        .detach()
        .cpu()
        .numpy()
        .reshape(dataset.px_height, dataset.px_width)
    )


def _all_frame_refs(dataset: Dataset) -> list[FrameRef]:
    splits = build_phase1_split(dataset)
    refs = sorted(
        [ref for split in ("training", "validation", "test") for ref in splits[split]],
        key=lambda ref: ref.original_frame_index,
    )
    frame_ids = [ref.original_frame_index for ref in refs]
    if frame_ids != list(range(len(frame_ids))):
        raise ValueError(f"NeUF frame IDs are not contiguous: {frame_ids[:5]}...{frame_ids[-5:]}")
    return refs


def _load_lpips(device: torch.device):
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is required for this comparison but is not installed in the qsub environment"
        ) from error
    model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _masked_lpips(
    model,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> float:
    rows, columns = np.nonzero(mask)
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(columns.min()), int(columns.max()) + 1
    valid = mask[row0:row1, col0:col1].astype(np.float32)
    pred = prediction[row0:row1, col0:col1].astype(np.float32) * valid
    truth = target[row0:row1, col0:col1].astype(np.float32) * valid
    pred_tensor = torch.from_numpy(pred)[None, None].to(device)
    truth_tensor = torch.from_numpy(truth)[None, None].to(device)
    pred_tensor = pred_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    truth_tensor = truth_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.inference_mode():
        return float(model(pred_tensor, truth_tensor, normalize=False).reshape(()).item())


def _masked_ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    _, similarity = structural_similarity(
        target.astype(np.float64),
        prediction.astype(np.float64),
        data_range=1.0,
        win_size=7,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,
        full=True,
    )
    return float(similarity[mask].mean())


def _gcnr(
    image: np.ndarray,
    first: tuple[slice, slice],
    second: tuple[slice, slice],
    histogram_bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, histogram_bins + 1)
    first_hist, _ = np.histogram(image[first], bins=edges)
    second_hist, _ = np.histogram(image[second], bins=edges)
    first_prob = first_hist.astype(np.float64) / max(int(first_hist.sum()), 1)
    second_prob = second_hist.astype(np.float64) / max(int(second_hist.sum()), 1)
    return float(1.0 - np.minimum(first_prob, second_prob).sum())


def _window_slice(row: int, column: int, size: int) -> tuple[slice, slice]:
    return slice(row, row + size), slice(column, column + size)


def _gt_roi_pairs(
    target: np.ndarray,
    mask: np.ndarray,
    *,
    roi_size: int,
    pair_count: int,
    histogram_bins: int,
) -> list[dict[str, object]]:
    """仅依据 GT 选择覆盖低、中、高对比难度的固定 ROI 对。"""
    low = gaussian_filter(target.astype(np.float64), sigma=2.0, mode="reflect")
    gradient = np.hypot(sobel(low, axis=0, mode="reflect"), sobel(low, axis=1, mode="reflect"))
    stride = max(4, roi_size // 2)
    candidates: list[dict[str, float | int]] = []
    for row in range(0, target.shape[0] - roi_size + 1, stride):
        for column in range(0, target.shape[1] - roi_size + 1, stride):
            window = _window_slice(row, column, roi_size)
            if not bool(mask[window].all()):
                continue
            mean = float(low[window].mean())
            if not 0.05 <= mean <= 0.95:
                continue
            candidates.append(
                {
                    "row": row,
                    "column": column,
                    "mean": mean,
                    "std": float(low[window].std()),
                    "gradient": float(gradient[window].mean()),
                    "center_row": row + 0.5 * roi_size,
                    "center_column": column + 0.5 * roi_size,
                }
            )
    if len(candidates) < 2:
        raise ValueError("GT does not contain enough fully valid ROI candidates")

    pair_candidates: list[tuple[float, int, int]] = []
    minimum_distance = float(roi_size)
    maximum_distance = float(roi_size * 5)
    for left in range(len(candidates)):
        first = candidates[left]
        for right in range(left + 1, len(candidates)):
            second = candidates[right]
            distance = math.hypot(
                float(first["center_row"]) - float(second["center_row"]),
                float(first["center_column"]) - float(second["center_column"]),
            )
            if not minimum_distance <= distance <= maximum_distance:
                continue
            contrast = abs(float(first["mean"]) - float(second["mean"]))
            nuisance = (
                float(first["std"])
                + float(second["std"])
                + 0.25 * (float(first["gradient"]) + float(second["gradient"]))
                + 1e-6
            )
            pair_candidates.append((contrast / nuisance, left, right))
    pair_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    # 先覆盖完整的候选排名，再用 GT gCNR 锚点选难度，避免只取最高对比导致饱和。
    sampled_ranks = np.unique(
        np.linspace(0, len(pair_candidates) - 1, min(1024, len(pair_candidates)), dtype=np.int64)
    )
    evaluated = []
    for rank in sampled_ranks:
        score, left, right = pair_candidates[int(rank)]
        first = candidates[left]
        second = candidates[right]
        first_window = _window_slice(int(first["row"]), int(first["column"]), roi_size)
        second_window = _window_slice(int(second["row"]), int(second["column"]), roi_size)
        evaluated.append(
            (
                _gcnr(target, first_window, second_window, histogram_bins),
                float(rank) / max(len(pair_candidates) - 1, 1),
                score,
                left,
                right,
            )
        )
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    anchors = np.linspace(0.35, 0.85, pair_count)
    for anchor in anchors:
        available = [item for item in evaluated if item[3] not in used and item[4] not in used]
        if not available:
            break
        target_gcnr, rank_fraction, score, left, right = min(
            available,
            key=lambda item: (abs(item[0] - float(anchor)) + 0.02 * item[1], item[1], item[3], item[4]),
        )
        first, second = candidates[left], candidates[right]
        if float(first["mean"]) > float(second["mean"]):
            first, second = second, first
        selected.append(
            {
                "low": {key: first[key] for key in ("row", "column", "mean", "std", "gradient")},
                "high": {key: second[key] for key in ("row", "column", "mean", "std", "gradient")},
                "selection_score": float(score),
                "target_gcnr_anchor": float(anchor),
                "target_selection_gcnr": float(target_gcnr),
                "candidate_rank_fraction": float(rank_fraction),
            }
        )
        used.update((left, right))
    if len(selected) != pair_count:
        raise ValueError(f"GT yielded only {len(selected)} disjoint ROI pairs, expected {pair_count}")
    return selected


def _roi_slices(entry: dict[str, object], roi_size: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    low = entry["low"]
    high = entry["high"]
    assert isinstance(low, dict) and isinstance(high, dict)
    return (
        _window_slice(int(low["row"]), int(low["column"]), roi_size),
        _window_slice(int(high["row"]), int(high["column"]), roi_size),
    )


def _frame_metrics(
    prediction_raw: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    roi_pairs: list[dict[str, object]],
    *,
    roi_size: int,
    histogram_bins: int,
    lpips_model,
    device: torch.device,
) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction_raw, dtype=np.float32), 0.0, 1.0)
    truth = np.asarray(target, dtype=np.float32)
    error = prediction - truth
    prediction_gcnr = []
    target_gcnr = []
    for entry in roi_pairs:
        first, second = _roi_slices(entry, roi_size)
        prediction_gcnr.append(_gcnr(prediction, first, second, histogram_bins))
        target_gcnr.append(_gcnr(truth, first, second, histogram_bins))
    pred_gcnr = float(np.mean(prediction_gcnr))
    truth_gcnr = float(np.mean(target_gcnr))
    return {
        "mse": float(np.mean(np.square(error)[mask])),
        "ssim": _masked_ssim(prediction, truth, mask),
        "lpips": _masked_lpips(lpips_model, prediction, truth, mask, device),
        "gcnr": pred_gcnr,
        "target_gcnr": truth_gcnr,
        "gcnr_abs_error": abs(pred_gcnr - truth_gcnr),
        "clip_fraction": float(np.mean((prediction_raw[mask] < 0.0) | (prediction_raw[mask] > 1.0))),
    }


def _summary(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    metrics = ("mse", "ssim", "lpips", "gcnr", "target_gcnr", "gcnr_abs_error", "clip_fraction")
    table = []
    payload: dict[str, object] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        model_summary: dict[str, object] = {"frame_count": len(selected)}
        flat: dict[str, object] = {"model": model, "frame_count": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            stats = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            model_summary[metric] = stats
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
        payload[model] = model_summary
        table.append(flat)
    return table, payload


def _load_ultra_runtime(config: ComparisonConfig, device: torch.device):
    ultra_root = config.workspace_root / "UltraNeRF-Studio"
    ultra_src = ultra_root / "src"
    for path in (ultra_root, ultra_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from ultranerf.evaluation.protocol import fixed_render_seed, isolated_rng
    from ultranerf.load_us import load_us_data
    from ultranerf.nerf_utils import create_nerf, render_us
    from ultranerf.probe_geometry import build_probe_geometry_from_args, remap_image_to_convex_grid
    from ultranerf.unerf_config import config_parser

    arguments = config_parser().parse_args(["--config", str(config.ultra_config_path)])
    arguments.datadir = str(config.ultra_dataset_dir)
    arguments.ft_path = str(config.checkpoints["UltraNeRF_Hash"])
    arguments.reconstruction = False
    images, poses, _ = load_us_data(arguments.datadir, confmap=arguments.confmap)
    geometry = build_probe_geometry_from_args(arguments)
    height, width = geometry.convex_render_shape
    step_height = float(geometry.convex_scale_y_mm) * 0.001
    step_width = float(geometry.convex_scale_x_mm) * 0.001
    far = (geometry.convex_outer_radius_mm - geometry.convex_inner_radius_mm) * 0.001
    _, render_kwargs, loaded_step, _, _ = create_nerf(arguments, device=device, mode="eval")
    render_kwargs.update({"near": 0.0, "far": far})
    render_kwargs["network_fn"].eval()

    def render(frame_id: int) -> np.ndarray:
        pose = torch.from_numpy(poses[frame_id, :3, :4]).to(device).unsqueeze(0)
        seed = fixed_render_seed(config.base_seed, frame_id, 0)
        with isolated_rng(seed, device), torch.inference_mode():
            output = render_us(
                height,
                width,
                step_width,
                step_height,
                c2w=pose,
                chunk=int(arguments.chunk),
                retraw=True,
                **render_kwargs,
            )
        prediction = output["intensity_map"].detach().cpu().numpy()[0, 0]
        if prediction.shape != (height, width) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid UltraNeRF frame {frame_id}: {prediction.shape}")
        return prediction.astype(np.float32)

    targets = [remap_image_to_convex_grid(image, geometry) for image in images]
    return geometry, targets, render, int(loaded_step), remap_image_to_convex_grid


def _load_neuf_model(checkpoint_path: Path, dataset: Dataset, device: torch.device) -> NeRF:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_dataset_geometry(checkpoint, dataset, checkpoint_path=checkpoint_path)
    if int(checkpoint.get("start", -1)) != 20_000:
        raise ValueError(f"NeUF checkpoint is not step 20000: {checkpoint_path}")
    model = NeRF(checkpoint)
    model.training_progress = 1.0
    model.eval()
    return model


def _render_neuf_native(
    model: NeRF,
    renderer: SliceRenderer,
    dataset: Dataset,
    ref: FrameRef,
) -> np.ndarray:
    points = _partition_tensor(dataset, ref, "points").unsqueeze(1)
    viewdirs = _partition_tensor(dataset, ref, "viewdirs").unsqueeze(1)
    with torch.inference_mode():
        prediction = renderer.query_points(model, points, viewdirs, alpha=1.0)
    return prediction.detach().cpu().numpy().reshape(dataset.px_height, dataset.px_width)


def _save_representative_plot(
    path: Path,
    targets: list[np.ndarray],
    predictions: dict[tuple[str, int], np.ndarray],
    models: list[str],
) -> None:
    frame_ids = (0, len(targets) // 2, len(targets) - 1)
    figure, axes = plt.subplots(len(frame_ids), len(models) + 1, figsize=(3 * (len(models) + 1), 9))
    for row, frame_id in enumerate(frame_ids):
        axes[row, 0].imshow(targets[frame_id], cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[row, 0].set_title(f"GT frame {frame_id}")
        axes[row, 0].axis("off")
        for column, model in enumerate(models, start=1):
            axes[row, column].imshow(
                np.clip(predictions[(model, frame_id)], 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[row, column].set_title(model)
            axes[row, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_comparison(config: ComparisonConfig) -> Path:
    started = time.monotonic()
    config = ComparisonConfig(
        **{
            **asdict(config),
            "workspace_root": Path(config.workspace_root).resolve(),
            "dataset_path": Path(config.dataset_path).resolve(),
            "ultra_dataset_dir": Path(config.ultra_dataset_dir).resolve(),
            "ultra_config_path": Path(config.ultra_config_path).resolve(),
            "checkpoints": {name: Path(path).resolve() for name, path in config.checkpoints.items()},
            "output_dir": Path(config.output_dir).resolve(),
        }
    )
    expected_models = {"NeUF_E0", "NeUF_E1", "NeUF_E2", "UltraNeRF_Hash"}
    if set(config.checkpoints) != expected_models:
        raise ValueError(f"Expected checkpoints {sorted(expected_models)}, got {sorted(config.checkpoints)}")
    for path in (config.dataset_path, config.ultra_config_path, *config.checkpoints.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = config.output_dir / "metrics"
    plot_dir = config.output_dir / "plots" / "step_final"
    checkpoint_hashes_before = {name: _sha256_file(path) for name, path in config.checkpoints.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The full comparison requires a qsub GPU allocation")
    lpips_model = _load_lpips(device)

    geometry, targets, render_ultra, loaded_step, remap = _load_ultra_runtime(config, device)
    if len(targets) != 242:
        raise ValueError(f"Expected 242 frames, got {len(targets)}")
    neuf_dataset = Dataset.open_from_save(config.dataset_path, map_location=device)
    refs = _all_frame_refs(neuf_dataset)
    if len(refs) != len(targets):
        raise ValueError(f"Dataset frame-count mismatch: NeUF={len(refs)}, Ultra={len(targets)}")
    common_mask = remap(metric_mask(neuf_dataset, erosion_px=3).astype(np.float32), geometry) >= 0.999
    if common_mask.shape != targets[0].shape or float(common_mask.mean()) < 0.85:
        raise ValueError(f"Unexpected common metric mask: shape={common_mask.shape}, fraction={common_mask.mean()}")

    target_differences = []
    for ref, target in zip(refs, targets):
        neuf_target = remap(_native_neuf_image(neuf_dataset, ref), geometry)
        target_differences.append(float(np.max(np.abs(neuf_target[common_mask] - target[common_mask]))))
    if max(target_differences) > 1.0 / 255.0 + 1e-6:
        raise ValueError(f"NeUF/Ultra GT mismatch: max abs difference={max(target_differences)}")

    roi_entries = []
    rois_by_frame: dict[int, list[dict[str, object]]] = {}
    for frame_id, target in enumerate(targets):
        pairs = _gt_roi_pairs(
            target,
            common_mask,
            roi_size=config.roi_size,
            pair_count=config.roi_pairs_per_frame,
            histogram_bins=config.histogram_bins,
        )
        rois_by_frame[frame_id] = pairs
        roi_entries.append({"frame_id": frame_id, "pairs": pairs, "target_sha256": _sha256_array(target)})
    roi_manifest = {
        "schema_version": 1,
        "selection_uses_model_outputs": False,
        "selection_source": "ground_truth_only",
        "rule": "GT low-pass homogeneous 24x24 windows; deterministic nearby pairs anchored at GT gCNR 0.35/0.60/0.85",
        "roi_size": config.roi_size,
        "pairs_per_frame": config.roi_pairs_per_frame,
        "histogram_edges": [0.0, 1.0],
        "histogram_bins": config.histogram_bins,
        "common_mask_sha256": _sha256_array(common_mask.astype(np.uint8)),
        "frames": roi_entries,
    }
    _write_json(config.output_dir / "manifests" / "gt_roi_manifest.json", roi_manifest)

    rows: list[dict[str, object]] = []
    representative: dict[tuple[str, int], np.ndarray] = {}
    representative_ids = {0, len(targets) // 2, len(targets) - 1}
    renderer = SliceRenderer(neuf_dataset)
    for model_name in ("NeUF_E0", "NeUF_E1", "NeUF_E2"):
        model = _load_neuf_model(config.checkpoints[model_name], neuf_dataset, device)
        for ref in tqdm(refs, desc=model_name, unit="frame", dynamic_ncols=False):
            frame_id = ref.original_frame_index
            prediction = remap(_render_neuf_native(model, renderer, neuf_dataset, ref), geometry)
            values = _frame_metrics(
                prediction,
                targets[frame_id],
                common_mask,
                rois_by_frame[frame_id],
                roi_size=config.roi_size,
                histogram_bins=config.histogram_bins,
                lpips_model=lpips_model,
                device=device,
            )
            rows.append({"model": model_name, "frame_id": frame_id, **values})
            if frame_id in representative_ids:
                representative[(model_name, frame_id)] = prediction
        del model
        gc.collect()
        torch.cuda.empty_cache()

    del renderer, neuf_dataset
    gc.collect()
    torch.cuda.empty_cache()
    for frame_id in tqdm(range(len(targets)), desc="UltraNeRF_Hash", unit="frame", dynamic_ncols=False):
        prediction = render_ultra(frame_id)
        values = _frame_metrics(
            prediction,
            targets[frame_id],
            common_mask,
            rois_by_frame[frame_id],
            roi_size=config.roi_size,
            histogram_bins=config.histogram_bins,
            lpips_model=lpips_model,
            device=device,
        )
        rows.append({"model": "UltraNeRF_Hash", "frame_id": frame_id, **values})
        if frame_id in representative_ids:
            representative[("UltraNeRF_Hash", frame_id)] = prediction

    summary_rows, summary_payload = _summary(rows)
    _write_csv(metric_dir / "per_frame_metrics.csv", rows)
    _write_csv(metric_dir / "summary.csv", summary_rows)
    _write_json(metric_dir / "summary.json", summary_payload)
    _save_representative_plot(
        plot_dir / "representative_frames.png",
        targets,
        representative,
        ["NeUF_E0", "NeUF_E1", "NeUF_E2", "UltraNeRF_Hash"],
    )
    checkpoint_hashes_after = {name: _sha256_file(path) for name, path in config.checkpoints.items()}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("A checkpoint changed during read-only evaluation")
    metadata = {
        "status": "complete",
        "frame_count": len(targets),
        "common_grid_shape_hw": list(targets[0].shape),
        "common_grid": "UltraNeRF convex training grid",
        "common_mask_valid_pixels": int(common_mask.sum()),
        "common_mask_valid_fraction": float(common_mask.mean()),
        "prediction_policy": "clip once to [0,1] before all four display-domain metrics; no per-frame normalization",
        "mse_aggregation": "mean of per-frame masked MSE; common mask has equal pixel count for every frame",
        "ssim_definition": "skimage Gaussian SSIM, win=7, sigma=1.5, K1=0.01, K2=0.03, masked map mean",
        "lpips_definition": "LPIPS AlexNet; masked tight crop, grayscale repeated to RGB, [0,1] mapped to [-1,1]",
        "gcnr_definition": "mean over three deterministic GT-only ROI pairs per frame; 64 fixed bins on [0,1]",
        "gcnr_interpretation": "closeness to target_gcnr is fidelity; larger prediction gCNR is not automatically better",
        "neuf_ultra_gt_max_abs_difference": max(target_differences),
        "ultra_loaded_step": loaded_step,
        "checkpoint_sha256": checkpoint_hashes_after,
        "elapsed_seconds": float(time.monotonic() - started),
        "device": str(device),
        "config": {
            **asdict(config),
            "workspace_root": str(config.workspace_root),
            "dataset_path": str(config.dataset_path),
            "ultra_dataset_dir": str(config.ultra_dataset_dir),
            "ultra_config_path": str(config.ultra_config_path),
            "checkpoints": {name: str(path) for name, path in config.checkpoints.items()},
            "output_dir": str(config.output_dir),
        },
    }
    _write_json(config.output_dir / "manifest.json", metadata)
    print(json.dumps(summary_rows, indent=2))
    return config.output_dir
