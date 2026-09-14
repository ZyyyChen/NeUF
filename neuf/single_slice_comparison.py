from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.phase1_data import (
    apply_single_slice_training_view,
    freeze_single_slice_manifest,
    hash_array,
    hash_file,
)
from neuf.slice_renderer import SliceRenderer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ORDER = ("E0", "E1", "E3 alpha=0", "E3 alpha=1")
ROI_ZONES = (
    ("shallow", (0.16, 0.40), (0.30, 0.70), (0.28, 0.50)),
    ("middle-left", (0.34, 0.66), (0.16, 0.50), (0.50, 0.34)),
    ("middle-right", (0.34, 0.66), (0.50, 0.84), (0.50, 0.66)),
    ("deep", (0.58, 0.90), (0.25, 0.75), (0.74, 0.50)),
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _freeze_json(path: Path, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Frozen comparison manifest changed: {path}")
        return
    _write_json(path, payload)


def _load_slice(dataset_path: Path, output_dir: Path, frame_index: int):
    dataset = Dataset.open_from_save(dataset_path)
    splits = freeze_single_slice_manifest(
        dataset,
        dataset_path,
        output_dir / "run_config",
        frame_index,
    )
    apply_single_slice_training_view(dataset, splits)
    target = (
        dataset.get_slice_valid_pixels(0)
        .reshape(dataset.px_height, dataset.px_width)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    mask = dataset.get_sector_mask().detach().cpu().numpy().astype(bool)
    return dataset, splits["training"][0], target, mask


def _positions(limit: int, stride: int) -> list[int]:
    values = list(range(0, limit + 1, stride))
    if values[-1] != limit:
        values.append(limit)
    return values


def _overlaps(first: dict, second: dict) -> bool:
    size = int(first["size"])
    other_size = int(second["size"])
    return not (
        int(first["left"]) + size <= int(second["left"])
        or int(second["left"]) + other_size <= int(first["left"])
        or int(first["top"]) + size <= int(second["top"])
        or int(second["top"]) + other_size <= int(first["top"])
    )


def _select_rois(target: np.ndarray, mask: np.ndarray, size: int) -> list[dict]:
    """从四个预定义空间区间选结构梯度较强的窗口，且不查看模型输出。"""
    height, width = target.shape
    if size < 8 or size > min(height, width):
        raise ValueError(f"Invalid ROI size {size} for image shape {target.shape}")
    gradient_y, gradient_x = np.gradient(target.astype(np.float64))
    gradient = np.hypot(gradient_y, gradient_x)
    candidates = []
    for top in _positions(height - size, max(8, size // 8)):
        for left in _positions(width - size, max(8, size // 8)):
            window = np.s_[top : top + size, left : left + size]
            if not mask[window].all():
                continue
            candidates.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "size": int(size),
                    "center_y": (top + size / 2) / height,
                    "center_x": (left + size / 2) / width,
                    "observed_gradient_rms": float(np.sqrt(np.mean(gradient[window] ** 2))),
                }
            )
    if len(candidates) < len(ROI_ZONES):
        raise ValueError(f"Only {len(candidates)} fully valid ROI candidates are available")

    selected = []
    for label, y_range, x_range, anchor in ROI_ZONES:
        eligible = [
            item
            for item in candidates
            if y_range[0] <= item["center_y"] <= y_range[1]
            and x_range[0] <= item["center_x"] <= x_range[1]
            and not any(_overlaps(item, previous) for previous in selected)
        ]
        pool = eligible or [
            item
            for item in candidates
            if not any(_overlaps(item, previous) for previous in selected)
        ]
        if not pool:
            raise ValueError("Could not select four non-overlapping ROIs")
        choice = min(
            pool,
            key=lambda item: (
                -item["observed_gradient_rms"],
                (item["center_y"] - anchor[0]) ** 2 + (item["center_x"] - anchor[1]) ** 2,
                item["top"],
                item["left"],
            ),
        )
        selected.append(
            {
                "label": label,
                "top": choice["top"],
                "left": choice["left"],
                "size": choice["size"],
                "observed_gradient_rms": choice["observed_gradient_rms"],
            }
        )
    return selected


def _comparison_manifest(
    dataset_path: Path,
    frame_ref,
    target: np.ndarray,
    mask: np.ndarray,
    rois: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "comparison_indices": [frame_ref.stable_slice_id],
        "original_frame_index": frame_ref.original_frame_index,
        "source_pool": frame_ref.source,
        "source_index": frame_ref.source_index,
        "target_sha256": hash_array(target),
        "mask_sha256": hash_array(mask.astype(np.uint8)),
        "shape_hw": list(target.shape),
        "selection": (
            f"Before training, select one fully masked {rois[0]['size']}px window from "
            "each fixed spatial zone by observed-image gradient RMS; model predictions "
            "are not used."
        ),
        "rois": rois,
        "display_range": [0.0, 1.0],
        "error_display_range": [0.0, 0.1],
        "orientation": "native observed transverse slice",
        "normalization": "shared dataset display domain; no per-image or per-ROI rescaling",
        "mask": "dataset ultrasound sector",
        "evaluation_scope": (
            "The training slice is reused for fitting diagnostics; no held-out or "
            "three-dimensional generalization claim."
        ),
    }


def prepare(dataset_path: Path, output_dir: Path, frame_index: int, roi_size: int):
    dataset, frame_ref, target, mask = _load_slice(dataset_path, output_dir, frame_index)
    rois = _select_rois(target, mask, roi_size)
    manifest = _comparison_manifest(dataset_path, frame_ref, target, mask, rois)
    _freeze_json(output_dir / "run_config" / "comparison_manifest.json", manifest)
    return dataset, target, mask, manifest


def _render_checkpoint(
    checkpoint_path: Path,
    dataset: Dataset,
    frame_index: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    validate_checkpoint_dataset_geometry(checkpoint, dataset, checkpoint_path=checkpoint_path)
    if checkpoint.get("single_slice_index") != frame_index:
        raise ValueError(f"Checkpoint does not belong to single slice {frame_index}: {checkpoint_path}")
    if checkpoint.get("phase1_split_ids") != dataset.phase1_split_ids:
        raise ValueError(f"Checkpoint slice identity differs from comparison data: {checkpoint_path}")

    model = NeRF(checkpoint).eval()
    model.training_progress = 1.0
    renderer = SliceRenderer(dataset)
    # 展平成 [N]，避免与 [N] 预测相乘时广播成不可接受的 [N,N]。
    mask = dataset.get_sector_mask(flatten=True, device=DEVICE).reshape(-1)
    with torch.no_grad():
        if model.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
            components = renderer.query_point_components(
                model,
                dataset.get_slice_valid_points(0),
                dataset.get_slice_valid_viewdirs(0),
                alpha=1.0,
            )
            outputs = {
                "E3 alpha=0": components["anatomy"],
                "E3 alpha=1": components["intensity"],
            }
        else:
            outputs = {
                "intensity": renderer.render_slice_from_dataset_valid(
                    model,
                    0,
                    reshaped=False,
                    alpha=1.0,
                )
            }
    height, width = dataset.px_height, dataset.px_width
    arrays = {
        name: (values.reshape(-1) * mask).reshape(height, width).detach().cpu().numpy()
        for name, values in outputs.items()
    }
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": hash_file(checkpoint_path),
        "field_head": model.field_head,
        "training_steps": int(checkpoint.get("start", 0)),
        "training_loss": checkpoint.get("training_loss"),
    }
    del model, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, arrays


def _gradient_rms(image: np.ndarray, mask: np.ndarray) -> float:
    gradients = []
    for axis in (0, 1):
        difference = np.diff(image, axis=axis)
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        pair_mask = mask[tuple(first)] & mask[tuple(second)]
        gradients.append(difference[pair_mask])
    values = np.concatenate(gradients)
    return float(np.sqrt(np.mean(values**2)))


def _metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    difference = prediction[mask] - target[mask]
    mse = float(np.mean(difference**2))
    return {
        "valid_pixels": int(mask.sum()),
        "mse_to_observed": mse,
        "mae_to_observed": float(np.mean(np.abs(difference))),
        "psnr_to_observed_db": float("inf") if mse == 0 else float(-10 * math.log10(mse)),
        "gradient_rms": _gradient_rms(prediction, mask),
    }


def _metric_rows(
    target: np.ndarray,
    mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    rois: list[dict],
) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "region": "full_sector",
                "roi_label": "",
                "top": "",
                "left": "",
                "size": "",
                "model": model,
                **_metrics(target, predictions[model], mask),
            }
        )
    for roi in rois:
        top, left, size = (int(roi[name]) for name in ("top", "left", "size"))
        window = np.s_[top : top + size, left : left + size]
        roi_mask = mask[window]
        for model in MODEL_ORDER:
            rows.append(
                {
                    "region": "fixed_roi",
                    "roi_label": roi["label"],
                    "top": top,
                    "left": left,
                    "size": size,
                    "model": model,
                    **_metrics(target[window], predictions[model][window], roi_mask),
                }
            )
    return rows


def _shown(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, image, np.nan)


def _save_plots(
    output_dir: Path,
    frame_index: int,
    target: np.ndarray,
    mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    rois: list[dict],
    labels: dict[str, str],
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colours = ("#ffbf00", "#00bcd4", "#e91e63", "#7cb342")
    panels = [("Observed / training target", target)] + [
        (labels[name], predictions[name]) for name in MODEL_ORDER
    ]

    figure, axes = plt.subplots(1, len(panels), figsize=(21, 4.8), squeeze=False)
    for axis, (title, image) in zip(axes[0], panels):
        axis.imshow(_shown(image, mask), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for colour, roi in zip(colours, rois):
            axis.add_patch(
                Rectangle(
                    (roi["left"], roi["top"]),
                    roi["size"],
                    roi["size"],
                    fill=False,
                    edgecolor=colour,
                    linewidth=1.5,
                )
            )
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"Patient 0 / frame {frame_index} / single-slice fitting only / shared [0,1]"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    overview = plot_dir / f"slice_{frame_index}_full_comparison.png"
    figure.savefig(overview, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(rois), len(panels), figsize=(15, 12), squeeze=False)
    for row, (roi, colour) in enumerate(zip(rois, colours)):
        top, left, size = (int(roi[name]) for name in ("top", "left", "size"))
        window = np.s_[top : top + size, left : left + size]
        for column, (title, image) in enumerate(panels):
            axes[row, column].imshow(
                image[window], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
            if row == 0:
                axes[row, column].set_title(title)
            axes[row, column].axis("off")
        axes[row, 0].text(
            0.02,
            0.98,
            f"{roi['label']}\nr={top}, c={left}",
            transform=axes[row, 0].transAxes,
            va="top",
            color=colour,
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    figure.suptitle(
        f"Four fixed {rois[0]['size']}×{rois[0]['size']} ROIs / "
        "identical coordinates and intensity scale"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    roi_path = plot_dir / f"slice_{frame_index}_four_roi_comparison.png"
    figure.savefig(roi_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(rois), len(MODEL_ORDER), figsize=(12, 12), squeeze=False)
    error_image = None
    for row, (roi, colour) in enumerate(zip(rois, colours)):
        top, left, size = (int(roi[name]) for name in ("top", "left", "size"))
        window = np.s_[top : top + size, left : left + size]
        for column, model in enumerate(MODEL_ORDER):
            error = np.abs(predictions[model][window] - target[window])
            error_image = axes[row, column].imshow(
                error, cmap="magma", vmin=0, vmax=0.1, interpolation="nearest"
            )
            if row == 0:
                axes[row, column].set_title(labels[model])
            axes[row, column].axis("off")
        axes[row, 0].text(
            0.02,
            0.98,
            roi["label"],
            transform=axes[row, 0].transAxes,
            va="top",
            color=colour,
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    figure.suptitle("Absolute error to observed training slice / shared [0,0.1]")
    figure.colorbar(error_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    error_path = plot_dir / f"slice_{frame_index}_four_roi_absolute_errors.png"
    figure.savefig(error_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [str(path.resolve()) for path in (overview, roi_path, error_path)]


def compare(args) -> dict:
    output_dir = Path(args.output_dir).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    dataset, target, mask, manifest = prepare(
        dataset_path,
        output_dir,
        args.frame_index,
        args.roi_size,
    )
    if args.prepare_only:
        summary = {
            "status": "prepared",
            "frame_index": args.frame_index,
            "stable_slice_id": manifest["comparison_indices"][0],
            "roi_count": len(manifest["rois"]),
            "manifest": str((output_dir / "run_config" / "comparison_manifest.json").resolve()),
        }
        print(json.dumps(summary, indent=2))
        return summary

    paths = {
        "E0": Path(args.e0_checkpoint).expanduser(),
        "E1": Path(args.e1_checkpoint).expanduser(),
        "E3": Path(args.e3_checkpoint).expanduser(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Missing comparison checkpoint: {paths}")
    e0_metadata, e0_outputs = _render_checkpoint(paths["E0"], dataset, args.frame_index)
    e1_metadata, e1_outputs = _render_checkpoint(paths["E1"], dataset, args.frame_index)
    e3_metadata, e3_outputs = _render_checkpoint(paths["E3"], dataset, args.frame_index)
    predictions = {
        "E0": e0_outputs["intensity"],
        "E1": e1_outputs["intensity"],
        **e3_outputs,
    }
    if tuple(predictions) != MODEL_ORDER:
        raise RuntimeError(f"Unexpected comparison model order: {tuple(predictions)}")

    checkpoints = {"E0": e0_metadata, "E1": e1_metadata, "E3": e3_metadata}
    _write_json(output_dir / "run_config" / "checkpoint_manifest.json", checkpoints)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_index": args.frame_index,
            "stable_slice_id": manifest["comparison_indices"][0],
            "target": torch.from_numpy(target.copy()),
            "mask": torch.from_numpy(mask.copy()),
            **{name: torch.from_numpy(value.copy()) for name, value in predictions.items()},
        },
        prediction_dir / f"slice_{args.frame_index}_predictions.pt",
    )

    rows = _metric_rows(target, mask, predictions, manifest["rois"])
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_path = metrics_dir / f"slice_{args.frame_index}_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    e0_steps = e0_metadata["training_steps"]
    e1_steps = e1_metadata["training_steps"]
    e3_steps = e3_metadata["training_steps"]
    labels = {
        "E0": f"E0 / {e0_steps} updates",
        "E1": f"E1 / {e1_steps} updates",
        "E3 alpha=0": f"E3 SVT α=0 / {e1_steps}+{e3_steps}",
        "E3 alpha=1": f"E3 SVT α=1 / {e1_steps}+{e3_steps}",
    }
    figures = _save_plots(
        output_dir,
        args.frame_index,
        target,
        mask,
        predictions,
        manifest["rois"],
        labels,
    )
    summary = {
        "status": "complete",
        "frame_index": args.frame_index,
        "stable_slice_id": manifest["comparison_indices"][0],
        "training_scope": "one slice only",
        "evaluation_scope": "same training slice only",
        "e3_training_budget": f"E1 {e1_steps} updates followed by SVT {e3_steps} updates",
        "figures": figures,
        "metrics": str(metric_path.resolve()),
        "limitations": [
            "The observed noisy slice is the training target, not a clean ground truth.",
            "Metrics measure fitting of frame 216 and do not measure 3D generalization.",
            "E3 receives E1 initialization plus an additional SVT training stage.",
        ],
    }
    _write_json(metrics_dir / "comparison_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare NeUF models on one fixed training slice")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-index", type=int, default=216)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--e0-checkpoint", default="")
    parser.add_argument("--e1-checkpoint", default="")
    parser.add_argument("--e3-checkpoint", default="")
    return parser.parse_args(argv)


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
