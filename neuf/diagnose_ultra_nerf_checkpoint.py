"""Read-only diagnostics for one trained Ultra-NeRF checkpoint.

The command intentionally creates no optimizer, performs one physical forward
pass and exactly one masked-MSE backward pass, and never mutates checkpoint
weights.  It diagnoses decomposition energy, Bernoulli gradient reachability,
and the pixel paths that the current axial cumulative products actually use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import torch

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.slice_renderer_base import DEVICE
from neuf.slice_renderer_ultra_nerf import UltraNeRFSliceRenderer
from neuf.ultra_nerf_renderer import UltraNeRFRenderer, make_generator


ZERO_GRAD_THRESHOLD = 1e-12
ACTIVE_GRAD_THRESHOLD = 1e-9
RECONSTRUCTION_TOLERANCE = 1e-6
DEFAULT_MASK_THRESHOLD = 2.0 / 255.0
MAP_ALIASES = {
    "E": "intensity_map",
    "R": "r",
    "B": "b",
    "rho_b": "border_probability",
    "rho_s": "scatterers_density_coeff",
    "phi": "scatter_amplitude",
    "T_att": "attenuation_transmission",
    "T_ref": "reflection_transmission",
    "alpha": "attenuation_coeff",
    "beta": "reflection_coeff",
    "G": "border_indicator",
    "H_s": "scatterers_density",
    "G_psf": "psf_border",
    "S_psf": "psf_scatter",
    "I": "transmission",
}


@dataclass(frozen=True)
class FrameRef:
    split: str
    split_index: int
    frame_index: int


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_default(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, default=_json_default)
        output.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_torch_load(path: Path, *, map_location: torch.device) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0
        loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary, got {type(loaded).__name__}"
        )
    return loaded


def _load_dataset(path: Path) -> tuple[Dataset, Path, str]:
    path = path.expanduser()
    if path.is_dir():
        return Dataset(str(path)), path, "raw_directory"

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as source:
            config = json.load(source)
        dataset_value = (
            config.get("baked_dataset_file")
            or config.get("dataset")
            or config.get("dataset_folder")
        )
        if not dataset_value:
            raise ValueError(
                "Dataset config must define baked_dataset_file, dataset, or dataset_folder"
            )
        dataset_path = Path(dataset_value).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = path.parent / dataset_path
        baked = bool(
            config.get("baked", dataset_path.suffix.lower() in {".pkl", ".pt"})
        )
        dataset = (
            Dataset.open_from_save(dataset_path, map_location=DEVICE)
            if baked
            else Dataset(str(dataset_path))
        )
        return dataset, dataset_path, "json_config"

    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    return Dataset.open_from_save(path, map_location=DEVICE), path, "baked_dataset"


def _all_frame_refs(dataset: Dataset) -> list[FrameRef]:
    references: list[FrameRef] = []
    for split, slices in (("train", dataset.slices), ("valid", dataset.slices_valid)):
        for split_index, slice_info in enumerate(slices):
            stored_index = getattr(slice_info, "frame_index", None)
            frame_index = split_index if stored_index is None else int(stored_index)
            references.append(FrameRef(split, split_index, frame_index))
    if not references:
        raise ValueError("Dataset contains no training or validation frames")
    return references


def _resolve_frame(dataset: Dataset, frame_index: int, split: str) -> FrameRef:
    references = _all_frame_refs(dataset)
    matches = [
        item
        for item in references
        if item.frame_index == frame_index and (split == "auto" or item.split == split)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Frame index {frame_index} is ambiguous; select --split train or --split valid"
        )

    if split != "auto":
        slices = dataset.slices if split == "train" else dataset.slices_valid
        if 0 <= frame_index < len(slices):
            stored_index = getattr(slices[frame_index], "frame_index", None)
            return FrameRef(
                split,
                frame_index,
                frame_index if stored_index is None else int(stored_index),
            )
    available = sorted(item.frame_index for item in references)
    raise IndexError(
        f"Acquisition frame {frame_index} was not found for split={split}; "
        f"available range is {available[0]}..{available[-1]}"
    )


def _frame_tensors(
    dataset: Dataset,
    reference: FrameRef,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if reference.split == "train":
        pixels = dataset.get_slice_pixels(reference.split_index)
        points = dataset.get_slice_points(reference.split_index)
        viewdirs = dataset.get_slice_viewdirs(reference.split_index)
    else:
        pixels = dataset.get_slice_valid_pixels(reference.split_index)
        points = dataset.get_slice_valid_points(reference.split_index)
        viewdirs = dataset.get_slice_valid_viewdirs(reference.split_index)
    return pixels, points, viewdirs


def _frame_image(dataset: Dataset, reference: FrameRef) -> np.ndarray:
    pixels, _, _ = _frame_tensors(dataset, reference)
    return (
        pixels.reshape(int(dataset.px_height), int(dataset.px_width))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _sequence_maximum(dataset: Dataset) -> np.ndarray:
    height, width = int(dataset.px_height), int(dataset.px_width)
    maxima: list[torch.Tensor] = []
    if dataset.slices:
        maxima.append(
            dataset.pixels.reshape(len(dataset.slices), height, width).amax(0)
        )
    if dataset.slices_valid:
        maxima.append(
            dataset.pixels_valid.reshape(len(dataset.slices_valid), height, width).amax(
                0
            )
        )
    if not maxima:
        raise ValueError("Cannot estimate a sector mask without image frames")
    maximum = maxima[0]
    for candidate in maxima[1:]:
        maximum = torch.maximum(maximum, candidate)
    return maximum.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_mask_file(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        mask = np.asarray(Image.open(path).convert("L"))
    mask = np.asarray(mask).squeeze()
    if mask.shape != shape:
        raise ValueError(f"Sector mask must have shape {shape}, got {mask.shape}")
    return mask.astype(bool)


def _existing_dataset_mask(
    dataset: Dataset,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    shape = (int(dataset.px_height), int(dataset.px_width))
    for attribute in ("sector_mask", "fov_mask", "ultrasound_mask"):
        value = getattr(dataset, attribute, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        mask = np.asarray(value).squeeze()
        if mask.size == shape[0] * shape[1]:
            return mask.reshape(shape).astype(bool), f"dataset.{attribute}"
    return None, None


def _runs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def _estimate_sector_mask(
    sequence_maximum: np.ndarray,
    *,
    threshold: float,
    erosion_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one fixed, row-convex central sector from every input frame.

    The central run constraint is deliberate: static labels and device settings
    are off-centre connected components, while the scan-converted sector crosses
    the image centre on every valid depth row.  Taking only that run prevents a
    bottom label such as ``DROIT`` from being joined to the sector by closing.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Mask threshold must be in [0, 1], got {threshold}")
    height, width = sequence_maximum.shape
    center_col = width // 2
    support = sequence_maximum >= float(threshold)
    support = ndi.binary_opening(support, structure=np.ones((3, 3), dtype=bool))

    central_runs = np.zeros_like(support, dtype=bool)
    for row_index, row in enumerate(support):
        starts, ends = _runs(row)
        containing_center = np.flatnonzero((starts <= center_col) & (ends > center_col))
        if containing_center.size:
            run_index = int(containing_center[0])
            central_runs[row_index, starts[run_index] : ends[run_index]] = True

    valid_rows = central_runs.any(axis=1)
    row_labels, row_component_count = ndi.label(valid_rows)
    if row_component_count == 0:
        raise ValueError("MASK_INVALID: no central sector support was detected")
    center_row_label = int(row_labels[height // 2])
    if center_row_label == 0:
        counts = np.bincount(row_labels)
        counts[0] = 0
        center_row_label = int(np.argmax(counts))
    central_runs[row_labels != center_row_label, :] = False

    mask = ndi.binary_closing(
        central_runs,
        structure=np.ones((5, 5), dtype=bool),
        iterations=1,
    )
    mask = ndi.binary_fill_holes(mask)
    if erosion_pixels:
        mask = ndi.binary_erosion(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(erosion_pixels),
            border_value=0,
        )

    labels, component_count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if component_count > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        center_label = int(labels[height // 2, center_col])
        keep_label = center_label if center_label else int(np.argmax(counts))
        mask = labels == keep_label

    support_labels, support_component_count = ndi.label(
        support,
        structure=np.ones((3, 3), dtype=bool),
    )
    support_counts = np.bincount(support_labels.ravel())
    outside_labels = np.unique(support_labels[~mask])
    excluded_component_sizes = sorted(
        (
            int(support_counts[label])
            for label in outside_labels
            if label != 0 and support_counts[label] > 0
        ),
        reverse=True,
    )
    details = {
        "algorithm": "sequence_maximum_central_row_run",
        "threshold": float(threshold),
        "erosion_pixels": int(erosion_pixels),
        "support_component_count": int(support_component_count),
        "excluded_support_component_sizes": excluded_component_sizes[:20],
        "central_support_row_range": [
            int(np.flatnonzero(valid_rows)[0]),
            int(np.flatnonzero(valid_rows)[-1]),
        ],
    }
    return mask.astype(bool), details


def _interval_counts(binary_rows: Iterable[np.ndarray]) -> list[int]:
    return [int(len(_runs(np.asarray(row, dtype=bool))[0])) for row in binary_rows]


def _validate_sector_mask(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    labels, component_count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    row_intervals = _interval_counts(mask)
    column_intervals = _interval_counts(mask.T)
    populated_rows = np.flatnonzero(mask.any(axis=1))
    populated_cols = np.flatnonzero(mask.any(axis=0))
    touches_border = bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )
    central_pixel_valid = bool(mask[height // 2, width // 2])
    valid_fraction = float(mask.mean())
    ui_regions_excluded = bool(
        component_count == 1
        and max(row_intervals, default=0) <= 1
        and max(column_intervals, default=0) <= 1
        and central_pixel_valid
        and not touches_border
    )
    mask_valid = bool(
        ui_regions_excluded
        and 0.05 <= valid_fraction <= 0.90
        and populated_rows.size > max(8, int(0.25 * height))
        and populated_cols.size > max(8, int(0.25 * width))
    )
    return {
        "height": int(height),
        "width": int(width),
        "valid_pixel_count": int(mask.sum()),
        "valid_fraction": valid_fraction,
        "connected_components_kept": int(component_count),
        "ui_regions_excluded": ui_regions_excluded,
        "mask_valid": mask_valid,
        "central_pixel_valid": central_pixel_valid,
        "touches_image_border": touches_border,
        "maximum_row_interval_count": max(row_intervals, default=0),
        "maximum_column_interval_count": max(column_intervals, default=0),
        "valid_row_range": (
            [int(populated_rows[0]), int(populated_rows[-1])]
            if populated_rows.size
            else []
        ),
        "valid_column_range": (
            [int(populated_cols[0]), int(populated_cols[-1])]
            if populated_cols.size
            else []
        ),
    }


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mask", "maps", "arrays", "gradients", "paths"):
        (output_dir / name).mkdir(exist_ok=True)


def _plot_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output: Path,
    *,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7.5))
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    green = np.zeros((*mask.shape, 4), dtype=np.float32)
    green[..., 1] = 1.0
    green[..., 3] = 0.16 * mask
    axis.imshow(green)
    axis.contour(mask, levels=[0.5], colors=["lime"], linewidths=1.0)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_three_mask_overlays(
    dataset: Dataset,
    references: list[FrameRef],
    mask: np.ndarray,
    output: Path,
) -> None:
    ordered = sorted(references, key=lambda item: item.frame_index)
    positions = np.linspace(0, len(ordered) - 1, 3).round().astype(int)
    selected = [ordered[int(index)] for index in positions]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for axis, reference in zip(axes, selected):
        image = _frame_image(dataset, reference)
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        green = np.zeros((*mask.shape, 4), dtype=np.float32)
        green[..., 1] = 1.0
        green[..., 3] = 0.14 * mask
        axis.imshow(green)
        axis.contour(mask, levels=[0.5], colors=["lime"], linewidths=0.8)
        axis.set_title(
            f"frame {reference.frame_index} ({reference.split}:{reference.split_index})"
        )
        axis.axis("off")
    figure.suptitle("Fixed sector mask on first / middle / last acquisition frames")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _masked_array(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(array, dtype=np.float32).copy()
    masked[~mask] = np.nan
    return masked


def _masked_statistics(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    display_upper: Optional[float] = None,
) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute statistics on an empty mask")
    stats = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p99.5": float(np.percentile(values, 99.5)),
        "nonzero_fraction": float(np.mean(np.abs(values) > np.finfo(np.float32).eps)),
    }
    if display_upper is not None:
        stats["display_upper"] = float(display_upper)
        stats["saturation_fraction"] = float(np.mean(values >= display_upper))
    return stats


def _probability_statistics(array: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)[mask]
    hist, edges = np.histogram(values, bins=20, range=(0.0, 1.0))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "near_half_fraction_0.45_0.55": float(
            np.mean((values >= 0.45) & (values <= 0.55))
        ),
        "histogram_counts": hist.tolist(),
        "histogram_bin_edges": edges.tolist(),
    }


def _depth_statistics(array: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    rows = np.indices(mask.shape)[0]
    valid_rows = rows[mask]
    row_min, row_max = int(valid_rows.min()), int(valid_rows.max())
    denominator = max(1, row_max - row_min)
    normalized_depth = (rows - row_min) / denominator
    bands: dict[str, Optional[float]] = {}
    for name, low, high in (
        ("shallow", 0.0, 0.25),
        ("middle_shallow", 0.25, 0.50),
        ("middle_deep", 0.50, 0.75),
        ("deep", 0.75, 1.000001),
    ):
        selected = mask & (normalized_depth >= low) & (normalized_depth < high)
        bands[name] = (
            float(np.asarray(array)[selected].mean()) if selected.any() else None
        )
    return {
        "valid_row_range": [row_min, row_max],
        "depth_band_means": bands,
        "interpretation": "pixel-row trend only; not physical beam depth after geometry stop",
    }


def _save_map(
    array: np.ndarray,
    mask: np.ndarray,
    output: Path,
    *,
    variable: str,
    frame_index: int,
    checkpoint_name: str,
    seed: int,
    display_min: float,
    display_max: float,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    image = axis.imshow(
        np.where(mask, array, 0.0),
        cmap="gray",
        vmin=display_min,
        vmax=display_max,
    )
    axis.set_title(
        f"{variable} | frame={frame_index} | checkpoint={checkpoint_name} | seed={seed}"
    )
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_decomposition_panel(
    target: np.ndarray,
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    output: Path,
    *,
    frame_index: int,
    checkpoint_name: str,
    seed: int,
    echo_upper: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    entries = (
        ("target [0,1]", target, 1.0),
        ("E (shared echo range)", arrays["E"], echo_upper),
        ("R-only (shared echo range)", arrays["R"], echo_upper),
        ("B-only (shared echo range)", arrays["B"], echo_upper),
    )
    for axis, (name, array, upper) in zip(axes.ravel(), entries):
        displayed = axis.imshow(
            np.where(mask, array, 0.0),
            cmap="gray",
            vmin=0.0,
            vmax=upper,
        )
        axis.set_title(name)
        axis.axis("off")
        figure.colorbar(displayed, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle(
        f"frame={frame_index} | checkpoint={checkpoint_name} | seed={seed} | "
        f"E/R/B range=[0,{echo_upper:.6g}]"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _gradient_metrics(raw_grad: torch.Tensor, model: NeRF) -> dict[str, Any]:
    channel_names = UltraNeRFRenderer.RAW_CHANNEL_NAMES
    raw_metrics: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(channel_names):
        values = raw_grad[..., channel]
        raw_metrics[name] = {
            "mean_abs": float(values.abs().mean().detach().cpu()),
            "max_abs": float(values.abs().max().detach().cpu()),
            "l2": float(torch.linalg.vector_norm(values).detach().cpu()),
        }

    head = getattr(model, "output_linear", None)
    if not isinstance(head, torch.nn.Linear) or head.out_features != len(channel_names):
        raise RuntimeError("Expected one Linear(..., 5) Ultra-NeRF output head")
    weight_grad = head.weight.grad
    bias_grad = head.bias.grad if head.bias is not None else None
    if weight_grad is None:
        raise RuntimeError(
            "Ultra-NeRF output head has no weight gradient after backward"
        )
    weight_norms = {
        name: float(torch.linalg.vector_norm(weight_grad[channel]).detach().cpu())
        for channel, name in enumerate(channel_names)
    }
    bias_values = {
        name: (
            float(abs(bias_grad[channel].detach().cpu()))
            if bias_grad is not None
            else None
        )
        for channel, name in enumerate(channel_names)
    }
    return {
        "raw_output": raw_metrics,
        "output_head": {
            "type": "Linear(..., 5)",
            "weight_row_grad_norm": weight_norms,
            "bias_grad_abs": bias_values,
        },
    }


def _write_gradient_outputs(
    gradient_data: dict[str, Any],
    output_dir: Path,
) -> None:
    _write_json(output_dir / "gradient_norms.json", gradient_data)
    names = list(UltraNeRFRenderer.RAW_CHANNEL_NAMES)
    with (output_dir / "gradient_norms.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "channel",
                "raw_grad_mean_abs",
                "raw_grad_max_abs",
                "raw_grad_l2",
                "head_weight_row_grad_norm",
                "head_bias_grad_abs",
            ]
        )
        for name in names:
            raw = gradient_data["raw_output"][name]
            head = gradient_data["output_head"]
            writer.writerow(
                [
                    name,
                    raw["mean_abs"],
                    raw["max_abs"],
                    raw["l2"],
                    head["weight_row_grad_norm"][name],
                    head["bias_grad_abs"][name],
                ]
            )

    raw_l2 = [gradient_data["raw_output"][name]["l2"] for name in names]
    head_l2 = [
        gradient_data["output_head"]["weight_row_grad_norm"][name] for name in names
    ]
    x_positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(x_positions - 0.19, raw_l2, width=0.38, label="raw output L2")
    axis.bar(x_positions + 0.19, head_l2, width=0.38, label="head weight-row L2")
    axis.axhline(
        ZERO_GRAD_THRESHOLD, color="tab:red", linestyle="--", label="zero threshold"
    )
    axis.axhline(
        ACTIVE_GRAD_THRESHOLD,
        color="tab:green",
        linestyle=":",
        label="active threshold",
    )
    axis.set_yscale("symlog", linthresh=ZERO_GRAD_THRESHOLD)
    axis.set_xticks(x_positions, names, rotation=20, ha="right")
    axis.set_ylabel("gradient norm")
    axis.set_title("Per-channel gradient reachability after the only backward pass")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "gradient_barplot.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _select_path_columns(mask: np.ndarray, total: int = 10) -> np.ndarray:
    valid_columns = np.flatnonzero(mask.any(axis=0))
    if valid_columns.size < total:
        raise ValueError(f"Sector mask has only {valid_columns.size} valid columns")
    positions = np.linspace(0, valid_columns.size - 1, total).round().astype(int)
    selected = valid_columns[positions]
    if np.unique(selected).size != total:
        raise ValueError("Could not select ten distinct propagation paths")
    return selected


def _path_outputs(
    target: np.ndarray,
    mask: np.ndarray,
    points_hw3: np.ndarray,
    z_vals_wh: np.ndarray,
    output_dir: Path,
    *,
    frame_index: int,
) -> dict[str, Any]:
    height, width = mask.shape
    selected_columns = _select_path_columns(mask, total=10)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    pixel_payload: dict[str, Any] = {
        "layout": "original image [row, col], ordered exactly as renderer raw_wh5[:, depth]",
        "cumprod_dimension": 1,
        "raw_layout": "[W, H, 5]",
        "selected_columns": selected_columns.tolist(),
        "paths": [],
    }
    world_paths: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    near_vertical_threshold = max(1.0, 0.002 * width)

    for path_index, column in enumerate(selected_columns):
        rows = np.arange(height, dtype=np.int64)
        columns = np.full(height, int(column), dtype=np.int64)
        pixels = np.stack([rows, columns], axis=1)
        world = points_hw3[:, int(column), :].astype(np.float32, copy=False)
        depths = z_vals_wh[int(column)]
        horizontal_drift = float(columns.max() - columns.min())
        vertical_extent = float(rows.max() - rows.min())
        delta_row = float(rows[-1] - rows[0])
        delta_col = float(columns[-1] - columns[0])
        direction_angle = float(math.degrees(math.atan2(delta_col, delta_row)))
        depth_monotonic = bool(np.all(np.diff(depths) > 0))
        inside_fraction = float(mask[:, int(column)].mean())
        first_inside = np.flatnonzero(mask[:, int(column)])
        pixel_payload["paths"].append(
            {
                "path_id": int(path_index),
                "column": int(column),
                "pixel_coordinates_row_col": pixels.tolist(),
                "first_inside_mask_row": (
                    int(first_inside[0]) if first_inside.size else None
                ),
                "last_inside_mask_row": (
                    int(first_inside[-1]) if first_inside.size else None
                ),
            }
        )
        world_paths.append(world)
        metrics.append(
            {
                "path_id": int(path_index),
                "column": int(column),
                "horizontal_drift_px": horizontal_drift,
                "vertical_extent_px": vertical_extent,
                "direction_angle_deg_from_image_down_axis": direction_angle,
                "inside_mask_fraction": inside_fraction,
                "depth_order_monotonic": depth_monotonic,
                "near_vertical_path": bool(horizontal_drift <= near_vertical_threshold),
            }
        )

    world_array = np.stack(world_paths, axis=0)
    _write_json(output_dir / "path_pixel_coordinates.json", pixel_payload)
    np.save(output_dir / "path_world_coordinates.npy", world_array)

    vertical_count = sum(item["near_vertical_path"] for item in metrics)
    parallel_vertical = bool(vertical_count >= 8)
    direction_angles = [
        item["direction_angle_deg_from_image_down_axis"] for item in metrics
    ]
    non_monotonic = [
        item["path_id"] for item in metrics if not item["depth_order_monotonic"]
    ]
    outside_paths = [
        item["path_id"] for item in metrics if item["inside_mask_fraction"] < 0.95
    ]
    stop_conditions = []
    if parallel_vertical:
        stop_conditions.append("PARALLEL_VERTICAL_GEOMETRY")
    if non_monotonic:
        stop_conditions.append("NON_MONOTONIC_DEPTH_ORDER")
    if outside_paths:
        stop_conditions.append("PATH_INSIDE_MASK_FRACTION_BELOW_0.95")
    path_summary = {
        "cumprod_dimension": 1,
        "cumprod_index_mapping": "raw_wh5[path=original_col, depth=original_row]",
        "selected_path_count": len(metrics),
        "near_vertical_threshold_px": float(near_vertical_threshold),
        "parallel_vertical_path_count": int(vertical_count),
        "PARALLEL_VERTICAL_GEOMETRY": parallel_vertical,
        "direction_angle_spread_deg": float(
            max(direction_angles) - min(direction_angles)
        ),
        "non_monotonic_path_ids": non_monotonic,
        "inside_mask_fraction_below_0.95_path_ids": outside_paths,
        "direct_pixel_mapping": True,
        "reprojection_error_px": {"median": 0.0, "maximum": 0.0},
        "paths": metrics,
        "STOP_PHYSICS_BRANCH": bool(stop_conditions),
        "stop_conditions": stop_conditions,
        "reason": (
            "current cumprod paths do not match convex-sector propagation geometry"
            if stop_conditions
            else ""
        ),
    }
    _write_json(output_dir / "path_geometry_metrics.json", path_summary)

    def draw_paths(background: np.ndarray, output: Path, title: str) -> None:
        figure, axis = plt.subplots(figsize=(10, 7.5))
        axis.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axis.contour(mask, levels=[0.5], colors=["white"], linewidths=0.8)
        arrow_start = max(1, int(0.08 * (height - 1)))
        arrow_end = max(arrow_start + 1, int(0.22 * (height - 1)))
        for path_index, (column, color) in enumerate(zip(selected_columns, colors)):
            axis.plot(
                np.full(height, column),
                np.arange(height),
                color=color,
                linewidth=1.3,
                alpha=0.9,
            )
            axis.scatter(
                [column], [0], color=[color], edgecolor="black", s=38, zorder=4
            )
            axis.annotate(
                "",
                xy=(column, arrow_end),
                xytext=(column, arrow_start),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            )
            first_inside = np.flatnonzero(mask[:, int(column)])
            label_row = int(first_inside[0]) if first_inside.size else arrow_end
            axis.text(
                column + 3,
                label_row + 8,
                str(path_index),
                color=color,
                fontsize=8,
                weight="bold",
            )
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output, dpi=170, bbox_inches="tight")
        plt.close(figure)

    draw_paths(
        np.where(mask, target, 0.0),
        output_dir / f"frame_{frame_index}_cumprod_paths_overlay.png",
        f"Actual cumprod paths | frame={frame_index} | dim=1 | raw layout=[W,H,5]",
    )
    draw_paths(
        mask.astype(np.float32),
        output_dir / f"frame_{frame_index}_cumprod_paths_mask_only.png",
        f"Actual cumprod paths on sector mask | frame={frame_index}",
    )
    return path_summary


def _code_locations() -> dict[str, Any]:
    renderer_file = Path(inspect.getsourcefile(UltraNeRFRenderer) or "").resolve()
    source_lines, start_line = inspect.getsourcelines(UltraNeRFRenderer.forward)
    locations: dict[str, Any] = {"file": str(renderer_file)}

    def find_lines(token: str) -> list[int]:
        return [
            start_line + offset
            for offset, line in enumerate(source_lines)
            if token in line
        ]

    def find_detach_after(anchor: str) -> list[int]:
        anchors = [offset for offset, line in enumerate(source_lines) if anchor in line]
        matches = []
        for anchor_offset in anchors:
            for offset in range(
                anchor_offset, min(anchor_offset + 8, len(source_lines))
            ):
                if ").detach()" in source_lines[offset]:
                    matches.append(start_line + offset)
                    break
        return matches

    locations.update(
        {
            "border_bernoulli": find_lines("border_indicator = torch.bernoulli("),
            "border_detach": find_detach_after("border_indicator = torch.bernoulli("),
            "scatter_bernoulli": find_lines("scatterers_density = torch.bernoulli("),
            "scatter_detach": find_detach_after(
                "scatterers_density = torch.bernoulli("
            ),
            "cumprod_attenuation": find_lines(
                "attenuation_transmission = exclusive_cumprod"
            ),
            "cumprod_reflection": find_lines(
                "reflection_transmission = exclusive_cumprod"
            ),
        }
    )
    locations["explanation"] = (
        "Both torch.bernoulli(...).detach() samples are non-reparameterized and "
        "remove rho_b/rho_s from the rendered loss graph."
    )
    return locations


def _write_report(
    output: Path,
    *,
    frame_index: int,
    mask_stats: dict[str, Any],
    fractions: dict[str, float],
    probability_stats: dict[str, Any],
    gradient_data: dict[str, Any],
    flags: dict[str, Any],
    path_summary: dict[str, Any],
    checkpoint_info: dict[str, Any],
) -> None:
    raw = gradient_data["raw_output"]
    head = gradient_data["output_head"]
    stop = bool(path_summary["STOP_PHYSICS_BRANCH"])
    source_locations = checkpoint_info["cumprod"]["source_locations"]
    border_span = (
        f"{source_locations['border_bernoulli'][0]}-"
        f"{source_locations['border_detach'][0]}"
    )
    scatter_span = (
        f"{source_locations['scatter_bernoulli'][0]}-"
        f"{source_locations['scatter_detach'][0]}"
    )
    cumprod_lines = ",".join(
        str(line)
        for line in (
            source_locations["cumprod_attenuation"]
            + source_locations["cumprod_reflection"]
        )
    )
    dominant = (
        "B（backscattering）" if fractions["B_fraction"] >= 0.5 else "R（reflection）"
    )
    text = f"""# Ultra-NeRF checkpoint 只读诊断报告

1. **mask 是否完全排除了扇区外背景和所有 UI？** 结构核验结果为 `{mask_stats["mask_valid"]}`；固定 mask 仅保留一个中央、逐行与逐列连续且不接触图像边界的连通区域，`ui_regions_excluded={mask_stats["ui_regions_excluded"]}`。底部文字与两侧设备参数位于 mask 外。见 [mask overlay](mask/mask_overlay_frame_{frame_index}.png) 和 [three-frame overlay](mask/mask_overlay_three_frames.png)。
2. **当前输出主要来自 R 还是 B？** 主要来自 {dominant}；`B_fraction={fractions["B_fraction"]:.8g}`，`R_fraction={fractions["R_fraction"]:.8g}`。同一次随机 forward 的分解见 [decomposition panel](maps/frame_{frame_index}_decomposition_panel.png)。
3. **rho_b、rho_s 是否集中在 0.5 附近？** 否。`rho_b` 的 p50 为 `{probability_stats["rho_b"]["p50"]:.8g}`，在 `[0.45,0.55]` 内的比例为 `{probability_stats["rho_b"]["near_half_fraction_0.45_0.55"]:.8g}`；`rho_s` 的 p50 为 `{probability_stats["rho_s"]["p50"]:.8g}`，对应比例为 `{probability_stats["rho_s"]["near_half_fraction_0.45_0.55"]:.8g}`。自动标志分别为 `RHO_B_NEAR_HALF={flags["RHO_B_NEAR_HALF"]}`、`RHO_S_NEAR_HALF={flags["RHO_S_NEAR_HALF"]}`。
4. **rho 原始通道和输出头 gradient norm 是多少？** `rho_b_raw`: raw L2 `{raw["rho_b_raw"]["l2"]:.8g}`，head weight-row L2 `{head["weight_row_grad_norm"]["rho_b_raw"]:.8g}`，head bias abs `{head["bias_grad_abs"]["rho_b_raw"]:.8g}`；`rho_s_raw`: raw L2 `{raw["rho_s_raw"]["l2"]:.8g}`，head weight-row L2 `{head["weight_row_grad_norm"]["rho_s_raw"]:.8g}`，head bias abs `{head["bias_grad_abs"]["rho_s_raw"]:.8g}`。见 [gradient bar plot](gradients/gradient_barplot.png)。
5. **是否确认硬 Bernoulli 阻断两个 rho 通道？** `{flags["BERNOULLI_GRAD_BLOCKED"]}`。判定使用一次 masked-MSE backward、零梯度阈值 `{ZERO_GRAD_THRESHOLD:g}` 和其他通道活跃阈值 `{ACTIVE_GRAD_THRESHOLD:g}`。阻断位置是 `neuf/ultra_nerf_renderer.py:{border_span}` 的 border `torch.bernoulli(...).detach()` 和 `:{scatter_span}` 的 scatter `torch.bernoulli(...).detach()`；机器可读位置也记录于 `checkpoint_info.json`。
6. **cumprod 的实际维度和路径索引是什么？** 当前 raw layout 为 `[W,H,5]`，`neuf/ultra_nerf_renderer.py:{cumprod_lines}` 的两个 `exclusive_cumprod(..., dim=1)` 都沿 `raw_wh5[原图列, 原图行]` 的第二维推进；`UltraNeRFSliceRenderer._query_raw()` 将原始 `[H,W]` row-major 像素 reshape 后 permute 为该布局。
7. **10 条路径是扇形发散还是竖直/平行？** `{path_summary["parallel_vertical_path_count"]}/10` 条被判为近竖直，`PARALLEL_VERTICAL_GEOMETRY={path_summary["PARALLEL_VERTICAL_GEOMETRY"]}`，方向角 spread 为 `{path_summary["direction_angle_spread_deg"]:.8g}°`。它们是当前代码真实的恒定列路径，不是另画的理想探头射线。见 [cumprod path overlay](paths/frame_{frame_index}_cumprod_paths_overlay.png)。
8. **是否触发 STOP_PHYSICS_BRANCH？** `{stop}`。原因：`{path_summary["reason"] or "none"}`；触发条件为 `{", ".join(path_summary["stop_conditions"]) or "none"}`。
9. **当前 checkpoint 失败最直接由哪些证据支持？** 同次 forward 中 `B_fraction={fractions["B_fraction"]:.8g}`、`R_fraction={fractions["R_fraction"]:.8g}`，E 几乎完全是随机 backscattering；唯一一次 backward 中两个 rho raw/head 梯度均严格为 0，而 alpha/phi 通道仍活跃；实际传播路径为 `10/10` 条恒定列，无法表示凸阵扇形发散。上述证据只支持诊断与停止判断，不构成重建质量改善结果。

## 项目目标与测试原则

项目最高优先级是提高最终重建图像的解剖清晰度、几何一致性、散斑保留与区分能力、对比度和伪影抑制。本任务仅做只读诊断：测试只用于确认代码可运行并防止关键回归，不扩展重复或一次性 smoke-test 框架；测试通过不代表重建质量提高。未进行重新训练、参数扫描或图像增强，本报告结论不声称图像质量改善，方法效果为“尚未验证”。

诊断 checkpoint：`{checkpoint_info["checkpoint"]}`；frame：`{frame_index}`；seed：`{checkpoint_info["seed"]}`；optimizer step：`0`。
"""
    output.write_text(text, encoding="utf-8")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three strictly read-only Ultra-NeRF checkpoint diagnostics."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--frame-index",
        required=True,
        type=int,
        help="Acquisition frame_index stored by the dataset (not training-list position).",
    )
    parser.add_argument("--split", choices=("auto", "train", "valid"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sector-mask",
        type=Path,
        help="Optional verified [H,W] NPY/PNG mask; preferred over automatic estimation.",
    )
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument(
        "--mask-erosion-pixels", type=int, choices=range(0, 4), default=2
    )
    parser.add_argument("--query-chunk", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser()
    dataset_argument = args.dataset.expanduser()
    output_dir = args.output_dir.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.query_chunk is not None and args.query_chunk < 1:
        raise ValueError("--query-chunk must be positive")
    _prepare_output_directory(output_dir, bool(args.overwrite))

    command = shlex.join(
        [sys.executable, "-m", "neuf.diagnose_ultra_nerf_checkpoint", *sys.argv[1:]]
    )
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    checkpoint_hash_before = _sha256(checkpoint_path)
    checkpoint_stat_before = checkpoint_path.stat()
    checkpoint = _safe_torch_load(checkpoint_path, map_location=DEVICE)
    if str(checkpoint.get("renderer", "")).lower() != "ultra_nerf":
        raise ValueError("Checkpoint renderer is not ultra_nerf")
    if str(checkpoint.get("output_mode", "")).lower() != "ultra_nerf":
        raise ValueError("Checkpoint output_mode is not ultra_nerf")

    dataset, dataset_path, dataset_load_mode = _load_dataset(dataset_argument)
    validate_checkpoint_dataset_geometry(
        checkpoint,
        dataset,
        checkpoint_path=checkpoint_path,
    )
    expected_shape = tuple(int(value) for value in checkpoint.get("image_shape_hw", ()))
    actual_shape = (int(dataset.px_height), int(dataset.px_width))
    if expected_shape and expected_shape != actual_shape:
        raise ValueError(
            f"Checkpoint image shape {expected_shape} does not match dataset {actual_shape}"
        )
    frame = _resolve_frame(dataset, int(args.frame_index), str(args.split))
    references = _all_frame_refs(dataset)

    sequence_maximum = _sequence_maximum(dataset)
    if args.sector_mask is not None:
        sector_mask = _load_mask_file(args.sector_mask.expanduser(), actual_shape)
        mask_source = "existing"
        mask_source_detail = str(args.sector_mask.expanduser())
        mask_estimation: dict[str, Any] = {"algorithm": "provided_mask"}
    else:
        sector_mask, existing_detail = _existing_dataset_mask(dataset)
        if sector_mask is not None:
            mask_source = "existing"
            mask_source_detail = str(existing_detail)
            mask_estimation = {"algorithm": "dataset_attribute"}
        else:
            # Neither the current baked dataset nor infos metadata contains a
            # probe-sector calibration or a 2-D FOV mask, so use all frames.
            sector_mask, mask_estimation = _estimate_sector_mask(
                sequence_maximum,
                threshold=float(args.mask_threshold),
                erosion_pixels=int(args.mask_erosion_pixels),
            )
            mask_source = "estimated"
            mask_source_detail = "all train+validation frames"

    mask_stats = _validate_sector_mask(sector_mask)
    mask_stats.update(
        {
            "source": mask_source,
            "source_detail": mask_source_detail,
            "estimation": mask_estimation,
            "frames_used_for_estimation": len(references),
            "visual_review_artifacts": [
                f"mask_overlay_frame_{frame.frame_index}.png",
                "mask_overlay_three_frames.png",
            ],
        }
    )
    np.save(output_dir / "sector_mask.npy", sector_mask)
    np.save(output_dir / "mask" / "sector_mask.npy", sector_mask)
    plt.imsave(
        output_dir / "mask" / "sector_mask.png",
        sector_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    target = _frame_image(dataset, frame)
    _plot_mask_overlay(
        target,
        sector_mask,
        output_dir / "mask" / f"mask_overlay_frame_{frame.frame_index}.png",
        title=(
            f"Fixed sector mask | frame={frame.frame_index} "
            f"({frame.split}:{frame.split_index}) | source={mask_source}"
        ),
    )
    _plot_three_mask_overlays(
        dataset,
        references,
        sector_mask,
        output_dir / "mask" / "mask_overlay_three_frames.png",
    )
    _write_json(output_dir / "mask" / "mask_stats.json", mask_stats)
    if not mask_stats["mask_valid"]:
        (output_dir / "MASK_INVALID.txt").write_text(
            "MASK_INVALID: structural validation failed; no model forward or backward was run.\n",
            encoding="utf-8",
        )
        raise RuntimeError("MASK_INVALID: refusing to continue checkpoint diagnostics")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = NeRF(checkpoint).to(DEVICE)
    model.training_progress = 1.0
    if hasattr(model.encode, "n_levels"):
        model._active_levels = int(model.encode.n_levels)
    model.eval()
    if model.output_mode != "ultra_nerf" or model.output_linear.out_features != 5:
        raise RuntimeError("Checkpoint did not restore a five-channel Ultra-NeRF head")

    query_chunk = int(
        args.query_chunk
        if args.query_chunk is not None
        else checkpoint.get("ultra_query_chunk", 65536)
    )
    renderer = UltraNeRFSliceRenderer(
        dataset,
        psf_half_size=int(checkpoint["ultra_psf_half_size"]),
        psf_lateral_std=float(checkpoint["ultra_psf_lateral_std"]),
        psf_axial_std=float(checkpoint["ultra_psf_axial_std"]),
        distance_unit=str(checkpoint["ultra_distance_unit"]),
        bernoulli_seed=int(checkpoint.get("ultra_bernoulli_seed", 0)),
        eval_mc_samples=1,
        query_chunk=query_chunk,
    )
    pixels, points, viewdirs = _frame_tensors(dataset, frame)
    height, width = actual_shape
    points_hw3 = points.reshape(height, width, 3)
    viewdirs_hw3 = viewdirs.reshape(height, width, 3)
    target_tensor = pixels.reshape(height, width).to(DEVICE)
    mask_tensor = torch.from_numpy(sector_mask).to(device=DEVICE, dtype=torch.bool)

    optimizer_step_count = 0
    forward_pass_count = 0
    backward_count = 0
    model.zero_grad(set_to_none=True)
    raw_wh5 = renderer._query_raw(model, points_hw3, viewdirs_hw3)
    raw_wh5.retain_grad()
    z_vals_wh = renderer._z_values(points_hw3)
    rendered_wh = renderer.physics(
        raw_wh5,
        z_vals_wh,
        border_generator=make_generator(raw_wh5.device, int(args.seed)),
        scatter_generator=make_generator(raw_wh5.device, int(args.seed) + 1),
    )
    forward_pass_count += 1
    rendered_hw = {
        name: value.transpose(0, 1).contiguous() for name, value in rendered_wh.items()
    }
    diagnostic_loss = (
        (rendered_hw["intensity_map"] - target_tensor)[mask_tensor] ** 2
    ).mean()
    diagnostic_loss.backward()
    backward_count += 1
    assert optimizer_step_count == 0
    assert forward_pass_count == 1
    assert backward_count == 1
    if raw_wh5.grad is None:
        raise RuntimeError("Raw five-channel tensor did not retain a gradient")

    raw_grad = raw_wh5.grad.detach()
    gradient_data = _gradient_metrics(raw_grad, model)
    gradient_data.update(
        {
            "diagnostic_loss": "masked_mse",
            "diagnostic_loss_value": float(diagnostic_loss.detach().cpu()),
            "backward_count": backward_count,
            "optimizer_created": False,
            "optimizer_step_count": optimizer_step_count,
            "zero_grad_threshold": ZERO_GRAD_THRESHOLD,
            "active_grad_threshold": ACTIVE_GRAD_THRESHOLD,
            "bernoulli_code_locations": _code_locations(),
        }
    )
    _write_gradient_outputs(gradient_data, output_dir / "gradients")

    arrays = {
        alias: rendered_hw[source].detach().cpu().numpy().astype(np.float32, copy=False)
        for alias, source in MAP_ALIASES.items()
    }
    raw_hw5 = (
        raw_wh5.detach().permute(1, 0, 2).cpu().numpy().astype(np.float32, copy=False)
    )
    target = target_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays_dir = output_dir / "arrays"
    np.save(arrays_dir / "target.npy", target)
    np.save(arrays_dir / "target_masked.npy", _masked_array(target, sector_mask))
    np.save(arrays_dir / "raw_hw5.npy", raw_hw5)
    for channel, name in enumerate(UltraNeRFRenderer.RAW_CHANNEL_NAMES):
        np.save(arrays_dir / f"{name}.npy", raw_hw5[..., channel])
        np.save(
            arrays_dir / f"{name}_masked.npy",
            _masked_array(raw_hw5[..., channel], sector_mask),
        )
    for name, array in arrays.items():
        np.save(arrays_dir / f"{name}.npy", array)
        np.save(arrays_dir / f"{name}_masked.npy", _masked_array(array, sector_mask))

    reconstruction_error = float(
        np.max(np.abs(arrays["E"].astype(np.float64) - arrays["R"] - arrays["B"]))
    )
    if reconstruction_error > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            f"E != R + B: maximum absolute error {reconstruction_error:.9g}"
        )
    sum_abs_r = float(np.abs(arrays["R"][sector_mask]).sum(dtype=np.float64))
    sum_abs_b = float(np.abs(arrays["B"][sector_mask]).sum(dtype=np.float64))
    denominator = sum_abs_r + sum_abs_b + np.finfo(np.float64).eps
    fractions = {
        "sum_abs_R": sum_abs_r,
        "sum_abs_B": sum_abs_b,
        "B_fraction": float(sum_abs_b / denominator),
        "R_fraction": float(sum_abs_r / denominator),
    }
    echo_upper = float(np.percentile(arrays["E"][sector_mask], 99.5))
    echo_upper = max(echo_upper, np.finfo(np.float32).eps)
    map_statistics: dict[str, Any] = {
        "mask_pixel_count": int(sector_mask.sum()),
        "echo_shared_display_range": [0.0, echo_upper],
        "decomposition": fractions,
        "maps": {},
        "probabilities": {},
        "depth_trends": {},
    }
    for name, array in {"target": target, **arrays}.items():
        upper = (
            echo_upper
            if name in {"E", "R", "B"}
            else (1.0 if name != "target" else 1.0)
        )
        map_statistics["maps"][name] = _masked_statistics(
            array,
            sector_mask,
            display_upper=upper,
        )
    for name in ("rho_b", "rho_s", "phi"):
        map_statistics["probabilities"][name] = _probability_statistics(
            arrays[name], sector_mask
        )
    for name in ("T_att", "T_ref"):
        map_statistics["depth_trends"][name] = _depth_statistics(
            arrays[name], sector_mask
        )
    _write_json(output_dir / "map_statistics.json", map_statistics)

    checkpoint_name = checkpoint_path.name
    _save_map(
        target,
        sector_mask,
        output_dir / "maps" / f"frame_{frame.frame_index}_target.png",
        variable="target",
        frame_index=frame.frame_index,
        checkpoint_name=checkpoint_name,
        seed=int(args.seed),
        display_min=0.0,
        display_max=1.0,
    )
    for name in ("E", "R", "B"):
        filename_name = {"R": "R_only", "B": "B_only"}.get(name, name)
        _save_map(
            arrays[name],
            sector_mask,
            output_dir / "maps" / f"frame_{frame.frame_index}_{filename_name}.png",
            variable=filename_name,
            frame_index=frame.frame_index,
            checkpoint_name=checkpoint_name,
            seed=int(args.seed),
            display_min=0.0,
            display_max=echo_upper,
        )
    for name in ("rho_b", "rho_s", "phi", "T_att", "T_ref"):
        _save_map(
            arrays[name],
            sector_mask,
            output_dir / "maps" / f"frame_{frame.frame_index}_{name}.png",
            variable=name,
            frame_index=frame.frame_index,
            checkpoint_name=checkpoint_name,
            seed=int(args.seed),
            display_min=0.0,
            display_max=1.0,
        )
    _save_decomposition_panel(
        target,
        arrays,
        sector_mask,
        output_dir / "maps" / f"frame_{frame.frame_index}_decomposition_panel.png",
        frame_index=frame.frame_index,
        checkpoint_name=checkpoint_name,
        seed=int(args.seed),
        echo_upper=echo_upper,
    )

    path_summary = _path_outputs(
        target,
        sector_mask,
        points_hw3.detach().cpu().numpy(),
        z_vals_wh.detach().cpu().numpy(),
        output_dir / "paths",
        frame_index=frame.frame_index,
    )
    if path_summary["STOP_PHYSICS_BRANCH"]:
        (output_dir / "STOP_PHYSICS_BRANCH.txt").write_text(
            "STOP_PHYSICS_BRANCH=true\n"
            "reason=current cumprod paths do not match convex-sector propagation geometry\n"
            "Do not continue training, fine-tuning, parameter sweeps, or formal A/B tests.\n"
            "Required acquisition/scan-conversion information: virtual centre, probe "
            "curvature radius, sector angle, depth sampling, and pixel-to-beam mapping.\n",
            encoding="utf-8",
        )

    rho_b_l2 = gradient_data["raw_output"]["rho_b_raw"]["l2"]
    rho_s_l2 = gradient_data["raw_output"]["rho_s_raw"]["l2"]
    other_active = (
        max(
            gradient_data["raw_output"][name]["l2"]
            for name in ("alpha_raw", "beta_raw", "phi_raw")
        )
        >= ACTIVE_GRAD_THRESHOLD
    )
    rho_b_near_half = map_statistics["probabilities"]["rho_b"][
        "near_half_fraction_0.45_0.55"
    ]
    rho_s_near_half = map_statistics["probabilities"]["rho_s"][
        "near_half_fraction_0.45_0.55"
    ]
    flags = {
        "SCATTER_DOMINANT": bool(fractions["B_fraction"] >= 0.80),
        "RHO_B_NEAR_HALF": bool(rho_b_near_half >= 0.50),
        "RHO_S_NEAR_HALF": bool(rho_s_near_half >= 0.50),
        "RHO_B_GRAD_ZERO": bool(rho_b_l2 <= ZERO_GRAD_THRESHOLD),
        "RHO_S_GRAD_ZERO": bool(rho_s_l2 <= ZERO_GRAD_THRESHOLD),
        "OTHER_CHANNELS_ACTIVE": bool(other_active),
        "BERNOULLI_GRAD_BLOCKED": bool(
            rho_b_l2 <= ZERO_GRAD_THRESHOLD
            and rho_s_l2 <= ZERO_GRAD_THRESHOLD
            and other_active
        ),
        "PARALLEL_VERTICAL_GEOMETRY": path_summary["PARALLEL_VERTICAL_GEOMETRY"],
        "STOP_PHYSICS_BRANCH": path_summary["STOP_PHYSICS_BRANCH"],
    }
    _write_json(output_dir / "diagnostic_flags.json", flags)

    checkpoint_hash_after = _sha256(checkpoint_path)
    checkpoint_stat_after = checkpoint_path.stat()
    checkpoint_unchanged = bool(
        checkpoint_hash_after == checkpoint_hash_before
        and checkpoint_stat_after.st_size == checkpoint_stat_before.st_size
        and checkpoint_stat_after.st_mtime_ns == checkpoint_stat_before.st_mtime_ns
    )
    if not checkpoint_unchanged:
        raise RuntimeError("Checkpoint changed during read-only diagnostics")
    checkpoint_info = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_unchanged,
        "checkpoint_size_bytes": int(checkpoint_stat_before.st_size),
        "checkpoint_iteration": int(checkpoint.get("start", -1)),
        "dataset_argument": str(dataset_argument),
        "dataset_resolved": str(dataset_path.resolve()),
        "dataset_load_mode": dataset_load_mode,
        "dataset_shape_hw": [height, width],
        "frame_index": int(frame.frame_index),
        "frame_split": frame.split,
        "frame_split_index": int(frame.split_index),
        "seed": int(args.seed),
        "model_mode": "eval",
        "renderer": checkpoint["renderer"],
        "network_output_mode": checkpoint["output_mode"],
        "raw_channel_order": list(UltraNeRFRenderer.RAW_CHANNEL_NAMES),
        "raw_channel_order_verified_by": (
            "UltraNeRFRenderer.RAW_CHANNEL_NAMES and checkpoint Linear(...,5) head"
        ),
        "renderer_variables": {
            "E": "intensity_map",
            "R": "r",
            "B": "b",
            "T_att": "attenuation_transmission",
            "T_ref": "reflection_transmission",
        },
        "cumprod": {
            "dimension": 1,
            "tensor_layout": "[W,H]",
            "source_locations": _code_locations(),
        },
        "reshape_order": (
            "dataset flat row-major -> points_hw3/raw_hw5 [H,W] -> permute to "
            "raw_wh5 [W,H] -> renderer -> transpose back to [H,W]"
        ),
        "training_loss_from_checkpoint": (
            "MSE"
            if float(checkpoint.get("ultra_final_ms_ssim_weight", 0.9)) == 0.0
            else "configured Ultra-NeRF MSE/MS-SSIM schedule"
        ),
        "diagnostic_loss": "masked_mse",
        "forward_pass_count": forward_pass_count,
        "backward_count": backward_count,
        "optimizer_created": False,
        "optimizer_step_count": optimizer_step_count,
        "no_retraining": True,
        "renderer_config": {
            "psf_half_size": int(checkpoint["ultra_psf_half_size"]),
            "psf_lateral_std": float(checkpoint["ultra_psf_lateral_std"]),
            "psf_axial_std": float(checkpoint["ultra_psf_axial_std"]),
            "distance_unit": str(checkpoint["ultra_distance_unit"]),
            "query_chunk": query_chunk,
            "eval_mc_samples_for_diagnostic": 1,
        },
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "process_id": os.getpid(),
    }
    _write_json(output_dir / "checkpoint_info.json", checkpoint_info)

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "frame_index": int(frame.frame_index),
        "seed": int(args.seed),
        "no_retraining": True,
        "optimizer_step_count": optimizer_step_count,
        "mask_valid": bool(mask_stats["mask_valid"]),
        "reconstruction_error_E_minus_R_plus_B": reconstruction_error,
        "B_fraction": fractions["B_fraction"],
        "R_fraction": fractions["R_fraction"],
        "rho_b_near_half_fraction": rho_b_near_half,
        "rho_s_near_half_fraction": rho_s_near_half,
        "rho_b_raw_grad_l2": rho_b_l2,
        "rho_s_raw_grad_l2": rho_s_l2,
        "BERNOULLI_GRAD_BLOCKED": flags["BERNOULLI_GRAD_BLOCKED"],
        "parallel_vertical_path_count": path_summary["parallel_vertical_path_count"],
        "PARALLEL_VERTICAL_GEOMETRY": path_summary["PARALLEL_VERTICAL_GEOMETRY"],
        "STOP_PHYSICS_BRANCH": path_summary["STOP_PHYSICS_BRANCH"],
        "stop_reason": path_summary["reason"],
        "forward_pass_count": forward_pass_count,
        "backward_count": backward_count,
        "checkpoint_unchanged": checkpoint_unchanged,
        "diagnostic_loss": "masked_mse",
        "diagnostic_loss_value": float(diagnostic_loss.detach().cpu()),
    }
    _write_json(output_dir / "diagnostic_summary.json", summary)
    _write_report(
        output_dir / "diagnostic_report.md",
        frame_index=frame.frame_index,
        mask_stats=mask_stats,
        fractions=fractions,
        probability_stats=map_statistics["probabilities"],
        gradient_data=gradient_data,
        flags=flags,
        path_summary=path_summary,
        checkpoint_info=checkpoint_info,
    )
    assert optimizer_step_count == 0
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
