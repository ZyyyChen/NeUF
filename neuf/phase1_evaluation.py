from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
GAUSSIAN_CONTROL_SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5)
BOOTSTRAP_SEED = 3407001


@dataclass(frozen=True)
class PredictionRecord:
    model: str
    seed: int
    slice_id: str
    split: str
    alpha: float
    prediction_raw: np.ndarray
    ground_truth: np.ndarray
    speckle: np.ndarray | None = None


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def eroded_finite_mask(
    sector_mask: np.ndarray,
    *images: np.ndarray,
    erosion_px: int = 3,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.asarray(sector_mask, dtype=bool)
    if erosion_px:
        mask = binary_erosion(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=erosion_px,
            border_value=0,
        )
    for image in images:
        mask = np.logical_and(mask, np.isfinite(image))
    if exclusion_mask is not None:
        mask = np.logical_and(mask, ~np.asarray(exclusion_mask, dtype=bool))
    if not mask.any():
        raise ValueError("Metric mask contains no finite valid pixels")
    return mask


def mask_normalized_gaussian(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    numerator = gaussian_filter(image * weights, sigma=sigma, mode="constant", cval=0.0)
    denominator = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)
    return numerator / np.maximum(denominator, epsilon)


def masked_ssim(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    ssim_result = structural_similarity(
        np.asarray(target, dtype=np.float64),
        np.asarray(predicted, dtype=np.float64),
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
        win_size=11,
    )
    ssim_map = ssim_result[1]
    return float(np.mean(ssim_map[mask]))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return np.hypot(sobel(image, axis=1, mode="nearest"), sobel(image, axis=0, mode="nearest"))


def gms(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    predicted_gradient = gradient_magnitude(predicted)
    target_gradient = gradient_magnitude(target)
    c = 0.01 ** 2
    similarity = (
        2.0 * predicted_gradient * target_gradient + c
    ) / (np.square(predicted_gradient) + np.square(target_gradient) + c)
    return float(np.mean(similarity[mask]))


def high_frequency_energy(image: np.ndarray, mask: np.ndarray) -> float:
    low = mask_normalized_gaussian(image, mask, sigma=2.0)
    return float(np.mean(np.square(np.asarray(image) - low)[mask]))


def slice_metrics(
    prediction_raw: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    reference_i1: np.ndarray | None = None,
    speckle: np.ndarray | None = None,
) -> dict[str, float]:
    raw = np.asarray(prediction_raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction = np.clip(raw, 0.0, 1.0)
    error = prediction - target
    mse = float(np.mean(np.square(error)[mask]))
    target_low = mask_normalized_gaussian(target, mask, sigma=2.0)
    prediction_low = mask_normalized_gaussian(prediction, mask, sigma=2.0)
    metrics = {
        "mae": float(np.mean(np.abs(error)[mask])),
        "mse": mse,
        "psnr": float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse)),
        "ssim": masked_ssim(prediction, target, mask),
        "lp_ssim_gt": masked_ssim(prediction_low, target_low, mask),
        "gms_gt": gms(prediction, target, mask),
        "hf": high_frequency_energy(prediction, mask),
        "brightness": float(np.mean(prediction[mask])),
        "below_zero_fraction": float(np.mean(raw[mask] < 0.0)),
        "above_one_fraction": float(np.mean(raw[mask] > 1.0)),
        "out_of_range_fraction": float(np.mean((raw[mask] < 0.0) | (raw[mask] > 1.0))),
    }
    if reference_i1 is not None:
        reference = np.clip(np.asarray(reference_i1, dtype=np.float64), 0.0, 1.0)
        reference_low = mask_normalized_gaussian(reference, mask, sigma=2.0)
        metrics["lp_ssim_i1"] = masked_ssim(prediction_low, reference_low, mask)
        metrics["brightness_drift_i1"] = abs(
            metrics["brightness"] - float(np.mean(reference[mask]))
        )
    else:
        metrics["lp_ssim_i1"] = float("nan")
        metrics["brightness_drift_i1"] = float("nan")
    if speckle is not None:
        speckle = np.asarray(speckle, dtype=np.float64)
        speckle_low = mask_normalized_gaussian(speckle, mask, sigma=2.0)
        metrics["leak_low"] = float(
            np.mean(np.square(speckle_low)[mask])
            / (np.mean(np.square(speckle)[mask]) + 1e-8)
        )
    else:
        metrics["leak_low"] = float("nan")
    return metrics


def roi_metrics(image: np.ndarray, roi: dict, *, epsilon: float = 1e-8) -> dict[str, float]:
    row = int(roi["row"])
    column = int(roi["column"])
    size = int(roi["size"])
    values = np.asarray(image, dtype=np.float64)[row:row + size, column:column + size]
    mean = float(values.mean())
    variance = float(values.var())
    return {
        "mean": mean,
        "std": float(values.std()),
        "sc": float(values.std() / (mean + epsilon)),
        "enl": float(mean ** 2 / (variance + epsilon)),
    }


def build_roi_manifest(
    ground_truth_by_slice: dict[str, np.ndarray],
    metric_mask: np.ndarray,
    *,
    roi_size: int = 32,
    rois_per_slice: int = 3,
) -> dict:
    """Select homogeneous non-overlapping ROIs using ground truth only."""
    entries = []
    for slice_id in sorted(ground_truth_by_slice):
        target = np.asarray(ground_truth_by_slice[slice_id], dtype=np.float64)
        target_low = mask_normalized_gaussian(target, metric_mask, sigma=2.0)
        low_gradient = gradient_magnitude(target_low)
        candidates = []
        roi_index = 0
        for row in range(0, target.shape[0] - roi_size + 1, roi_size):
            for column in range(0, target.shape[1] - roi_size + 1, roi_size):
                roi_mask = metric_mask[row:row + roi_size, column:column + roi_size]
                if not roi_mask.all():
                    roi_index += 1
                    continue
                values = target[row:row + roi_size, column:column + roi_size]
                mean = float(values.mean())
                if 0.10 <= mean <= 0.90:
                    candidates.append(
                        (
                            float(low_gradient[row:row + roi_size, column:column + roi_size].mean()),
                            f"{slice_id}:{roi_index:06d}",
                            row,
                            column,
                            mean,
                        )
                    )
                roi_index += 1
        candidates.sort(key=lambda item: (item[0], item[1]))
        for selected_index, (gradient, tie_breaker, row, column, mean) in enumerate(
            candidates[:rois_per_slice]
        ):
            entries.append(
                {
                    "roi_id": f"{slice_id}_roi_{selected_index}",
                    "slice_id": slice_id,
                    "row": int(row),
                    "column": int(column),
                    "size": int(roi_size),
                    "ground_truth_mean": mean,
                    "lowpass_gradient_mean": gradient,
                    "tie_breaker": tie_breaker,
                }
            )
    return {
        "schema_version": 1,
        "selection_uses_model_outputs": False,
        "metric_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(metric_mask, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "ground_truth_sha256": {
            slice_id: hashlib.sha256(
                np.ascontiguousarray(image, dtype=np.float32).tobytes()
            ).hexdigest()
            for slice_id, image in sorted(ground_truth_by_slice.items())
        },
        "roi_size": roi_size,
        "requested_per_slice": rois_per_slice,
        "total_rois": len(entries),
        "speckle_conclusion_available": len(entries) >= 30,
        "rois": entries,
    }


def paired_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Paired bootstrap inputs must be non-empty equal-length vectors")
    differences = left - right
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n": int(differences.size),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": None, "sd": None, "median": None, "iqr": None}
    q1, q3 = np.quantile(array, [0.25, 0.75])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "iqr": float(q3 - q1),
    }


def alpha_control_diagnostics(rows: list[dict]) -> list[dict]:
    """Compute per-seed/slice alpha–HF correlation and >2% reverse steps."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row["model"] == "E2":
            grouped.setdefault((int(row["seed"]), row["slice_id"]), []).append(row)
    diagnostics = []
    for (seed, slice_id), values in sorted(grouped.items()):
        values.sort(key=lambda item: float(item["alpha"]))
        alphas = np.asarray([item["alpha"] for item in values], dtype=np.float64)
        energies = np.asarray([item["hf"] for item in values], dtype=np.float64)
        rho = float(cast(float, spearmanr(alphas, energies)[0]))
        reverse_steps = 0
        for previous, current in zip(energies[:-1], energies[1:]):
            if current < previous * 0.98:
                reverse_steps += 1
        diagnostics.append(
            {
                "seed": seed,
                "slice_id": slice_id,
                "spearman_alpha_hf": rho,
                "reverse_steps_over_2pct": reverse_steps,
                "adjacent_steps": max(0, len(energies) - 1),
            }
        )
    return diagnostics


def select_matched_gaussian_sigma(
    e1_validation: dict[tuple[int, str], np.ndarray],
    e2_i0_validation: dict[tuple[int, str], np.ndarray],
    e2_i1_validation: dict[tuple[int, str], np.ndarray],
    mask: np.ndarray,
) -> dict:
    keys = sorted(set(e1_validation) & set(e2_i0_validation) & set(e2_i1_validation))
    if not keys:
        raise ValueError("No paired validation predictions are available for C0 selection")
    target_reductions = []
    for key in keys:
        hf_zero = high_frequency_energy(np.clip(e2_i0_validation[key], 0, 1), mask)
        hf_one = high_frequency_energy(np.clip(e2_i1_validation[key], 0, 1), mask)
        target_reductions.append((hf_one - hf_zero) / max(hf_one, 1e-12))
    target = float(np.median(target_reductions))
    candidates = []
    for sigma in GAUSSIAN_CONTROL_SIGMAS:
        reductions = []
        for key in keys:
            source = np.clip(e1_validation[key], 0, 1)
            smoothed = mask_normalized_gaussian(source, mask, sigma=sigma)
            source_hf = high_frequency_energy(source, mask)
            smooth_hf = high_frequency_energy(smoothed, mask)
            reductions.append((source_hf - smooth_hf) / max(source_hf, 1e-12))
        reduction = float(np.median(reductions))
        relative_gap = abs(reduction - target) / max(abs(target), 1e-12)
        candidates.append(
            {
                "sigma_px": sigma,
                "median_hf_reduction": reduction,
                "target_e2_median_hf_reduction": target,
                "relative_gap": relative_gap,
            }
        )
    selected = min(candidates, key=lambda item: (item["relative_gap"], item["sigma_px"]))
    return {
        "selected_sigma_px": selected["sigma_px"],
        "relative_gap": selected["relative_gap"],
        "g3_smoothing_match_conclusive": selected["relative_gap"] <= 0.05,
        "candidates": candidates,
        "validation_pairs": len(keys),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_sanity_check() -> dict[str, float]:
    coordinates = np.indices((64, 64)).sum(axis=0)
    reference = np.full((64, 64), 0.5, dtype=np.float64)
    mask = np.ones_like(reference, dtype=bool)
    identical = slice_metrics(reference, reference, mask)
    noisy = reference + 0.1 * np.where(coordinates % 2, -1.0, 1.0)
    hf_reference = high_frequency_energy(reference, mask)
    hf_noisy = high_frequency_energy(noisy, mask)
    if identical["mae"] != 0 or not np.isclose(identical["ssim"], 1.0, atol=1e-8):
        raise AssertionError("Identical-image metric sanity check failed")
    if not np.isclose(identical["gms_gt"], 1.0, atol=1e-8):
        raise AssertionError("Identical-image GMS sanity check failed")
    if hf_noisy <= hf_reference:
        raise AssertionError("Known high-frequency noise did not increase HF")
    return {
        "identical_mae": identical["mae"],
        "identical_ssim": identical["ssim"],
        "identical_gms": identical["gms_gt"],
        "clean_hf": hf_reference,
        "noisy_hf": hf_noisy,
    }
