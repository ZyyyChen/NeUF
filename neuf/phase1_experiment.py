from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.phase1_data import (
    FrameRef,
    Phase1DataBlockedError,
    freeze_phase1_manifests,
    hash_array,
    metric_mask,
    pose_array,
    split_content_overlaps,
)
from neuf.phase1_evaluation import (
    ALPHAS,
    BOOTSTRAP_SEED,
    alpha_control_diagnostics,
    build_roi_manifest,
    mask_normalized_gaussian,
    paired_bootstrap_ci,
    roi_metrics,
    select_matched_gaussian_sigma,
    slice_metrics,
    summary,
    write_csv,
)
from neuf.slice_renderer import SliceRenderer


EXPECTED_SEEDS = (3407, 3408, 3409)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _freeze_json(path: Path, payload) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Frozen evaluation manifest differs: {path}")
        return
    _json_dump(path, payload)


def _parse_checkpoint(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("checkpoint must use SEED=/path/to/checkpoint.pkl") from error
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return seed, path


def _partition_tensor(dataset: Dataset, ref: FrameRef, name: str) -> torch.Tensor:
    source_name = name if ref.source == "training_pool" else f"{name}_valid"
    tensor = getattr(dataset, source_name)
    return tensor[ref.slice_info.start:ref.slice_info.end]


def _reference_image(dataset: Dataset, ref: FrameRef) -> np.ndarray:
    return _partition_tensor(dataset, ref, "pixels").detach().cpu().numpy().reshape(
        dataset.px_height, dataset.px_width
    )


def _query_image(
    renderer: SliceRenderer,
    model: NeRF,
    dataset: Dataset,
    ref: FrameRef,
    *,
    component: str = "intensity",
    alpha: float = 1.0,
) -> np.ndarray:
    points = _partition_tensor(dataset, ref, "points").unsqueeze(1)
    viewdirs = _partition_tensor(dataset, ref, "viewdirs").unsqueeze(1)
    output = renderer.query_points(
        model,
        points,
        viewdirs,
        alpha=alpha,
        component=component,
    )
    return output.detach().cpu().numpy().reshape(dataset.px_height, dataset.px_width)


def _query_components(
    renderer: SliceRenderer,
    model: NeRF,
    dataset: Dataset,
    ref: FrameRef,
) -> dict[str, np.ndarray]:
    points = _partition_tensor(dataset, ref, "points").unsqueeze(1)
    viewdirs = _partition_tensor(dataset, ref, "viewdirs").unsqueeze(1)
    components = renderer.query_point_components(model, points, viewdirs, alpha=1.0)
    return {
        name: value.detach().cpu().numpy().reshape(dataset.px_height, dataset.px_width)
        for name, value in components.items()
    }


def _load_predictions(
    dataset: Dataset,
    splits: dict[str, list[FrameRef]],
    checkpoints: dict[str, dict[int, Path]],
    *,
    split_names=("validation", "test"),
    experiments=("E0", "E1", "E2"),
) -> tuple[dict, dict, dict]:
    expected_heads = {
        "E0": NeRF.LEGACY_FIELD_HEAD,
        "E1": NeRF.MATCHED_FIELD_HEAD,
        "E2": NeRF.ANATOMY_SPECKLE_FIELD_HEAD,
    }
    expected_encodings = {
        "E0": "HASH",
        "E1": "DUAL_HASH",
        "E2": "DUAL_HASH",
    }
    renderer = SliceRenderer(dataset)
    test_predictions = {}
    validation_predictions = {}
    checkpoint_metadata = {}
    expected_split_ids = {
        name: [ref.stable_slice_id for ref in refs] for name, refs in splits.items()
    }
    expected_pose_hash = hash_array(
        np.stack(
            [
                pose_array(ref.slice_info)
                for name in ("training", "validation", "test")
                for ref in splits[name]
            ]
        )
    )
    for experiment in experiments:
        for seed, checkpoint_path in sorted(checkpoints[experiment].items()):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            validate_checkpoint_dataset_geometry(
                checkpoint,
                dataset,
                checkpoint_path=checkpoint_path,
            )
            actual_head = str(checkpoint.get("field_head", NeRF.LEGACY_FIELD_HEAD)).lower()
            if actual_head != expected_heads[experiment]:
                raise ValueError(
                    f"{experiment} seed {seed} has field_head={actual_head}, "
                    f"expected={expected_heads[experiment]}"
                )
            actual_encoding = str(checkpoint.get("encoding", "")).upper()
            if actual_encoding != expected_encodings[experiment]:
                raise ValueError(
                    f"{experiment} seed {seed} has encoding={actual_encoding}, "
                    f"expected={expected_encodings[experiment]}"
                )
            if int(checkpoint.get("seed", seed)) != seed:
                raise ValueError(f"Checkpoint seed metadata does not match CLI seed {seed}")
            integrity_errors = []
            if checkpoint.get("phase1_split_ids") != expected_split_ids:
                integrity_errors.append("split IDs differ from frozen manifest")
            if checkpoint.get("phase1_pose_hash_before") != expected_pose_hash:
                integrity_errors.append("pre-training pose hash differs from frozen data")
            if checkpoint.get("phase1_pose_hash_current") != expected_pose_hash:
                integrity_errors.append("post-training pose hash changed or is missing")
            if checkpoint.get("optimize_poses", False) or "pose_refiner_state_dict" in checkpoint:
                integrity_errors.append("pose optimization state is present")
            if checkpoint.get("sagittal_mat") or "sagittal_pose_refiner_state_dict" in checkpoint:
                integrity_errors.append("sagittal supervision state is present")
            if checkpoint.get("training_mode") != "Patch":
                integrity_errors.append("training mode is not Patch")
            if int(checkpoint.get("patch_size", -1)) != 64:
                integrity_errors.append("patch size is not 64")
            if int(checkpoint.get("points_per_iter", -1)) != 49152:
                integrity_errors.append("resolved point budget is not 49,152")
            if int(checkpoint.get("start", -1)) != 20000:
                integrity_errors.append("checkpoint is not the frozen final iteration 20,000")
            if int(checkpoint.get("iterations", -1)) != 20000:
                integrity_errors.append("configured training budget is not 20,000")
            expected_loss = {
                "E0": "masked_mse",
                "E1": "masked_mse",
                "E2": "masked_mse",
            }[experiment]
            if checkpoint.get("training_loss") != expected_loss:
                integrity_errors.append(
                    f"training loss is not the frozen {expected_loss}"
                )
            if experiment == "E1" and bool(checkpoint.get("use_gate", True)):
                integrity_errors.append("E1 must not contain a spatial gate")
            if experiment == "E2" and bool(checkpoint.get("use_gate", True)):
                integrity_errors.append("E2 must not contain a spatial gate")
            if integrity_errors:
                raise ValueError(
                    f"{checkpoint_path} violates Phase 1 integrity: {integrity_errors}"
                )
            model = NeRF(checkpoint)
            model.training_progress = 1.0
            model.eval()
            checkpoint_metadata[f"{experiment}_{seed}"] = {
                "path": str(checkpoint_path.resolve()),
                "parameter_counts": model.parameter_counts(),
                "training_progress": model.training_progress,
                "start": int(checkpoint.get("start", -1)),
                "source_control": checkpoint.get("source_control"),
            }
            with torch.no_grad():
                for split_name in split_names:
                    destination = (
                        validation_predictions if split_name == "validation" else test_predictions
                    )
                    for ref in splits[split_name]:
                        key_prefix = (experiment, seed, ref.stable_slice_id)
                        if experiment == "E2":
                            components = _query_components(renderer, model, dataset, ref)
                            anatomy = components["anatomy"]
                            speckle = components["speckle"]
                            for alpha in ALPHAS:
                                destination[key_prefix + (alpha,)] = anatomy + alpha * speckle
                            destination[key_prefix + ("anatomy",)] = anatomy
                            destination[key_prefix + ("speckle",)] = speckle
                        else:
                            destination[key_prefix + (1.0,)] = _query_image(
                                renderer, model, dataset, ref
                            )
            del model, checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return test_predictions, validation_predictions, checkpoint_metadata


def _seed_averaged_by_slice(rows: list[dict], model: str, alpha: float, metric: str) -> dict[str, float]:
    grouped = {}
    for row in rows:
        if row["model"] == model and float(row["alpha"]) == float(alpha):
            grouped.setdefault(row["slice_id"], []).append(float(row[metric]))
    return {slice_id: float(np.mean(values)) for slice_id, values in grouped.items()}


def _paired_from_rows(
    rows: list[dict],
    left_model: str,
    left_alpha: float,
    right_model: str,
    right_alpha: float,
    metric: str,
) -> dict:
    left = _seed_averaged_by_slice(rows, left_model, left_alpha, metric)
    right = _seed_averaged_by_slice(rows, right_model, right_alpha, metric)
    keys = sorted(set(left) & set(right))
    return paired_bootstrap_ci(
        np.asarray([left[key] for key in keys]),
        np.asarray([right[key] for key in keys]),
    )


def _aggregate(rows: list[dict]) -> dict:
    output = {}
    metrics = [
        "mae", "psnr", "ssim", "lp_ssim_gt", "gms_gt", "hf", "brightness",
        "out_of_range_fraction", "lp_ssim_i1", "brightness_drift_i1", "leak_low",
    ]
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for alpha in sorted({float(row["alpha"]) for row in model_rows}):
            selected = [row for row in model_rows if float(row["alpha"]) == alpha]
            output[f"{model}_alpha_{alpha:.2f}"] = {
                metric: summary(row[metric] for row in selected) for metric in metrics
            }
    return output


def _alpha_slice_diagnostics(rows: list[dict]) -> list[dict]:
    slice_ids = sorted({row["slice_id"] for row in rows if row["model"] == "E2"})
    output = []
    for slice_id in slice_ids:
        energies = []
        for alpha in ALPHAS:
            values = [
                row["hf"] for row in rows
                if row["model"] == "E2" and row["slice_id"] == slice_id
                and float(row["alpha"]) == alpha
            ]
            energies.append(float(np.mean(values)))
        rho = float(cast(float, spearmanr(ALPHAS, energies)[0]))
        reverse = sum(
            current < previous * 0.98
            for previous, current in zip(energies[:-1], energies[1:])
        )
        output.append(
            {
                "slice_id": slice_id,
                "spearman_alpha_hf": rho,
                "reverse_steps_over_2pct": int(reverse),
                "adjacent_steps": 4,
            }
        )
    return output


def _make_figures(
    output_dir: Path,
    display_ids: list[str],
    ground_truth: dict[str, np.ndarray],
    predictions: dict,
    checkpoints: dict[str, dict[int, Path]],
    selected_sigma: float,
    rows: list[dict],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    representative_seed = sorted(checkpoints["E2"])[0]
    e0_seed = sorted(checkpoints["E0"])[0]
    e1_seed = sorted(checkpoints["E1"])[0]

    fig, axes = plt.subplots(len(display_ids), 6, figsize=(18, 3 * len(display_ids)), squeeze=False)
    for row_index, slice_id in enumerate(display_ids):
        panels = [
            ("GT", ground_truth[slice_id], "gray", 0, 1),
            ("E0", predictions[("E0", e0_seed, slice_id, 1.0)], "gray", 0, 1),
            ("E1", predictions[("E1", e1_seed, slice_id, 1.0)], "gray", 0, 1),
            ("E2 A", predictions[("E2", representative_seed, slice_id, "anatomy")], "gray", 0, 1),
            ("E2 S", predictions[("E2", representative_seed, slice_id, "speckle")], "coolwarm", -0.5, 0.5),
            ("E2 I1", predictions[("E2", representative_seed, slice_id, 1.0)], "gray", 0, 1),
        ]
        for axis, (title, image, colour_map, minimum, maximum) in zip(axes[row_index], panels):
            axis.imshow(image, cmap=colour_map, vmin=minimum, vmax=maximum)
            axis.set_title(f"{slice_id}\n{title}")
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "fixed_test_slices_components.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(display_ids), len(ALPHAS), figsize=(15, 3 * len(display_ids)), squeeze=False)
    for row_index, slice_id in enumerate(display_ids):
        for axis, alpha in zip(axes[row_index], ALPHAS):
            axis.imshow(
                predictions[("E2", representative_seed, slice_id, alpha)],
                cmap="gray", vmin=0, vmax=1,
            )
            axis.set_title(f"alpha={alpha:.2f}")
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "fixed_test_slices_alpha_sweep.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(display_ids), 3, figsize=(9, 3 * len(display_ids)), squeeze=False)
    for row_index, slice_id in enumerate(display_ids):
        target = ground_truth[slice_id]
        error_images = [
            np.abs(predictions[("E0", e0_seed, slice_id, 1.0)] - target),
            np.abs(predictions[("E1", e1_seed, slice_id, 1.0)] - target),
            np.abs(predictions[("E2", representative_seed, slice_id, 1.0)] - target),
        ]
        for axis, title, image in zip(axes[row_index], ("E0", "E1", "E2 I1"), error_images):
            axis.imshow(image, cmap="magma", vmin=0, vmax=0.5)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "fixed_test_slices_error_maps.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric, title in zip(axes, ("hf", "lp_ssim_i1", "gms_gt"), ("HF", "LP-SSIM vs I1", "GMS vs GT")):
        means = [
            np.mean([row[metric] for row in rows if row["model"] == "E2" and row["alpha"] == alpha])
            for alpha in ALPHAS
        ]
        axis.plot(ALPHAS, means, marker="o")
        axis.set_xlabel("alpha")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "alpha_metric_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(display_ids), 3, figsize=(9, 3 * len(display_ids)), squeeze=False)
    for row_index, slice_id in enumerate(display_ids):
        e1 = predictions[("E1", e1_seed, slice_id, 1.0)]
        panels = (
            ("E1", e1),
            (f"C0 sigma={selected_sigma:g}", mask_normalized_gaussian(e1, np.isfinite(e1), selected_sigma)),
            ("E2 alpha=0", predictions[("E2", representative_seed, slice_id, 0.0)]),
        )
        for axis, (title, image) in zip(axes[row_index], panels):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "matched_smoothing_control.png", dpi=160)
    plt.close(fig)


def run_evaluation(args) -> str:
    output_dir = Path(args.output_dir).expanduser()
    dataset = Dataset.open_from_save(args.dataset)
    splits = freeze_phase1_manifests(
        dataset,
        args.dataset,
        output_dir,
        allow_small_dataset=args.allow_small_dataset,
    )
    checkpoint_groups = {
        "E0": dict(args.e0),
        "E1": dict(args.e1),
        "E2": dict(args.e2),
    }
    for experiment, checkpoints in checkpoint_groups.items():
        if set(checkpoints) != set(EXPECTED_SEEDS) and not args.allow_incomplete_seeds:
            raise ValueError(
                f"{experiment} must provide seeds {EXPECTED_SEEDS}, got {sorted(checkpoints)}"
            )

    ground_truth = {
        ref.stable_slice_id: _reference_image(dataset, ref)
        for ref in splits["test"]
    }
    base_metric_mask = metric_mask(dataset, erosion_px=3)
    if any(not np.isfinite(image[base_metric_mask]).all() for image in ground_truth.values()):
        raise ValueError("Ground-truth test data contain non-finite pixels inside metric mask")

    roi_manifest = build_roi_manifest(ground_truth, base_metric_mask)
    _freeze_json(output_dir / "manifests" / "roi_manifest.json", roi_manifest)
    rois_by_slice = {}
    for roi in roi_manifest["rois"]:
        rois_by_slice.setdefault(roi["slice_id"], []).append(roi)

    _, validation_predictions, validation_checkpoint_metadata = _load_predictions(
        dataset,
        splits,
        checkpoint_groups,
        split_names=("validation",),
        experiments=("E1", "E2"),
    )

    e1_validation = {
        (seed, slice_id): value
        for (model, seed, slice_id, alpha), value in validation_predictions.items()
        if model == "E1" and alpha == 1.0
    }
    e2_i0_validation = {
        (seed, slice_id): value
        for (model, seed, slice_id, alpha), value in validation_predictions.items()
        if model == "E2" and alpha == 0.0
    }
    e2_i1_validation = {
        (seed, slice_id): value
        for (model, seed, slice_id, alpha), value in validation_predictions.items()
        if model == "E2" and alpha == 1.0
    }
    smoothing = select_matched_gaussian_sigma(
        e1_validation, e2_i0_validation, e2_i1_validation, base_metric_mask
    )
    _json_dump(output_dir / "metrics" / "matched_smoothing_selection.json", smoothing)
    selected_sigma = float(smoothing["selected_sigma_px"])

    evaluation_lock_path = output_dir / "logs" / "formal_test_evaluation_lock.json"
    if evaluation_lock_path.exists():
        raise RuntimeError(
            "Frozen test has already been queried; refusing a second test evaluation: "
            f"{evaluation_lock_path}"
        )
    _json_dump(
        evaluation_lock_path,
        {
            "status": "test evaluation started after validation-only C0 selection",
            "selected_c0_sigma_px": selected_sigma,
            "checkpoints": {
                experiment: {str(seed): str(path.resolve()) for seed, path in values.items()}
                for experiment, values in checkpoint_groups.items()
            },
        },
    )
    predictions, _, test_checkpoint_metadata = _load_predictions(
        dataset,
        splits,
        checkpoint_groups,
        split_names=("test",),
    )
    checkpoint_metadata = {
        **validation_checkpoint_metadata,
        **test_checkpoint_metadata,
    }
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("A test prediction contains NaN/Inf; G7 cannot pass")

    rows = []
    roi_rows = []
    for key, prediction in sorted(predictions.items(), key=lambda item: str(item[0])):
        model, seed, slice_id, alpha_or_component = key
        if not isinstance(alpha_or_component, float):
            continue
        alpha = alpha_or_component
        i1 = predictions.get(("E2", seed, slice_id, 1.0)) if model == "E2" else None
        speckle = predictions.get(("E2", seed, slice_id, "speckle")) if model == "E2" else None
        values = slice_metrics(
            prediction,
            ground_truth[slice_id],
            base_metric_mask,
            reference_i1=i1,
            speckle=speckle,
        )
        row = {"model": model, "seed": seed, "slice_id": slice_id, "alpha": alpha, **values}
        rows.append(row)
        clipped = np.clip(prediction, 0.0, 1.0)
        for roi in rois_by_slice.get(slice_id, []):
            roi_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "slice_id": slice_id,
                    "roi_id": roi["roi_id"],
                    "alpha": alpha,
                    **roi_metrics(clipped, roi),
                }
            )

    for seed in sorted(checkpoint_groups["E1"]):
        for ref in splits["test"]:
            slice_id = ref.stable_slice_id
            source = np.clip(predictions[("E1", seed, slice_id, 1.0)], 0.0, 1.0)
            smoothed = mask_normalized_gaussian(source, base_metric_mask, selected_sigma)
            predictions[("C0", seed, slice_id, 1.0)] = smoothed
            values = slice_metrics(smoothed, ground_truth[slice_id], base_metric_mask)
            rows.append({"model": "C0", "seed": seed, "slice_id": slice_id, "alpha": 1.0, **values})
            for roi in rois_by_slice.get(slice_id, []):
                roi_rows.append(
                    {
                        "model": "C0", "seed": seed, "slice_id": slice_id,
                        "roi_id": roi["roi_id"], "alpha": 1.0,
                        **roi_metrics(smoothed, roi),
                    }
                )

    write_csv(output_dir / "metrics" / "per_slice_metrics.csv", rows)
    write_csv(output_dir / "metrics" / "per_roi_metrics.csv", roi_rows)
    aggregate = _aggregate(rows)
    _json_dump(output_dir / "metrics" / "aggregate_metrics.json", aggregate)

    bootstrap = {}
    for baseline in ("E0", "E1"):
        for metric_name in ("psnr", "ssim"):
            bootstrap[f"E2_alpha1_minus_{baseline}_{metric_name}"] = _paired_from_rows(
                rows, "E2", 1.0, baseline, 1.0, metric_name
            )
    bootstrap["E2_alpha0_minus_alpha1_lp_ssim_gt"] = _paired_from_rows(
        rows, "E2", 0.0, "E2", 1.0, "lp_ssim_gt"
    )
    bootstrap["E2_alpha0_minus_C0_gms_gt"] = _paired_from_rows(
        rows, "E2", 0.0, "C0", 1.0, "gms_gt"
    )
    _json_dump(output_dir / "metrics" / "paired_bootstrap_ci.json", bootstrap)

    alpha_diagnostics = _alpha_slice_diagnostics(rows)
    seed_alpha_diagnostics = alpha_control_diagnostics(rows)
    _json_dump(
        output_dir / "metrics" / "alpha_control_diagnostics.json",
        {"seed_slice": seed_alpha_diagnostics, "seed_averaged_slice": alpha_diagnostics},
    )

    hf_zero = _seed_averaged_by_slice(rows, "E2", 0.0, "hf")
    hf_one = _seed_averaged_by_slice(rows, "E2", 1.0, "hf")
    hf_reductions = [(hf_one[key] - hf_zero[key]) / max(hf_one[key], 1e-12) for key in hf_one]
    roi_sc = {}
    for row in roi_rows:
        if row["model"] == "E2" and row["alpha"] in {0.0, 1.0}:
            roi_sc.setdefault((row["roi_id"], row["alpha"]), []).append(row["sc"])
    sc_reductions = []
    for roi_id in {key[0] for key in roi_sc}:
        zero = np.mean(roi_sc[(roi_id, 0.0)])
        one = np.mean(roi_sc[(roi_id, 1.0)])
        sc_reductions.append((one - zero) / max(one, 1e-12))

    brightness_zero = _seed_averaged_by_slice(rows, "E2", 0.0, "brightness")
    brightness_one = _seed_averaged_by_slice(rows, "E2", 1.0, "brightness")
    brightness_differences = [abs(brightness_zero[key] - brightness_one[key]) for key in brightness_one]
    leak_values = [row["leak_low"] for row in rows if row["model"] == "E2" and row["alpha"] == 1.0]
    all_alpha_lp_means = [
        np.mean([row["lp_ssim_i1"] for row in rows if row["model"] == "E2" and row["alpha"] == alpha])
        for alpha in ALPHAS
    ]
    reverse_total = sum(item["reverse_steps_over_2pct"] for item in alpha_diagnostics)
    adjacent_total = sum(item["adjacent_steps"] for item in alpha_diagnostics)

    g1_checks = [
        bootstrap[f"E2_alpha1_minus_{baseline}_{metric}"]["ci95_lower"] >= threshold
        for baseline in ("E0", "E1")
        for metric, threshold in (("psnr", -0.50), ("ssim", -0.010))
    ]
    content_overlaps = split_content_overlaps(splits)
    gates = {
        "G0": {
            "status": (
                "PASS"
                if sum(len(value) for value in splits.values()) >= 50
                and not any(content_overlaps.values())
                else "BLOCKED"
            ),
            "actual": {
                "split_counts": {name: len(value) for name, value in splits.items()},
                "content_overlap_counts": content_overlaps,
            },
            "threshold": "at least 50 slices; disjoint frozen split and unchanged geometry",
        },
        "G1": {
            "status": "PASS" if all(g1_checks) else "FAIL",
            "actual": {key: value for key, value in bootstrap.items() if "alpha1_minus" in key},
            "threshold": "CI lower PSNR >= -0.50 dB and SSIM >= -0.010 vs E0 and E1",
        },
        "G2": {
            "status": (
                "INCONCLUSIVE" if roi_manifest["total_rois"] < 30
                else "PASS" if np.median(hf_reductions) >= 0.25 and np.median(sc_reductions) >= 0.20
                else "FAIL"
            ),
            "actual": {
                "median_hf_reduction": float(np.median(hf_reductions)),
                "median_sc_reduction": float(np.median(sc_reductions)) if sc_reductions else None,
                "roi_count": roi_manifest["total_rois"],
            },
            "threshold": "median HF reduction >= 25%; median ROI SC reduction >= 20%; >=30 ROIs",
        },
        "G3": {
            "status": (
                "INCONCLUSIVE" if not smoothing["g3_smoothing_match_conclusive"]
                else "PASS" if bootstrap["E2_alpha0_minus_alpha1_lp_ssim_gt"]["ci95_lower"] >= -0.010
                and bootstrap["E2_alpha0_minus_C0_gms_gt"]["mean_difference"] >= 0.010
                and bootstrap["E2_alpha0_minus_C0_gms_gt"]["ci95_lower"] > 0
                else "FAIL"
            ),
            "actual": {
                "smoothing": smoothing,
                "lp_ssim": bootstrap["E2_alpha0_minus_alpha1_lp_ssim_gt"],
                "gms_vs_c0": bootstrap["E2_alpha0_minus_C0_gms_gt"],
            },
            "threshold": "LP-SSIM CI lower >= -0.010; GMS gain >=0.010 and CI lower >0",
        },
        "G4": {
            "status": "PASS" if np.median([item["spearman_alpha_hf"] for item in alpha_diagnostics]) >= 0.95
            and np.mean([item["spearman_alpha_hf"] >= 0.90 for item in alpha_diagnostics]) >= 0.90
            and reverse_total / max(adjacent_total, 1) <= 0.05 else "FAIL",
            "actual": {
                "median_spearman": float(np.median([item["spearman_alpha_hf"] for item in alpha_diagnostics])),
                "fraction_slices_rho_ge_0.90": float(np.mean([item["spearman_alpha_hf"] >= 0.90 for item in alpha_diagnostics])),
                "reverse_step_fraction": reverse_total / max(adjacent_total, 1),
            },
            "threshold": "median rho>=0.95; >=90% slices rho>=0.90; reverse steps<=5%",
        },
        "G5": {
            "status": "PASS" if min(all_alpha_lp_means) >= 0.98
            and np.median(brightness_differences) <= 0.02
            and np.quantile(brightness_differences, 0.95) <= 0.03 else "FAIL",
            "actual": {
                "minimum_alpha_mean_lp_ssim_i1": float(min(all_alpha_lp_means)),
                "median_brightness_difference": float(np.median(brightness_differences)),
                "p95_brightness_difference": float(np.quantile(brightness_differences, 0.95)),
            },
            "threshold": "LP-SSIM mean>=0.98; brightness median<=0.02 and p95<=0.03",
        },
        "G6": {
            "status": "PASS" if np.median(leak_values) <= 0.20 and np.median(hf_reductions) >= 0.25 else "FAIL",
            "actual": {"median_leak_low": float(np.median(leak_values))},
            "threshold": "median Leak_low<=0.20 and G2 HF reduction passes",
        },
        "G7": {
            "status": "PASS" if all(set(group) == set(EXPECTED_SEEDS) for group in checkpoint_groups.values()) else "INCONCLUSIVE",
            "actual": {name: sorted(group) for name, group in checkpoint_groups.items()},
            "threshold": "all three fixed seeds complete, finite, and fully reported",
        },
    }
    statuses = {gate["status"] for gate in gates.values()}
    if "BLOCKED" in statuses:
        overall = "BLOCKED"
    elif "INCONCLUSIVE" in statuses:
        overall = "INCONCLUSIVE"
    elif "FAIL" in statuses:
        overall = "FAIL"
    else:
        overall = "PASS"
    acceptance = {
        "overall": f"PHASE 1: {overall}",
        "gates": gates,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    _json_dump(output_dir / "metrics" / "acceptance_gates.json", acceptance)
    _json_dump(output_dir / "metrics" / "checkpoint_metadata.json", checkpoint_metadata)

    display_manifest = json.loads((output_dir / "manifests" / "display_slice_ids.json").read_text())
    _make_figures(
        output_dir,
        display_manifest["stable_slice_ids"],
        ground_truth,
        predictions,
        checkpoint_groups,
        selected_sigma,
        rows,
    )
    return acceptance["overall"]


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen Phase 1 held-out evaluator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="phase1_image_quality")
    parser.add_argument("--e0", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--e1", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--e2", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--allow-small-dataset", action="store_true", help="Smoke only")
    parser.add_argument("--allow-incomplete-seeds", action="store_true", help="Diagnostic only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_evaluation(args)
    except Phase1DataBlockedError as error:
        raise SystemExit(f"PHASE 1: BLOCKED — {error}") from error
    print(result)


if __name__ == "__main__":
    main()
