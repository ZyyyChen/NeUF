from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.phase1_data import (
    apply_single_slice_training_view,
    apply_phase1_training_split,
    current_pose_hash,
    freeze_phase1_manifests,
    freeze_single_slice_manifest,
)
from neuf.phase1_losses import masked_mean
from neuf.slice_renderer import SliceRenderer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_PATH = "data/simu_56/us/baked_dataset.pkl"
VALIDATION_PREVIEW_COUNT = 4


def _source_control_state() -> dict[str, object]:
    """Record whether the checkpoint came from a reproducible clean commit."""
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        untracked_source_paths = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "neuf",
                "code_seg_3D",
                "experiments",
                "jobs",
                "scripts",
                "tests",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "tracked_changed_path_count": None,
            "untracked_source_path_count": None,
        }
    return {
        "git_commit": commit,
        "git_dirty": bool(tracked_status or untracked_source_paths),
        "tracked_changed_path_count": len(tracked_status),
        "untracked_source_path_count": len(untracked_source_paths),
    }


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    images: Path
    tensorboard: Path
    latest_checkpoint: Path


@dataclass(frozen=True)
class ValidationPreview:
    slice_id: str
    target: torch.Tensor
    prediction: torch.Tensor
    anatomy: torch.Tensor | None = None
    speckle: torch.Tensor | None = None
    structure: torch.Tensor | None = None
    boundary: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    mask: torch.Tensor | None = None


def _save_alpha_previews(
    output_dir: Path, iteration: int, previews: list[ValidationPreview], *,
    selection: str = "first four frames in frozen validation split order",
    metric_scope: str | None = None,
) -> None:
    """同一组固定验证帧显示五档 alpha；含噪观测误差不充当降噪真值指标。"""
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows, manifest = [], []
    for item in previews:
        mask = item.mask.bool()
        valid_rc = torch.nonzero(mask)
        center = ((valid_rc.amin(0) + valid_rc.amax(0)) // 2).tolist()
        height, width = mask.shape
        top, left = max(0, min(center[0] - 64, height - 128)), max(0, min(center[1] - 64, width - 128))
        crop = (slice(top, min(height, top + 128)), slice(left, min(width, left + 128)))
        images = [("Observed", item.target)] + [
            (f"alpha={alpha:g}", item.anatomy + alpha * item.residual) for alpha in alphas
        ]
        figure, axes = plt.subplots(2, len(images), figsize=(18, 7), squeeze=False)
        for column, (name, values) in enumerate(images):
            shown = np.where(mask.numpy(), values.numpy(), np.nan)
            for row, region in enumerate((np.s_[:, :], crop)):
                axes[row, column].imshow(shown[region], cmap="gray", vmin=0, vmax=1)
                axes[row, column].set_title(name + (" / fixed ROI" if row else ""))
                axes[row, column].axis("off")
        figure.suptitle(f"{item.slice_id}, step {iteration}; common display [0,1]; no clean GT")
        figure.tight_layout()
        figure.savefig(output_dir / f"alpha_{iteration:06d}_{item.slice_id}.png", dpi=150)
        plt.close(figure)
        for alpha in alphas:
            values = item.anatomy + alpha * item.residual
            gradients = []
            for dim in (0, 1):
                pair_mask = mask.narrow(dim, 1, mask.shape[dim] - 1) & mask.narrow(dim, 0, mask.shape[dim] - 1)
                gradients.append(torch.diff(values, dim=dim)[pair_mask])
            gradient = torch.cat(gradients)
            rows.append({
                "iteration": iteration, "slice_id": item.slice_id, "alpha": alpha,
                "valid_pixels": int(mask.sum()),
                "mse_to_noisy_observation": float((values[mask] - item.target[mask]).square().mean()),
                "mean": float(values[mask].mean()),
                "adjacent_gradient_rms": float(gradient.square().mean().sqrt()),
                "boundary_rms": float(item.boundary[mask].square().mean().sqrt()),
                "residual_mean": float(item.residual[mask].mean()),
                "residual_rms": float(item.residual[mask].square().mean().sqrt()),
            })
        manifest.append({"slice_id": item.slice_id, "crop_rc_hw": [top, left, crop[0].stop - top, crop[1].stop - left]})
    metrics_dir = output_dir.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"alpha_metrics_step_{iteration:06d}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config_dir = output_dir.parent / "run_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "alpha_comparison_manifest.json").write_text(json.dumps({
        "comparison_indices": [item.slice_id for item in previews], "cases": manifest,
        "selection": selection,
        "alphas": alphas, "display_range": [0, 1], "orientation": "native observed slice",
        "mask": "full ultrasound sector", "normalization": "shared dataset display domain; no rescaling",
        "metric_scope": metric_scope or (
            f"{len(previews)} fixed previews only; separate reconstruction validation "
            "mean uses all validation frames"
        ),
        "interpretation": "Gradient RMS is descriptive; no claim of boundary accuracy, pure speckle or segmentation benefit.",
    }, indent=2) + "\n", encoding="utf-8")


def _save_validation_preview(
    output_dir: Path,
    iteration: int,
    previews: list[ValidationPreview],
    writer: SummaryWriter | None = None,
    *,
    selection: str = "first four frames in frozen validation split order",
    metric_scope: str | None = None,
) -> list[Path]:
    """Save one GT/prediction/error figure and tensor file per validation slice."""
    if not previews:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    has_components = all(item.anatomy is not None for item in previews)
    has_decomposition = all(item.structure is not None for item in previews)
    columns = 7 if has_decomposition else (5 if has_components else 3)
    saved_paths = []
    for item in previews:
        figure, axes = plt.subplots(
            1,
            columns,
            figsize=(3.2 * columns, 3.2),
            squeeze=False,
        )
        error = torch.abs(item.prediction - item.target)
        panels: list[tuple[str, torch.Tensor, str, float, float]] = [
            ("GT", item.target, "gray", 0.0, 1.0),
            ("prediction", item.prediction, "gray", 0.0, 1.0),
        ]
        if has_components and item.anatomy is not None and item.speckle is not None:
            panels.extend(
                [
                    ("anatomy", item.anatomy, "gray", 0.0, 1.0),
                    ("speckle", item.speckle, "coolwarm", -0.5, 0.5),
                ]
            )
        if has_decomposition:
            panels.extend([
                ("structure", item.structure, "gray", 0.0, 1.0),
                ("boundary", item.boundary, "coolwarm", -0.5, 0.5),
                ("residual", item.residual, "coolwarm", -0.5, 0.5),
                ("anatomy S+B", item.anatomy, "gray", 0.0, 1.0),
            ])
        panels.append(("absolute error", error, "magma", 0.0, 0.5))
        for axis, (title, image, colour_map, minimum, maximum) in zip(axes[0], panels):
            axis.imshow(image.numpy(), cmap=colour_map, vmin=minimum, vmax=maximum)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(item.slice_id)

        base_name = f"validation_{iteration:06d}_{item.slice_id}"
        image_path = output_dir / f"{base_name}.png"
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        figure.savefig(image_path, dpi=150)
        plt.close(figure)
        tensor_payload: dict[str, object] = {
            "iteration": int(iteration),
            "slice_id": item.slice_id,
            # Clone views before serialization so each per-slice file does not retain
            # the backing storage of an entire validation batch.
            "target": item.target.clone(),
            "prediction": item.prediction.clone(),
        }
        if item.anatomy is not None and item.speckle is not None:
            tensor_payload["anatomy"] = item.anatomy.clone()
            tensor_payload["speckle"] = item.speckle.clone()
        if has_decomposition:
            for name in ("structure", "boundary", "residual", "anatomy"):
                tensor_payload[name] = getattr(item, name).clone()
            tensor_payload["mask"] = item.mask.clone()
        torch.save(tensor_payload, output_dir / f"{base_name}.pt")
        saved_paths.append(image_path)

        if writer is not None:
            prefix = f"validation/{item.slice_id}"
            writer.add_image(
                f"{prefix}/target",
                item.target.clamp(0, 1),
                iteration,
                dataformats="HW",
            )
            writer.add_image(
                f"{prefix}/prediction",
                item.prediction.clamp(0, 1),
                iteration,
                dataformats="HW",
            )
            writer.add_image(
                f"{prefix}/absolute_error",
                (error / 0.5).clamp(0, 1),
                iteration,
                dataformats="HW",
            )
            if item.anatomy is not None and item.speckle is not None:
                writer.add_image(
                    f"{prefix}/anatomy",
                    item.anatomy.clamp(0, 1),
                    iteration,
                    dataformats="HW",
                )
                writer.add_image(
                    f"{prefix}/speckle",
                    (item.speckle + 0.5).clamp(0, 1),
                    iteration,
                    dataformats="HW",
                )
    if has_decomposition:
        _save_alpha_previews(
            output_dir,
            iteration,
            previews,
            selection=selection,
            metric_scope=metric_scope,
        )
    return saved_paths


@torch.no_grad()
def render_validation_preview(
    model: NeRF,
    dataset: Dataset,
    renderer: SliceRenderer,
    output_dir: str | Path,
    iteration: int,
    writer: SummaryWriter | None = None,
) -> float:
    """Evaluate every validation slice and save four deterministic preview rows."""
    was_training = model.training
    model.eval()
    losses = []
    previews = []
    mask = dataset.get_sector_mask(flatten=True, device=DEVICE)
    split_ids = getattr(dataset, "phase1_split_ids", {}).get("validation", [])
    try:
        for index in range(len(dataset.slices_valid)):
            prediction = renderer.render_slice_from_dataset_valid(
                model,
                index,
                reshaped=False,
                alpha=1.0,
            )
            target = dataset.get_slice_valid_pixels(index)
            losses.append(float(masked_mean((prediction - target).square(), mask).cpu()))
            if index >= VALIDATION_PREVIEW_COUNT:
                continue
            slice_info = dataset.slices_valid[index]
            fallback_id = f"frame_{int(getattr(slice_info, 'frame_index', index)):06d}"
            preview = ValidationPreview(
                slice_id=str(split_ids[index]) if index < len(split_ids) else fallback_id,
                target=target.reshape(dataset.px_height, dataset.px_width).detach().cpu(),
                prediction=(
                    prediction.reshape(dataset.px_height, dataset.px_width)
                    .detach()
                    .cpu()
                ),
            )
            if model.field_head == NeRF.ANATOMY_SPECKLE_FIELD_HEAD:
                preview = ValidationPreview(
                    slice_id=preview.slice_id,
                    target=preview.target,
                    prediction=preview.prediction,
                    anatomy=renderer.render_slice_from_dataset_valid(
                        model,
                        index,
                        reshaped=True,
                        component="anatomy",
                    ).detach().cpu(),
                    speckle=renderer.render_slice_from_dataset_valid(
                        model,
                        index,
                        reshaped=True,
                        component="speckle",
                    ).detach().cpu(),
                )
            elif model.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
                values = renderer.query_point_components(
                    model, dataset.get_slice_valid_points(index),
                    dataset.get_slice_valid_viewdirs(index),
                    alpha=1.0,
                )
                maps = {
                    name: (value * mask).reshape(dataset.px_height, dataset.px_width).cpu()
                    for name, value in values.items()
                }
                preview = ValidationPreview(
                    slice_id=preview.slice_id, target=preview.target,
                    prediction=preview.prediction, anatomy=maps["anatomy"],
                    structure=maps["structure"], boundary=maps["boundary"],
                    residual=maps["residual"],
                    mask=mask.reshape(dataset.px_height, dataset.px_width).cpu(),
                )
            previews.append(preview)
    finally:
        model.train(was_training)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    if writer is not None:
        writer.add_scalar("loss/validation_mse", mean_loss, iteration)
    single_slice_index = getattr(dataset, "single_slice_index", None)
    selection = (
        f"intentional train-and-preview overfit frame {single_slice_index}"
        if single_slice_index is not None
        else "first four frames in frozen validation split order"
    )
    metric_scope = (
        "one training slice reused for preview; no held-out or 3D generalization claim"
        if single_slice_index is not None
        else None
    )
    image_paths = _save_validation_preview(
        Path(output_dir),
        iteration,
        previews,
        writer,
        selection=selection,
        metric_scope=metric_scope,
    )
    if image_paths:
        tqdm.tqdm.write(
            f"Saved {len(image_paths)} validation preview groups to "
            f"{Path(output_dir).resolve()}"
        )
    return mean_loss


class NeUFTrainer:
    """Trainer for the retained fixed-geometry hash fields."""

    def __init__(self, **kwargs) -> None:
        self.dataset_path = Path(kwargs.get("dataset", DEFAULT_DATASET_PATH)).expanduser()
        self.checkpoint_value = str(kwargs.get("checkpoint", "")).strip()
        self.checkpoint_path = Path(self.checkpoint_value).expanduser()
        self.root = Path(kwargs.get("root", "runs/basic_hash")).expanduser()
        self.encoding = str(kwargs.get("encoding", "HASH")).upper()
        self.field_head = str(
            kwargs.get("field_head", NeRF.LEGACY_FIELD_HEAD)
        ).lower()
        self.intensity_activation = str(
            kwargs.get("intensity_activation", "identity")
        ).lower()
        single_slice_value = kwargs.get("single_slice_index")
        self.single_slice_index = (
            None if single_slice_value is None else int(single_slice_value)
        )
        self.phase1_image_quality = bool(
            kwargs.get("phase1_image_quality", False)
            or self.field_head != NeRF.LEGACY_FIELD_HEAD
            or self.single_slice_index is not None
        )
        self.phase1_output_dir = Path(
            kwargs.get("phase1_output_dir", "phase1_image_quality")
        ).expanduser()
        self.phase1_allow_small_dataset = bool(
            kwargs.get("phase1_allow_small_dataset", False)
        )

        self.training_mode = str(kwargs.get("training_mode", "Patch")).capitalize()
        self.patch_size = int(kwargs.get("patch_size", 64))
        self.points_per_iter = int(kwargs.get("points_per_iter", 49152))
        self.iterations = int(kwargs.get("nb_iters_max", 20000))
        self.plot_frequency = int(kwargs.get("plot_freq", 1000))
        self.save_frequency = int(kwargs.get("save_freq", 5000))
        self.seed = int(kwargs.get("seed", 3407))
        self.lr = float(kwargs.get("lr", 5e-4))
        self.lr_decay_factor = float(kwargs.get("lr_decay_factor", 0.1))
        self.grad_clip_norm = float(kwargs.get("grad_clip_norm", 1.0))

        self.hash_n_levels = int(kwargs.get("hash_n_levels", 16))
        self.hash_n_features_per_level = int(
            kwargs.get("hash_n_features_per_level", 2)
        )
        self.hash_log2_hashmap_size = int(kwargs.get("hash_log2_hashmap_size", 19))
        self.hash_base_resolution = int(kwargs.get("hash_base_resolution", 16))
        self.hash_finest_resolution = int(kwargs.get("hash_finest_resolution", 256))
        self.dual_n_levels_low = int(kwargs.get("dual_n_levels_low", 8))
        self.dual_n_levels_high = int(kwargs.get("dual_n_levels_high", 8))
        self.dual_base_resolution_low = int(
            kwargs.get("dual_base_resolution_low", 16)
        )
        self.dual_finest_resolution_low = int(
            kwargs.get("dual_finest_resolution_low", 64)
        )
        self.dual_base_resolution_high = int(
            kwargs.get("dual_base_resolution_high", 64)
        )
        self.dual_finest_resolution_high = int(
            kwargs.get("dual_finest_resolution_high", 512)
        )
        self.dual_use_gate = bool(kwargs.get("dual_use_gate", False))
        self.dual_hf_activate_ratio = float(
            kwargs.get("dual_hf_activate_ratio", 0.2)
        )
        self.dual_hf_max_weight = float(kwargs.get("dual_hf_max_weight", 1.0))

        self.stv_checkpoint = str(kwargs.get("stv_checkpoint", "")).strip()
        self.stv_tile_size = int(kwargs.get("stv_tile_size", 128))
        self.stv_weights = {
            name: float(kwargs.get(f"stv_{name}_weight", 1.0))
            for name in ("structure", "boundary", "residual")
        }
        self.stv_anatomy_config = {
            "anatomy_weight": 1.0, "edge_weight": 0.1, "spatial_weight": 0.01,
            "spatial_step_mm": 0.5, "spatial_points": 256, "edge_scale": 0.03,
        }
        self.stv_anatomy_explicit = set()
        for name in self.stv_anatomy_config:
            value = kwargs.get(f"stv_{name}")
            if value is not None:
                self.stv_anatomy_config[name] = int(value) if name == "spatial_points" else float(value)
                self.stv_anatomy_explicit.add(name)
        self.init_e1_checkpoint = str(kwargs.get("init_e1_checkpoint", "")).strip()
        self.stv_teacher = None
        self.stv_metadata = None
        self.stv_dataset_signature = None
        self.e1_initialization = None
        # 只缓存训练分区的完整真实切片；采样顺序不会改变 teacher 上下文。
        self.stv_target_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.stv_batch_targets: dict[str, torch.Tensor] | None = None

        self._validate_configuration()
        self.training_loss_name = {
            NeRF.LEGACY_FIELD_HEAD: "masked_mse",
            NeRF.MATCHED_FIELD_HEAD: "masked_mse",
            NeRF.ANATOMY_SPECKLE_FIELD_HEAD: "masked_mse",
            NeRF.FROZEN_STV_FIELD_HEAD: "masked_mse_frozen_stv_components",
        }[self.field_head]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.generator = torch.Generator(device=DEVICE)
        self.generator.manual_seed(self.seed)
        self.spatial_generator = torch.Generator(device=DEVICE)
        self.spatial_generator.manual_seed(self.seed + 1)

        self.source_control = _source_control_state()
        self.paths = self._create_run_paths()
        self.checkpoint = self._load_checkpoint()
        if self.checkpoint is not None and self.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
            # 旧训练断点继续使用原损失；新损失实验应从 E1 初始化，不能伪装成恢复。
            previous = self.checkpoint.get("frozen_stv_anatomy", {
                "anatomy_weight": 0.0, "edge_weight": 0.0, "spatial_weight": 0.0,
                "spatial_step_mm": 0.5, "spatial_points": 256, "edge_scale": 0.03,
            })
            for name in self.stv_anatomy_config:
                if name not in self.stv_anatomy_explicit:
                    self.stv_anatomy_config[name] = previous[name]
            if self.stv_anatomy_config != previous:
                raise ValueError("Resume requires identical frozen STV anatomy loss settings")
        self._validate_configuration()
        if self.field_head == NeRF.FROZEN_STV_FIELD_HEAD and any(
            self.stv_anatomy_config[name] for name in ("anatomy_weight", "edge_weight", "spatial_weight")
        ):
            self.training_loss_name = "masked_mse_frozen_stv_anatomy_v2"
        self.dataset = Dataset.open_from_save(self.dataset_path)
        if self.checkpoint is not None:
            validate_checkpoint_dataset_geometry(
                self.checkpoint,
                self.dataset,
                checkpoint_path=self.checkpoint_path,
            )

        self.phase1_splits = None
        self.pose_hash_before = None
        if self.single_slice_index is not None:
            self.phase1_splits = freeze_single_slice_manifest(
                self.dataset,
                self.dataset_path,
                self.phase1_output_dir,
                self.single_slice_index,
            )
            apply_single_slice_training_view(self.dataset, self.phase1_splits)
            self.pose_hash_before = current_pose_hash(self.dataset)
            if self.checkpoint is not None and self.checkpoint.get("single_slice_index") != self.single_slice_index:
                raise ValueError("Checkpoint was not trained with the requested single slice")
        elif self.phase1_image_quality:
            self.phase1_splits = freeze_phase1_manifests(
                self.dataset,
                self.dataset_path,
                self.phase1_output_dir,
                allow_small_dataset=self.phase1_allow_small_dataset,
            )
            apply_phase1_training_split(self.dataset, self.phase1_splits)
            self.pose_hash_before = current_pose_hash(self.dataset)
        if self.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
            self.stv_dataset_signature = {
                "frames": {
                    name: [[ref.stable_slice_id, ref.pixel_hash, ref.pose_hash] for ref in refs]
                    for name, refs in self.phase1_splits.items()
                },
                "calibration": self.dataset.physical_calibration_signature(),
                "sector_mask": self.dataset.sector_mask_signature(),
            }

        self.renderer = SliceRenderer(self.dataset)
        self.model = self._build_model()
        if self.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
            from neuf.frozen_stv import FrozenSTVTeacher

            self.stv_teacher = FrozenSTVTeacher(
                self.stv_checkpoint, DEVICE, tile_size=self.stv_tile_size,
            )
            self.stv_metadata = self.stv_teacher.metadata()
            if self.checkpoint is not None:
                previous_teacher = dict(self.checkpoint.get("frozen_stv_teacher") or {})
                current_teacher = dict(self.stv_metadata)
                # 来源可搬家，但权重 hash、采样规则和分解定义必须一致。
                previous_teacher.pop("checkpoint", None)
                current_teacher.pop("checkpoint", None)
                if previous_teacher != current_teacher:
                    raise ValueError("Resume requires the identical frozen STV teacher and configuration")
                if self.checkpoint.get("frozen_stv_weights") != self.stv_weights:
                    raise ValueError("Resume requires identical frozen STV component weights")
                if self.checkpoint.get("phase1_split_ids") != self.dataset.phase1_split_ids:
                    raise ValueError("Resume requires identical training/validation/test splits")
                if self.checkpoint.get("frozen_stv_dataset_signature") != self.stv_dataset_signature:
                    raise ValueError("Resume requires identical image content, poses, mask and calibration")
                self.e1_initialization = self.checkpoint.get("e1_initialization")
        self.optimizer = torch.optim.Adam(self.model.grad_vars(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.lr_decay_factor ** (1.0 / max(1, self.iterations)),
        )
        self.start_iteration = 0
        self.stage3_lr_scaled = False
        if self.checkpoint is not None:
            if "optimizer_state_dict" in self.checkpoint:
                self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            for name, generator in (("patch_rng_state", self.generator), ("spatial_rng_state", self.spatial_generator)):
                if name in self.checkpoint:
                    generator.set_state(self.checkpoint[name].cpu())
            self.start_iteration = int(self.checkpoint.get("start", 0))
            self.stage3_lr_scaled = bool(
                self.checkpoint.get("phase1_stage3_lr_scaled", False)
            )

        self.writer = SummaryWriter(str(self.paths.tensorboard))
        self._write_run_manifest()

    def _validate_configuration(self) -> None:
        if self.encoding not in {"HASH", "DUAL_HASH"}:
            raise ValueError("encoding must be HASH or DUAL_HASH")
        if self.field_head not in {
            NeRF.LEGACY_FIELD_HEAD,
            NeRF.MATCHED_FIELD_HEAD,
            NeRF.ANATOMY_SPECKLE_FIELD_HEAD,
            NeRF.FROZEN_STV_FIELD_HEAD,
        }:
            raise ValueError(f"Unsupported field head: {self.field_head}")
        expected_encoding = NeRF.FIELD_HEAD_ENCODINGS[self.field_head]
        if self.encoding != expected_encoding:
            raise ValueError(
                f"{self.field_head} requires --encoding {expected_encoding}, "
                f"got {self.encoding}"
            )
        if self.training_mode not in {"Patch", "Random"}:
            raise ValueError("training-mode must be Patch or Random")
        if self.single_slice_index is not None and self.single_slice_index < 0:
            raise ValueError("single-slice-index must be nonnegative")
        if self.patch_size < 1 or self.points_per_iter < 1:
            raise ValueError("patch-size and points-per-iter must be positive")
        if self.training_mode == "Patch" and self.points_per_iter < self.patch_size**2:
            raise ValueError("points-per-iter must contain at least one full patch")
        if self.phase1_image_quality and self.training_mode != "Patch":
            raise ValueError("Phase 1 requires --training-mode Patch")
        if self.phase1_image_quality and self.patch_size != 64:
            raise ValueError("Phase 1 uses the frozen patch size 64")
        if self.iterations < 1 or self.plot_frequency < 1 or self.save_frequency < 1:
            raise ValueError("iteration and output frequencies must be positive")
        if self.field_head == NeRF.FROZEN_STV_FIELD_HEAD:
            for name, value in self.stv_anatomy_config.items():
                if not math.isfinite(value) or value < 0 or ("weight" not in name and value == 0):
                    raise ValueError(f"Invalid STV anatomy setting {name}={value}")
            if self.stv_anatomy_config["edge_weight"] and (self.training_mode != "Patch" or self.patch_size < 3):
                raise ValueError("STV anatomy edge loss requires patches of at least 3 pixels")
            if not self.stv_checkpoint or not Path(self.stv_checkpoint).expanduser().is_file():
                raise ValueError("Frozen STV training requires an existing --stv-checkpoint")
            if self.stv_tile_size < 1:
                raise ValueError("stv-tile-size must be positive")
            if any(not math.isfinite(v) or v < 0 for v in self.stv_weights.values()):
                raise ValueError("STV component weights must be finite and nonnegative")
            if not any(self.stv_weights.values()):
                raise ValueError("At least one STV component weight must be positive")
            if self.checkpoint_value and self.init_e1_checkpoint:
                raise ValueError("Use --checkpoint for resume or --init-e1-checkpoint for initialization")
            if self.intensity_activation != "identity":
                raise ValueError("Frozen STV components require signed identity outputs")
        elif self.stv_checkpoint or self.init_e1_checkpoint:
            raise ValueError("STV teacher and E1 initialization apply only to frozen_stv_components_v1")

    def _create_run_paths(self) -> RunPaths:
        checkpoints = self.root / "checkpoints"
        images = self.root / "images"
        tensorboard = self.root / "tensorboard"
        latest = self.root / "latest" / "ckpt.pkl"
        self.metrics_dir = self.root / "metrics"
        self.config_dir = self.root
        if self.field_head == NeRF.FROZEN_STV_FIELD_HEAD or self.single_slice_index is not None:
            images = self.root / "plots"
            tensorboard = self.metrics_dir / "tensorboard"
            latest = checkpoints / "latest.pkl"
            self.config_dir = self.root / "run_config"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        for path in (checkpoints, images, tensorboard, latest.parent):
            path.mkdir(parents=True, exist_ok=True)
        return RunPaths(self.root, checkpoints, images, tensorboard, latest)

    def _load_checkpoint(self) -> dict | None:
        if not self.checkpoint_value:
            return None
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        return torch.load(
            self.checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

    def _build_model(self) -> NeRF:
        if self.checkpoint is not None:
            model = NeRF(self.checkpoint)
            if model.field_head != self.field_head:
                raise ValueError(
                    f"CLI field head {self.field_head} does not match checkpoint "
                    f"{model.field_head}"
                )
            if model.encoding_type != self.encoding:
                raise ValueError(
                    f"CLI encoding {self.encoding} does not match checkpoint "
                    f"{model.encoding_type}"
                )
            return model

        model = NeRF(
            field_head=self.field_head,
            intensity_activation=self.intensity_activation,
        )
        bounding_box = self.dataset.get_bounding_box()
        if self.encoding == "HASH":
            model.init_hash_encoding(
                bounding_box,
                n_levels=self.hash_n_levels,
                n_features_per_level=self.hash_n_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution=self.hash_base_resolution,
                finest_resolution=self.hash_finest_resolution,
            )
        else:
            model.init_dual_encoding(
                bounding_box=bounding_box,
                n_levels_low=self.dual_n_levels_low,
                n_levels_high=self.dual_n_levels_high,
                n_features_per_level=self.hash_n_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution_low=self.dual_base_resolution_low,
                finest_resolution_low=self.dual_finest_resolution_low,
                base_resolution_high=self.dual_base_resolution_high,
                finest_resolution_high=self.dual_finest_resolution_high,
                use_gate=False,
                hf_activate_ratio=self.dual_hf_activate_ratio,
                hf_max_weight=self.dual_hf_max_weight,
            )
        model.init_model()
        if self.init_e1_checkpoint:
            source_path = Path(self.init_e1_checkpoint).expanduser()
            source = torch.load(source_path, map_location=DEVICE, weights_only=False)
            validate_checkpoint_dataset_geometry(source, self.dataset, checkpoint_path=source_path)
            for key, expected in (
                ("dataset_physical_calibration", self.dataset.physical_calibration_signature()),
                ("phase1_pose_hash_current", self.pose_hash_before),
                ("phase1_split_ids", getattr(self.dataset, "phase1_split_ids", None)),
            ):
                if key in source and source[key] is not None and source[key] != expected:
                    raise ValueError(f"E1 initialization has incompatible {key}")
            model.load_e1_initialization(source)
            with source_path.open("rb") as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.e1_initialization = {"path": str(source_path.resolve()), "sha256": digest.hexdigest()}
        return model

    def _write_run_manifest(self) -> None:
        payload = {
            "dataset": str(self.dataset_path.resolve()),
            "checkpoint": str(self.checkpoint_path) if self.checkpoint is not None else None,
            "encoding": self.encoding,
            "field_head": self.field_head,
            "phase1_image_quality": self.phase1_image_quality,
            "phase1_output_dir": str(self.phase1_output_dir.resolve()),
            "single_slice_index": self.single_slice_index,
            "training_mode": self.training_mode,
            "patch_size": self.patch_size,
            "points_per_iter": self.points_per_iter,
            "iterations": self.iterations,
            "plot_frequency": self.plot_frequency,
            "validation_preview_count": VALIDATION_PREVIEW_COUNT,
            "seed": self.seed,
            "learning_rate": self.lr,
            "training_loss": self.training_loss_name,
            "parameter_counts": self.model.parameter_counts(),
            "device": str(DEVICE),
            "source_control": self.source_control,
            "frozen_stv_teacher": self.stv_metadata,
            "frozen_stv_weights": self.stv_weights if self.stv_teacher is not None else None,
            "frozen_stv_anatomy": self.stv_anatomy_config if self.stv_teacher is not None else None,
            "inference_composition": "S+B+alpha*R, alpha in [0,1]",
            "e1_initialization": self.e1_initialization,
        }
        (self.config_dir / "run_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))

    def _set_teacher_targets(self, indices: torch.Tensor) -> None:
        """按同一组训练像素索引取伪目标，不读取 validation/test 图像。"""
        if self.stv_teacher is None:
            return
        flat_indices = indices.detach().cpu().reshape(-1)
        frame_indices = flat_indices // self.dataset.pixels_per_slice
        pixel_indices = flat_indices % self.dataset.pixels_per_slice
        names = (*self.stv_weights, "edge_confidence", "smooth_weight")
        targets = {name: torch.empty((len(flat_indices), 1)) for name in names}
        frame_mask = self.dataset.get_sector_mask(device=DEVICE).reshape(
            1, 1, self.dataset.px_height, self.dataset.px_width,
        )
        for frame in torch.unique(frame_indices).tolist():
            if frame not in self.stv_target_cache:
                image = self.dataset.get_slice_pixels(frame).reshape_as(frame_mask).float()
                result = self.stv_teacher.predict_frame(image, frame_mask)
                from neuf.frozen_stv import anatomy_smooth_weight

                spacing = min(self.dataset.roi_px_size_width_mm, self.dataset.roi_px_size_height_mm)
                result["smooth_weight"] = anatomy_smooth_weight(
                    result["anatomy"], result["edge_confidence"], frame_mask,
                    edge_scale=self.stv_anatomy_config["edge_scale"],
                    margin_pixels=math.ceil(self.stv_anatomy_config["spatial_step_mm"] / spacing),
                )
                self.stv_target_cache[frame] = {
                    name: result[name].reshape(-1, 1).detach().cpu().clone()
                    for name in names
                }
            selected = frame_indices == frame
            for name, values in targets.items():
                values[selected] = self.stv_target_cache[frame][name][pixel_indices[selected]]
        self.stv_batch_targets = {name: value.to(DEVICE) for name, value in targets.items()}

    def _sample_random(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_total = self.dataset.valid_pixel_count * len(self.dataset.slices)
        ranks = torch.randint(
            valid_total,
            (self.points_per_iter,),
            device=DEVICE,
            generator=self.generator,
        )
        indices = self.dataset.map_valid_training_ranks(ranks)
        self._set_teacher_targets(indices)
        target = self.dataset.pixels[indices].unsqueeze(1)
        points = self.dataset.points[indices].unsqueeze(1)
        viewdirs = self.dataset.viewdirs[indices].unsqueeze(1)
        mask = torch.ones_like(target, dtype=torch.bool)
        return target, points, viewdirs, mask

    def _sample_patches(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_area = self.patch_size**2
        patch_count = self.points_per_iter // patch_area
        height = int(self.dataset.px_height)
        width = int(self.dataset.px_width)
        max_row = height - self.patch_size + 1
        max_col = width - self.patch_size + 1
        if max_row <= 0 or max_col <= 0:
            raise ValueError("patch-size is larger than the ultrasound frame")

        sector = self.dataset.get_sector_mask(device=DEVICE)
        center = self.patch_size // 2
        center_mask = sector[center:center + max_row, center:center + max_col]
        valid_origins = torch.nonzero(center_mask, as_tuple=False)
        if not len(valid_origins):
            raise ValueError("No patch centres lie inside the ultrasound sector")

        slice_indices = torch.randint(
            len(self.dataset.slices),
            (patch_count,),
            device=DEVICE,
            generator=self.generator,
        )
        origin_indices = torch.randint(
            len(valid_origins),
            (patch_count,),
            device=DEVICE,
            generator=self.generator,
        )
        origins = valid_origins[origin_indices]
        slice_starts = torch.as_tensor(
            [item.start for item in self.dataset.slices],
            dtype=torch.long,
            device=DEVICE,
        )
        row_offsets = torch.arange(self.patch_size, device=DEVICE)[:, None] * width
        col_offsets = torch.arange(self.patch_size, device=DEVICE)[None, :]
        patch_offsets = (row_offsets + col_offsets).reshape(1, -1)
        starts = (
            slice_starts[slice_indices]
            + origins[:, 0] * width
            + origins[:, 1]
        )[:, None]
        indices = (starts + patch_offsets).reshape(-1)
        local_indices = torch.remainder(indices, self.dataset.pixels_per_slice)
        mask = self.dataset.get_sector_mask(flatten=True, device=DEVICE)[local_indices]
        self._set_teacher_targets(indices)
        return (
            self.dataset.pixels[indices].unsqueeze(1),
            self.dataset.points[indices].unsqueeze(1),
            self.dataset.viewdirs[indices].unsqueeze(1),
            mask,
        )

    def _sample_batch(self):
        return self._sample_patches() if self.training_mode == "Patch" else self._sample_random()

    def _query_batch(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.field_head in {NeRF.ANATOMY_SPECKLE_FIELD_HEAD, NeRF.FROZEN_STV_FIELD_HEAD}:
            components = self.renderer.query_point_components(
                self.model,
                points,
                viewdirs,
                alpha=1.0,
            )
            return components["intensity"], components
        prediction = self.renderer.query_points(self.model, points, viewdirs)
        return prediction, None

    def _training_loss(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        components: dict[str, torch.Tensor] | None,
        stage: int,
        points: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del stage
        reconstruction = masked_mean((prediction - target).square(), mask)
        if self.stv_teacher is not None:
            from neuf.frozen_stv import (
                anatomy_spatial_loss, frozen_stv_anatomy_loss, frozen_stv_component_loss,
            )

            if components is None or self.stv_batch_targets is None:
                raise RuntimeError("Frozen STV training requires sampled component targets")
            component_loss, metrics = frozen_stv_component_loss(
                components, self.stv_batch_targets, mask,
                **{f"{name}_weight": value for name, value in self.stv_weights.items()},
            )
            cfg = self.stv_anatomy_config
            anatomy_loss, anatomy_metrics = frozen_stv_anatomy_loss(
                components, self.stv_batch_targets, mask, patch_size=self.patch_size,
                anatomy_weight=cfg["anatomy_weight"], edge_weight=cfg["edge_weight"],
            )
            spatial = reconstruction * 0
            spatial_metrics = {}
            if cfg["spatial_weight"]:
                if points is None:
                    raise ValueError("STV spatial loss requires sampled physical points")
                spatial, spatial_metrics = anatomy_spatial_loss(
                    self.model, points, components["anatomy"], mask,
                    self.stv_batch_targets["smooth_weight"], self.dataset.get_bounding_box(),
                    step_mm=cfg["spatial_step_mm"], max_points=cfg["spatial_points"],
                    generator=self.spatial_generator,
                )
            loss = reconstruction + component_loss + anatomy_loss + cfg["spatial_weight"] * spatial
            return loss, {"reconstruction": reconstruction, **metrics, **anatomy_metrics, **spatial_metrics}
        return reconstruction, {"reconstruction": reconstruction}

    def _validate(self, iteration: int) -> float:
        return render_validation_preview(
            self.model,
            self.dataset,
            self.renderer,
            self.paths.images,
            iteration,
            self.writer,
        )

    def _checkpoint_payload(self, iteration: int) -> dict:
        payload = self.model.get_save_dict()
        payload.update(
            {
                "start": int(iteration),
                "seed": self.seed,
                "baked": True,
                "baked_dataset_file": str(self.dataset_path.resolve()),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "ultrasound_sector_mask": self.dataset.sector_mask_signature(),
                "dataset_physical_calibration": self.dataset.physical_calibration_signature(),
                "phase1_image_quality": self.phase1_image_quality,
                "single_slice_index": self.single_slice_index,
                "training_loss": self.training_loss_name,
                "training_mode": self.training_mode,
                "patch_size": self.patch_size,
                "points_per_iter": self.points_per_iter,
                "iterations": self.iterations,
                "plot_frequency": self.plot_frequency,
                "validation_preview_count": VALIDATION_PREVIEW_COUNT,
                "phase1_split_ids": getattr(self.dataset, "phase1_split_ids", None),
                "phase1_pose_hash_before": self.pose_hash_before,
                "phase1_pose_hash_current": (
                    current_pose_hash(self.dataset)
                    if self.phase1_image_quality
                    else None
                ),
                "phase1_stage3_lr_scaled": self.stage3_lr_scaled,
                "source_control": self.source_control,
                "frozen_stv_teacher": self.stv_metadata,
                "frozen_stv_dataset_signature": self.stv_dataset_signature,
                "frozen_stv_weights": self.stv_weights if self.stv_teacher is not None else None,
                "frozen_stv_anatomy": self.stv_anatomy_config if self.stv_teacher is not None else None,
                "patch_rng_state": self.generator.get_state(),
                "spatial_rng_state": self.spatial_generator.get_state(),
                "e1_initialization": self.e1_initialization,
            }
        )
        return payload

    def _save_checkpoint(self, iteration: int) -> None:
        payload = self._checkpoint_payload(iteration)
        numbered = self.paths.checkpoints / f"ckpt_{iteration}.pkl"
        torch.save(payload, numbered)
        torch.save(payload, self.paths.latest_checkpoint)
        print(f"Saved checkpoint: {self.paths.latest_checkpoint.resolve()}")

    def _write_geometry_integrity(self) -> None:
        if not self.phase1_image_quality or self.pose_hash_before is None:
            return
        pose_hash_after = current_pose_hash(self.dataset)
        payload = {
            "pose_hash_before": self.pose_hash_before,
            "pose_hash_after": pose_hash_after,
            "unchanged": pose_hash_after == self.pose_hash_before,
            "field_head": self.field_head,
            "seed": self.seed,
        }
        (self.config_dir / "phase1_geometry_integrity.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not payload["unchanged"]:
            raise RuntimeError("Phase 1 fixed geometry changed during training")

    def run(self) -> None:
        started = time.perf_counter()
        start_timestamp = datetime.now().astimezone().isoformat()
        completed = self.start_iteration
        step_time_total = 0.0
        status = "failed"
        timing_path = self.metrics_dir / "timing.csv"
        timing_file = timing_path.open("a" if self.start_iteration else "w", newline="")
        timing = csv.writer(timing_file)
        if timing_file.tell() == 0:
            timing.writerow(["epoch", "global_step", "step_time_sec", "elapsed_sec", "loss"])
        progress = tqdm.trange(
            self.start_iteration,
            self.iterations,
            desc="Training",
            dynamic_ncols=True,
        )
        try:
            for iteration in progress:
                step_started = time.perf_counter()
                training_progress = iteration / max(1, self.iterations)
                self.model.training_progress = training_progress
                stage = self.model.set_phase1_training_stage(training_progress)
                stage_label = (
                    "joint"
                    if self.field_head in {NeRF.ANATOMY_SPECKLE_FIELD_HEAD, NeRF.FROZEN_STV_FIELD_HEAD}
                    else str(stage)
                )
                if (
                    self.phase1_image_quality
                    and self.field_head not in {NeRF.ANATOMY_SPECKLE_FIELD_HEAD, NeRF.FROZEN_STV_FIELD_HEAD}
                    and stage == 3
                    and not self.stage3_lr_scaled
                ):
                    for group in self.optimizer.param_groups:
                        group["lr"] *= 0.1
                    self.stage3_lr_scaled = True

                target, points, viewdirs, mask = self._sample_batch()
                prediction, components = self._query_batch(points, viewdirs)
                loss, loss_components = self._training_loss(
                    target,
                    prediction,
                    mask,
                    components,
                    stage,
                    points=points,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                    )
                self.optimizer.step()
                self.scheduler.step()

                step = iteration + 1
                completed = step
                loss_value = float(loss.detach().cpu())
                step_seconds = time.perf_counter() - step_started
                step_time_total += step_seconds
                timing.writerow([1, step, step_seconds, time.perf_counter() - started, loss_value])
                timing_file.flush()
                progress.set_postfix(loss=f"{loss_value:.6f}", stage=stage_label, sec=f"{step_seconds:.2f}")
                self.writer.add_scalar("loss/train", loss_value, step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], step)
                for name, value in loss_components.items():
                    self.writer.add_scalar(f"loss/components/{name}", float(value.detach().cpu()), step)

                if step % self.plot_frequency == 0 or step == self.iterations:
                    validation_loss = self._validate(step)
                    tqdm.tqdm.write(
                        f"iteration={step} train={loss_value:.6f} "
                        f"validation_mse={validation_loss:.6f}"
                    )
                if step % self.save_frequency == 0 or step == self.iterations:
                    self._save_checkpoint(step)
            status = "complete"
        finally:
            elapsed = time.perf_counter() - started
            timing_file.close()
            self.writer.close()
            progress.close()
            summary = {
                "configured_epochs": 1, "completed_epochs": int(status == "complete"),
                "configured_total_steps": self.iterations, "completed_total_steps": completed,
                "previous_session_steps": self.start_iteration,
                "current_session_steps": completed - self.start_iteration,
                "start_timestamp": start_timestamp,
                "end_timestamp": datetime.now().astimezone().isoformat(),
                "total_wall_time_sec": elapsed,
                "mean_step_time_sec": step_time_total / max(1, completed - self.start_iteration),
                "status": status,
                "final_result_path": (
                    str(self.paths.latest_checkpoint.resolve())
                    if self.paths.latest_checkpoint.is_file() else None
                ),
            }
            (self.metrics_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8",
            )
            print(f"Elapsed seconds: {elapsed:.1f}")
            self._write_geometry_integrity()


# Backward-compatible public name used by a few local scripts.
NeUF = NeUFTrainer


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the basic single-head HashGrid or the current fixed-geometry "
            "Phase 1 DUAL_HASH fields."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--evaluate-only", action="store_true", help="仅用 NeUF checkpoint 评价固定验证集与 alpha，不加载 teacher")
    parser.add_argument("--stv-checkpoint", default="", help="训练专用的冻结 Neural STV checkpoint")
    parser.add_argument("--stv-tile-size", type=int, default=128)
    parser.add_argument("--init-e1-checkpoint", default="", help="仅初始化编码器及三分量主干，不恢复 E1 优化器")
    for name in ("structure", "boundary", "residual"):
        parser.add_argument(f"--stv-{name}-weight", type=float, default=1.0)
    for name in ("anatomy-weight", "edge-weight", "spatial-weight", "spatial-step-mm", "edge-scale"):
        parser.add_argument(f"--stv-{name}", type=float, default=None)
    parser.add_argument("--stv-spatial-points", type=int, default=None)
    parser.add_argument("--root", default="runs/basic_hash")
    parser.add_argument("--encoding", type=str.upper, choices=["HASH", "DUAL_HASH"], default="HASH")
    parser.add_argument(
        "--field-head",
        choices=[
            NeRF.LEGACY_FIELD_HEAD,
            NeRF.MATCHED_FIELD_HEAD,
            NeRF.ANATOMY_SPECKLE_FIELD_HEAD,
            NeRF.FROZEN_STV_FIELD_HEAD,
        ],
        default=NeRF.LEGACY_FIELD_HEAD,
    )
    parser.add_argument(
        "--intensity-activation",
        choices=["identity", "sigmoid"],
        default="identity",
    )
    parser.add_argument("--phase1-image-quality", action="store_true")
    parser.add_argument("--phase1-output-dir", default="phase1_image_quality")
    parser.add_argument("--phase1-allow-small-dataset", action="store_true")
    parser.add_argument(
        "--single-slice-index",
        type=int,
        default=None,
        help="仅训练并预览该原始 frame index；这是同图拟合诊断，不是独立验证",
    )
    parser.add_argument("--training-mode", choices=["Patch", "Random"], default="Patch")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--points-per-iter", type=int, default=49152)
    parser.add_argument("--nb-iters-max", type=int, default=20000)
    parser.add_argument("--plot-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    hash_group = parser.add_argument_group("Basic HashGrid")
    hash_group.add_argument("--hash-n-levels", type=int, default=16)
    hash_group.add_argument("--hash-n-features-per-level", type=int, default=2)
    hash_group.add_argument("--hash-log2-hashmap-size", type=int, default=19)
    hash_group.add_argument("--hash-base-resolution", "--hash-n-min", dest="hash_base_resolution", type=int, default=16)
    hash_group.add_argument("--hash-finest-resolution", "--hash-n-max", dest="hash_finest_resolution", type=int, default=256)

    dual_group = parser.add_argument_group("Current Phase 1 DUAL_HASH")
    dual_group.add_argument("--dual-n-levels-low", type=int, default=8)
    dual_group.add_argument("--dual-n-levels-high", type=int, default=8)
    dual_group.add_argument("--dual-base-resolution-low", type=int, default=16)
    dual_group.add_argument("--dual-finest-resolution-low", type=int, default=64)
    dual_group.add_argument("--dual-base-resolution-high", type=int, default=64)
    dual_group.add_argument("--dual-finest-resolution-high", type=int, default=512)
    dual_group.add_argument("--dual-no-gate", dest="dual_use_gate", action="store_false", default=False)
    dual_group.add_argument("--dual-hf-activate-ratio", type=float, default=0.2)
    dual_group.add_argument("--dual-hf-max-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.evaluate_only:
        if not args.checkpoint:
            raise ValueError("--evaluate-only requires --checkpoint")
        root = Path(args.root).expanduser()
        dataset_path = Path(args.dataset).expanduser()
        checkpoint_path = Path(args.checkpoint).expanduser()
        started = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        dataset = Dataset.open_from_save(dataset_path)
        validate_checkpoint_dataset_geometry(checkpoint, dataset, checkpoint_path=checkpoint_path)
        splits = freeze_phase1_manifests(dataset, dataset_path, root / "run_config" / "splits")
        apply_phase1_training_split(dataset, splits)
        if checkpoint.get("phase1_split_ids") != dataset.phase1_split_ids:
            raise ValueError("Evaluation requires the checkpoint's identical validation split")
        model = NeRF(checkpoint).eval()
        model.training_progress = 1.0
        mse = render_validation_preview(model, dataset, SliceRenderer(dataset), root / "plots", int(checkpoint.get("start", 0)))
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "complete", "checkpoint": str(checkpoint_path.resolve()),
            "dataset": str(dataset_path.resolve()), "validation_count": len(dataset.slices_valid),
            "alpha1_validation_mse": mse, "teacher_used": False,
            "comparison": "existing weights, inference alpha sweep; no training update",
            "elapsed_seconds": time.perf_counter() - started,
        }
        (metrics_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    trainer = NeUFTrainer(**vars(args))
    trainer.run()


if __name__ == "__main__":
    main()
