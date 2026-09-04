from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.phase1_data import (
    apply_phase1_training_split,
    current_pose_hash,
    freeze_phase1_manifests,
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


def _save_validation_preview(
    output_dir: Path,
    iteration: int,
    previews: list[ValidationPreview],
    writer: SummaryWriter | None = None,
) -> list[Path]:
    """Save one GT/prediction/error figure and tensor file per validation slice."""
    if not previews:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    has_components = all(item.anatomy is not None for item in previews)
    columns = 5 if has_components else 3
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
            previews.append(preview)
    finally:
        model.train(was_training)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    if writer is not None:
        writer.add_scalar("loss/validation_mse", mean_loss, iteration)
    image_paths = _save_validation_preview(Path(output_dir), iteration, previews, writer)
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
        self.phase1_image_quality = bool(
            kwargs.get("phase1_image_quality", False)
            or self.field_head != NeRF.LEGACY_FIELD_HEAD
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

        self._validate_configuration()
        self.training_loss_name = {
            NeRF.LEGACY_FIELD_HEAD: "masked_mse",
            NeRF.MATCHED_FIELD_HEAD: "masked_mse",
            NeRF.ANATOMY_SPECKLE_FIELD_HEAD: "masked_mse",
        }[self.field_head]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.generator = torch.Generator(device=DEVICE)
        self.generator.manual_seed(self.seed)

        self.source_control = _source_control_state()
        self.paths = self._create_run_paths()
        self.checkpoint = self._load_checkpoint()
        self.dataset = Dataset.open_from_save(self.dataset_path)
        if self.checkpoint is not None:
            validate_checkpoint_dataset_geometry(
                self.checkpoint,
                self.dataset,
                checkpoint_path=self.checkpoint_path,
            )

        self.phase1_splits = None
        self.pose_hash_before = None
        if self.phase1_image_quality:
            self.phase1_splits = freeze_phase1_manifests(
                self.dataset,
                self.dataset_path,
                self.phase1_output_dir,
                allow_small_dataset=self.phase1_allow_small_dataset,
            )
            apply_phase1_training_split(self.dataset, self.phase1_splits)
            self.pose_hash_before = current_pose_hash(self.dataset)

        self.renderer = SliceRenderer(self.dataset)
        self.model = self._build_model()
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

    def _create_run_paths(self) -> RunPaths:
        checkpoints = self.root / "checkpoints"
        images = self.root / "images"
        tensorboard = self.root / "tensorboard"
        latest = self.root / "latest" / "ckpt.pkl"
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
        return model

    def _write_run_manifest(self) -> None:
        payload = {
            "dataset": str(self.dataset_path.resolve()),
            "checkpoint": str(self.checkpoint_path) if self.checkpoint is not None else None,
            "encoding": self.encoding,
            "field_head": self.field_head,
            "phase1_image_quality": self.phase1_image_quality,
            "phase1_output_dir": str(self.phase1_output_dir.resolve()),
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
        }
        (self.paths.root / "run_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))

    def _sample_random(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_total = self.dataset.valid_pixel_count * len(self.dataset.slices)
        ranks = torch.randint(
            valid_total,
            (self.points_per_iter,),
            device=DEVICE,
            generator=self.generator,
        )
        indices = self.dataset.map_valid_training_ranks(ranks)
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
        if self.field_head == NeRF.ANATOMY_SPECKLE_FIELD_HEAD:
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del components, stage
        reconstruction = masked_mean((prediction - target).square(), mask)
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
        (self.paths.root / "phase1_geometry_integrity.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not payload["unchanged"]:
            raise RuntimeError("Phase 1 fixed geometry changed during training")

    def run(self) -> None:
        started = time.time()
        progress = tqdm.trange(
            self.start_iteration,
            self.iterations,
            desc="Training",
            dynamic_ncols=True,
        )
        try:
            for iteration in progress:
                training_progress = iteration / max(1, self.iterations)
                self.model.training_progress = training_progress
                stage = self.model.set_phase1_training_stage(training_progress)
                stage_label = (
                    "joint"
                    if self.field_head == NeRF.ANATOMY_SPECKLE_FIELD_HEAD
                    else str(stage)
                )
                if (
                    self.phase1_image_quality
                    and self.field_head != NeRF.ANATOMY_SPECKLE_FIELD_HEAD
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
                loss_value = float(loss.detach().cpu())
                progress.set_postfix(loss=f"{loss_value:.6f}", stage=stage_label)
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
        finally:
            self._write_geometry_integrity()
            self.writer.close()
            progress.close()
            print(f"Elapsed seconds: {time.time() - started:.1f}")


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
    parser.add_argument("--root", default="runs/basic_hash")
    parser.add_argument("--encoding", type=str.upper, choices=["HASH", "DUAL_HASH"], default="HASH")
    parser.add_argument(
        "--field-head",
        choices=[
            NeRF.LEGACY_FIELD_HEAD,
            NeRF.MATCHED_FIELD_HEAD,
            NeRF.ANATOMY_SPECKLE_FIELD_HEAD,
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
    trainer = NeUFTrainer(**vars(args))
    trainer.run()


if __name__ == "__main__":
    main()
