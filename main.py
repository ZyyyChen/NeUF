from __future__ import annotations

import argparse
import csv
import datetime
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from nerf_network import NeRF
from slice_renderer import SliceRenderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_PATH = (
    "/home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/"
    "Recalage/Patient0/us_recal_original/baked_dataset.pkl"
)
VALIDATION_SLICE_NAMES = ("A1", "B1", "C1", "D1")
DEFAULT_HASH_N_LEVELS = 16
DEFAULT_HASH_N_FEATURES_PER_LEVEL = 2
DEFAULT_HASH_LOG2_HASHMAP_SIZE = 19
DEFAULT_HASH_BASE_RESOLUTION = 16
DEFAULT_HASH_FINEST_RESOLUTION = 256


@dataclass(frozen=True)
class RunPaths:
    log_dir: Path
    checkpoint_dir: Path
    image_dir: Path
    loss_dir: Path
    latest_checkpoint: Path


class NeUF:
    def __init__(self, **kwargs):
        self.grad_weight = float(kwargs.get("grad_weight", 0.1))
        self.grad_clip_norm = float(kwargs.get("grad_clip_norm", 1.0))
        self.seed = int(kwargs.get("seed", 19981708))
        self.N_iters = int(kwargs.get("nb_iters_max", 10000))
        self.i_plot = int(kwargs.get("plot_freq", 100))
        self.i_save = int(kwargs.get("save_freq", 100))
        self.baked_dataset = bool(kwargs.get("baked_dataset", True))
        self.training_mode = kwargs.get("training_mode", "Random")
        self.points_per_iter = int(kwargs.get("points_per_iter", 50000))
        self.jitter_training = bool(kwargs.get("jitter_training", False))
        self.patch_size = int(kwargs.get("patch_size", 32))
        self.grad_blur_kernel_size = int(kwargs.get("grad_blur_kernel_size", 6))
        self.grad_blur_sigma = float(kwargs.get("grad_blur_sigma", 1.5))
        self.tv_weight = float(kwargs.get("tv_weight", 15))
        self.slice_mix_interval = int(kwargs.get("slice_mix_interval", 10))
        self.smoothness_delta = float(kwargs.get("smoothness_delta", 0.3))
        self.lr = float(kwargs.get("lr", 5e-4))
        self.lr_decay_factor = float(kwargs.get("lr_decay_factor", 0.1))
        self.encoding = kwargs.get("encoding", "None")
        self.datasetFolder = kwargs.get("dataset", DEFAULT_DATASET_PATH)
        self.ckptFile = kwargs.get("checkpoint", "")
        self.rootPoint = Path(kwargs.get("root", ".")).expanduser()

        self.hash_n_levels = int(kwargs.get("hash_n_levels", DEFAULT_HASH_N_LEVELS))
        self.hash_n_features_per_level = int(
            kwargs.get("hash_n_features_per_level", DEFAULT_HASH_N_FEATURES_PER_LEVEL)
        )
        self.hash_log2_hashmap_size = int(
            kwargs.get("hash_log2_hashmap_size", DEFAULT_HASH_LOG2_HASHMAP_SIZE)
        )
        self.hash_base_resolution = int(
            kwargs.get("hash_base_resolution", DEFAULT_HASH_BASE_RESOLUTION)
        )
        self.hash_finest_resolution = int(
            kwargs.get("hash_finest_resolution", DEFAULT_HASH_FINEST_RESOLUTION)
        )
        self.dual_pe_type = kwargs.get("dual_pe_type", "hash")
        self.dual_n_levels_low = int(kwargs.get("dual_n_levels_low", 8))
        self.dual_n_levels_high = int(kwargs.get("dual_n_levels_high", 8))
        self.dual_finest_resolution_low = int(kwargs.get("dual_finest_resolution_low", 64))
        self.dual_finest_resolution_high = int(kwargs.get("dual_finest_resolution_high", 512))
        self.dual_base_resolution_low = int(kwargs.get("dual_base_resolution_low", 16))
        self.dual_base_resolution_high = int(kwargs.get("dual_base_resolution_high", 64))
        self.dual_sigma_low = float(kwargs.get("dual_sigma_low", 1.0))
        self.dual_sigma_high = float(kwargs.get("dual_sigma_high", 20.0))
        self.dual_n_freq = int(kwargs.get("dual_n_freq", 64))
        self.dual_use_gate = bool(kwargs.get("dual_use_gate", True))
        self.dual_hf_activate_ratio = float(kwargs.get("dual_hf_activate_ratio", 0.6))
        self.dual_hf_max_weight = float(kwargs.get("dual_hf_max_weight", 1.0))
        self.dual_sparsity_weight = float(kwargs.get("dual_sparsity_weight", 0.01))
        self.dual_gate_weight = float(kwargs.get("dual_gate_weight", 0.5))
        self.progressive_training = bool(kwargs.get("progressive_training", False))
        self.progressive_start_levels = int(kwargs.get("progressive_start_levels", 4))
        self.progressive_step_interval = int(kwargs.get("progressive_step_interval", 1000))

        self._validate_configuration()

        self.dataset: Dataset
        self.nerf: NeRF
        self.optimizer: torch.optim.Optimizer
        self.slice_renderer: SliceRenderer
        self.start = 0

        self.criterion = torch.nn.L1Loss()
        self.reconstruction_criterion = torch.nn.MSELoss()
        self.scharr_x, self.scharr_y = self._build_scharr_kernels()
        self.previous_validation_slices: dict[str, torch.Tensor] = {}
        self.gt_saved = False
        self.tb_reference_images_logged = False

        self.random_permutation: Optional[torch.Tensor] = None
        self.random_start_index = 0
        self.training_slice_starts: Optional[torch.Tensor] = None
        self._current_points = None
        self._current_viewdirs = None
        self._current_target_slice = None

        checkpoint = self._load_checkpoint(self.ckptFile)
        if checkpoint is not None:
            self._initialize_from_checkpoint(checkpoint)
        else:
            self._initialize_from_scratch()
        self._print_hash_configuration()

        self.slice_renderer = SliceRenderer(self.dataset)
        self.training_slice_starts = self._build_training_slice_starts()
        self._validate_dataset_configuration()
        self.run_paths = self._create_run_paths()
        self.logPath = str(self.run_paths.log_dir)

        self.tb_writer = SummaryWriter(log_dir=str(self.run_paths.log_dir / "tensorboard"))
        self.loss_csv_path = self.run_paths.loss_dir / "loss_history.csv"
        self._initialize_loss_csv()

    def _validate_configuration(self) -> None:
        if self.N_iters < 0:
            raise ValueError(f"nb_iters_max must be >= 0, got {self.N_iters}")
        if self.i_plot <= 0:
            raise ValueError(f"plot_freq must be >= 1, got {self.i_plot}")
        if self.i_save <= 0:
            raise ValueError(f"save_freq must be >= 1, got {self.i_save}")
        if self.training_mode not in {"Random", "Slice", "Patch"}:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        if self.points_per_iter <= 0:
            raise ValueError(f"points_per_iter must be >= 1, got {self.points_per_iter}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        if self.grad_blur_kernel_size <= 0:
            raise ValueError(
                f"grad_blur_kernel_size must be >= 1, got {self.grad_blur_kernel_size}"
            )
        if self.grad_blur_sigma <= 0:
            raise ValueError(f"grad_blur_sigma must be > 0, got {self.grad_blur_sigma}")
        if self.tv_weight < 0:
            raise ValueError(f"tv_weight must be >= 0, got {self.tv_weight}")
        if self.grad_clip_norm < 0:
            raise ValueError(f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0 < self.lr_decay_factor <= 1):
            raise ValueError(
                f"lr_decay_factor must be in (0, 1], got {self.lr_decay_factor}"
            )
        if self.hash_n_levels < 2:
            raise ValueError(f"hash_n_levels must be >= 2, got {self.hash_n_levels}")
        if self.hash_n_features_per_level <= 0:
            raise ValueError(
                "hash_n_features_per_level must be >= 1, "
                f"got {self.hash_n_features_per_level}"
            )
        if self.hash_log2_hashmap_size <= 0:
            raise ValueError(
                f"hash_log2_hashmap_size must be >= 1, got {self.hash_log2_hashmap_size}"
            )
        if self.hash_base_resolution <= 0:
            raise ValueError(
                f"hash_base_resolution must be >= 1, got {self.hash_base_resolution}"
            )
        if self.hash_finest_resolution < self.hash_base_resolution:
            raise ValueError(
                "hash_finest_resolution must be >= hash_base_resolution, "
                f"got {self.hash_finest_resolution} < {self.hash_base_resolution}"
            )
        if self.progressive_start_levels <= 0:
            raise ValueError(
                "progressive_start_levels must be >= 1, "
                f"got {self.progressive_start_levels}"
            )
        if self.progressive_step_interval <= 0:
            raise ValueError(
                "progressive_step_interval must be >= 1, "
                f"got {self.progressive_step_interval}"
            )
        if self.dual_pe_type not in {"hash", "fourier"}:
            raise ValueError(f"dual_pe_type must be 'hash' or 'fourier', got {self.dual_pe_type}")
        if self.dual_n_levels_low <= 0 or self.dual_n_levels_high <= 0:
            raise ValueError(
                "dual_n_levels_low and dual_n_levels_high must be >= 1, "
                f"got {self.dual_n_levels_low}, {self.dual_n_levels_high}"
            )
        if self.dual_base_resolution_low <= 0 or self.dual_base_resolution_high <= 0:
            raise ValueError(
                "dual base resolutions must be >= 1, "
                f"got {self.dual_base_resolution_low}, {self.dual_base_resolution_high}"
            )
        if self.dual_finest_resolution_low < self.dual_base_resolution_low:
            raise ValueError(
                "dual_finest_resolution_low must be >= dual_base_resolution_low, "
                f"got {self.dual_finest_resolution_low} < {self.dual_base_resolution_low}"
            )
        if self.dual_finest_resolution_high < self.dual_base_resolution_high:
            raise ValueError(
                "dual_finest_resolution_high must be >= dual_base_resolution_high, "
                f"got {self.dual_finest_resolution_high} < {self.dual_base_resolution_high}"
            )
        if self.dual_sigma_low <= 0 or self.dual_sigma_high <= 0:
            raise ValueError(
                "dual sigma values must be > 0, "
                f"got {self.dual_sigma_low}, {self.dual_sigma_high}"
            )
        if self.dual_n_freq <= 0:
            raise ValueError(f"dual_n_freq must be >= 1, got {self.dual_n_freq}")
        if not 0 <= self.dual_hf_activate_ratio <= 1:
            raise ValueError(
                "dual_hf_activate_ratio must be in [0, 1], "
                f"got {self.dual_hf_activate_ratio}"
            )
        if self.dual_hf_max_weight < 0:
            raise ValueError(
                f"dual_hf_max_weight must be >= 0, got {self.dual_hf_max_weight}"
            )
        if self.dual_sparsity_weight < 0 or self.dual_gate_weight < 0:
            raise ValueError(
                "dual loss weights must be >= 0, "
                f"got {self.dual_sparsity_weight}, {self.dual_gate_weight}"
            )

    def _validate_dataset_configuration(self) -> None:
        if self.training_mode != "Patch":
            return

        if self.patch_size > self.dataset.px_height or self.patch_size > self.dataset.px_width:
            raise ValueError(
                "patch_size must fit inside one training slice, "
                f"got patch_size={self.patch_size}, "
                f"slice={self.dataset.px_width}x{self.dataset.px_height}"
            )

        patch_area = self.patch_size ** 2
        if self.points_per_iter < patch_area:
            raise ValueError(
                "points_per_iter must be at least patch_size**2 in Patch mode, "
                f"got points_per_iter={self.points_per_iter}, patch_size={self.patch_size}"
            )

        if self.jitter_training:
            print("Patch mode disables per-pixel training jitter to preserve patch adjacency.")

    @staticmethod
    def _hash_per_level_scale(n_levels: int, base_resolution: float, finest_resolution: float) -> float:
        return (finest_resolution / base_resolution) ** (1.0 / (n_levels - 1))

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _print_hash_configuration(self) -> None:
        if self.nerf.get_encode_name() == "DUAL_HASH":
            encoder = self.nerf.dual_encoder.enc_high
        elif self.nerf.get_encode_name() == "HASH":
            encoder = self.nerf.encode
        else:
            return

        n_levels = int(encoder.n_levels)
        base_resolution = self._to_float(encoder.base_resolution)
        finest_resolution = self._to_float(encoder.finest_resolution)
        per_level_scale = self._hash_per_level_scale(
            n_levels,
            base_resolution,
            finest_resolution,
        )
        print(
            "HashGrid config: "
            f"L={n_levels}, "
            f"N_min={base_resolution:g}, "
            f"N_max={finest_resolution:g}, "
            f"per_level_scale={per_level_scale:.3f}, "
            f"features_per_level={encoder.n_features_per_level}, "
            f"log2_hashmap_size={encoder.log2_hashmap_size}"
        )

    def _build_scharr_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0).unsqueeze(0) / 32.0
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0).unsqueeze(0) / 32.0
        return scharr_x, scharr_y

    def _load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path:
            return None
        return torch.load(checkpoint_path, map_location=DEVICE)

    def _initialize_from_checkpoint(self, checkpoint: dict) -> None:
        print(f"Restarting from checkpoint: {self.ckptFile}")
        checkpoint_seed = int(checkpoint["seed"])
        np.random.seed(checkpoint_seed)
        torch.manual_seed(checkpoint_seed)
        self.seed = checkpoint_seed

        if checkpoint.get("baked", False):
            dataset_path = checkpoint.get(
                "baked_dataset_file",
                checkpoint.get("dataset_folder", self.datasetFolder),
            )
            self.dataset = Dataset.open_from_save(dataset_path)
        else:
            dataset_path = checkpoint.get(
                "dataset_folder",
                checkpoint.get("baked_dataset_file", self.datasetFolder),
            )
            self.dataset = Dataset(dataset_path)

        self.nerf = NeRF(checkpoint)
        self.optimizer = torch.optim.Adam(
            params=self.nerf.grad_vars(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start = int(checkpoint["start"])
        self.scheduler = self._build_lr_scheduler()

    def _initialize_from_scratch(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.dataset = self._load_dataset()
        self.nerf = NeRF()
        self._initialize_model_encoding()
        self.nerf.init_model(8, 256)
        self.optimizer = torch.optim.Adam(
            params=self.nerf.grad_vars(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = self._build_lr_scheduler()

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR:
        gamma = self.lr_decay_factor ** (1.0 / max(1, self.N_iters))
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=gamma,
            last_epoch=self.start - 1,
        )

    def _load_dataset(self) -> Dataset:
        if self.baked_dataset:
            return Dataset.open_from_save(self.datasetFolder)
        return Dataset(self.datasetFolder)

    def _initialize_model_encoding(self) -> None:
        encoding_name = str(self.encoding).lower()
        if encoding_name == "freq":
            self.nerf.init_base_encoding(
                use_directions=False,
                use_encoding=True,
                num_freq=16,
                num_freq_dir=4,
            )
            return

        if encoding_name == "hash":
            self.nerf.init_hash_encoding(
                bounding_box=self.dataset.get_bounding_box(),
                n_levels=self.hash_n_levels,
                n_features_per_level=self.hash_n_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution=self.hash_base_resolution,
                finest_resolution=self.hash_finest_resolution,
                use_encoding=True,
                use_directions=False,
            )
            return

        if encoding_name == "none":
            self.nerf.init_hash_encoding(
                bounding_box=self.dataset.get_bounding_box(),
                n_levels=self.hash_n_levels,
                n_features_per_level=self.hash_n_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution=self.hash_base_resolution,
                finest_resolution=self.hash_finest_resolution,
                use_encoding=False,
                use_directions=False,
            )
            return

        if encoding_name in ("dual_hash", "dual_freq"):
            pe_type = "hash" if encoding_name == "dual_hash" else "fourier"
            self.nerf.init_dual_encoding(
                pe_type=pe_type,
                bounding_box=self.dataset.get_bounding_box(),
                n_levels_low=self.dual_n_levels_low,
                n_levels_high=self.dual_n_levels_high,
                n_features_per_level=self.hash_n_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution_low=self.dual_base_resolution_low,
                finest_resolution_low=self.dual_finest_resolution_low,
                base_resolution_high=self.dual_base_resolution_high,
                finest_resolution_high=self.dual_finest_resolution_high,
                sigma_low=self.dual_sigma_low,
                sigma_high=self.dual_sigma_high,
                n_freq=self.dual_n_freq,
                use_gate=self.dual_use_gate,
                hf_activate_ratio=self.dual_hf_activate_ratio,
                hf_max_weight=self.dual_hf_max_weight,
            )
            return

        raise ValueError(f"Unknown encoding: {self.encoding}")

    def _create_run_paths(self) -> RunPaths:
        dated_log_dir = self.rootPoint / "logs" / datetime.date.today().strftime("%d-%m-%Y")
        dated_log_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{self.nerf.get_rep_name()}_{self.getDatasetName()}"
        run_index = 0
        log_dir = dated_log_dir / f"{base_name}_{run_index}"
        while log_dir.exists():
            run_index += 1
            log_dir = dated_log_dir / f"{base_name}_{run_index}"

        checkpoint_dir = log_dir / "checkpoints"
        image_dir = log_dir / "images"
        loss_dir = log_dir / "losses"
        latest_dir = self.rootPoint / "latest"

        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        image_dir.mkdir(parents=True, exist_ok=False)
        loss_dir.mkdir(parents=True, exist_ok=False)
        latest_dir.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            image_dir=image_dir,
            loss_dir=loss_dir,
            latest_checkpoint=latest_dir / "ckpt.pkl",
        )

    def _initialize_loss_csv(self) -> None:
        with self.loss_csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["iteration", "loss_train", "loss_valid", "loss_gt", "loss_valid_tv"])

    def _create_checkpoint_payload(self, iteration: int) -> dict:
        params = self.nerf.get_save_dict()
        params.update(
            {
                "seed": self.seed,
                "baked": self.baked_dataset,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "start": iteration,
                "bounding_box": self.dataset.get_bounding_box(),
            }
        )
        dataset_key = "baked_dataset_file" if self.baked_dataset else "dataset_folder"
        params[dataset_key] = self.datasetFolder
        return params

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint_path = self.run_paths.checkpoint_dir / f"{iteration}.pkl"
        torch.save(self._create_checkpoint_payload(iteration), checkpoint_path)
        shutil.copy2(checkpoint_path, self.run_paths.latest_checkpoint)

    def _reshape_valid_slice(self, getter, index: int) -> Optional[torch.Tensor]:
        if index >= len(self.dataset.slices_valid):
            return None

        slice_tensor = getter(index)
        return torch.reshape(slice_tensor, (self.dataset.px_height, self.dataset.px_width))

    def _iter_validation_indices(self):
        return range(min(len(self.dataset.slices_valid), len(VALIDATION_SLICE_NAMES)))

    def _render_validation_slices(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        rendered: dict[str, torch.Tensor] = {}
        references: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for index in self._iter_validation_indices():
                name = VALIDATION_SLICE_NAMES[index]
                rendered[name] = self.slice_renderer.render_slice_from_dataset_valid(
                    self.nerf,
                    index,
                    reshaped=True,
                ).detach()
                reference = self._reshape_valid_slice(self.dataset.get_slice_valid_pixels, index)
                if reference is not None:
                    references[name] = reference.detach()

        return rendered, references

    def _compute_validation_loss(
        self,
        rendered: dict[str, torch.Tensor],
        references: dict[str, torch.Tensor],
    ) -> tuple[Optional[float], Optional[float]]:
        if not rendered or not references:
            return None, None

        losses: list[float] = []
        tv_values: list[float] = []
        for name, prediction in rendered.items():
            if name in references:
                losses.append(self.criterion(prediction, references[name]).item())

            tv = (
                torch.mean(torch.abs(prediction[1:, :] - prediction[:-1, :]))
                + torch.mean(torch.abs(prediction[:, 1:] - prediction[:, :-1]))
            )
            tv_values.append(tv.item())

        loss_valid = float(np.mean(losses)) if losses else None
        tv_valid = float(np.mean(tv_values)) if tv_values else None
        return loss_valid, tv_valid

    def _compute_temporal_validation_loss(self, rendered: dict[str, torch.Tensor]) -> Optional[float]:
        if not self.previous_validation_slices:
            return None

        deltas = []
        for name, current_slice in rendered.items():
            previous_slice = self.previous_validation_slices.get(name)
            if previous_slice is None:
                continue
            deltas.append(torch.mean(torch.square(previous_slice - current_slice)).item())

        if not deltas:
            return None
        return float(np.mean(deltas))

    def _save_tensor_and_image(self, base_name: str, tensor: torch.Tensor) -> None:
        cpu_tensor = tensor.detach().cpu()
        torch.save(cpu_tensor, self.run_paths.image_dir / f"{base_name}.pt")
        plt.imsave(self.run_paths.image_dir / f"{base_name}.png", cpu_tensor.numpy(), cmap="gray")

    def _save_ground_truth_images(self, references: dict[str, torch.Tensor]) -> None:
        if self.gt_saved or not references:
            return

        for name, reference in references.items():
            self._save_tensor_and_image(f"{name}_gt", reference)

        self.gt_saved = True
        tqdm.tqdm.write(f"Ground truth images saved to {self.run_paths.image_dir}")

    def _save_preview_images(self, iteration: int, rendered: dict[str, torch.Tensor]) -> None:
        for name, tensor in rendered.items():
            self._save_tensor_and_image(f"{name}_{iteration}", tensor)

    def _prepare_tensorboard_image(self, tensor: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        image = tensor.detach().cpu().float()
        if normalize:
            image_min = torch.min(image)
            image_max = torch.max(image)
            if float(image_max - image_min) > 0:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = torch.zeros_like(image)
        elif float(torch.max(image)) > 1.0:
            image = image / 255.0

        return torch.clamp(image, 0.0, 1.0)

    def _write_tensorboard_images(
        self,
        iteration: int,
        rendered: dict[str, torch.Tensor],
        references: dict[str, torch.Tensor],
    ) -> None:
        for name, prediction in rendered.items():
            self.tb_writer.add_image(
                f"validation/{name}/prediction",
                self._prepare_tensorboard_image(prediction),
                iteration,
                dataformats="HW",
            )

            reference = references.get(name)
            if reference is not None and not self.tb_reference_images_logged:
                self.tb_writer.add_image(
                    f"validation/{name}/reference",
                    self._prepare_tensorboard_image(reference),
                    iteration,
                    dataformats="HW",
                )

            if reference is not None:
                abs_diff = torch.abs(prediction.detach().cpu() - reference.detach().cpu())
                self.tb_writer.add_image(
                    f"validation/{name}/abs_diff",
                    self._prepare_tensorboard_image(abs_diff, normalize=True),
                    iteration,
                    dataformats="HW",
                )

        self.tb_reference_images_logged = True

    def _write_metrics(
        self,
        iteration: int,
        loss_train: Optional[float],
        loss_valid: Optional[float],
        loss_gt: Optional[float],
        loss_valid_tv: Optional[float] = None,
    ) -> None:
        if loss_train is not None:
            self.tb_writer.add_scalar("loss/train", loss_train, iteration)
        if loss_valid is not None:
            self.tb_writer.add_scalar("loss/valid", loss_valid, iteration)
        if loss_gt is not None:
            self.tb_writer.add_scalar("loss/gt", loss_gt, iteration)
        if loss_valid_tv is not None:
            self.tb_writer.add_scalar("loss/valid_tv", loss_valid_tv, iteration)

        with self.loss_csv_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([iteration, loss_train, loss_valid, loss_gt, loss_valid_tv])

        self.tb_writer.flush()

    def _run_validation(
        self,
        iteration: int,
        train_losses: list[float],
        start_time: float,
        last_plot_time: float,
    ) -> tuple[Optional[dict], float]:
        now = time.time()
        secs_per_iter = (now - last_plot_time) / self.i_plot
        total_time = now - start_time
        elapsed = str(datetime.timedelta(seconds=int(total_time)))

        rendered, references = self._render_validation_slices()
        if not rendered:
            return None, now

        loss_train = float(np.mean(train_losses)) if train_losses else None
        loss_valid, loss_valid_tv = self._compute_validation_loss(rendered, references)
        loss_gt = self._compute_temporal_validation_loss(rendered)

        self._save_ground_truth_images(references)
        self._save_preview_images(iteration, rendered)
        if self.nerf.dual_encoder is not None:
            self._save_dual_freq_preview(iteration)
        self._write_tensorboard_images(iteration, rendered, references)
        self._write_metrics(iteration, loss_train, loss_valid, loss_gt, loss_valid_tv)
        self.previous_validation_slices = {
            name: tensor.detach().clone() for name, tensor in rendered.items()
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        params = {
            name: rendered.get(name) for name in VALIDATION_SLICE_NAMES
        }
        params.update(
            {
                "iteration": iteration,
                "time": elapsed,
                "loss_train": loss_train,
                "loss_valid": loss_valid,
                "loss_gt": loss_gt,
                "i_plot": self.i_plot,
                "secs_per_iter": secs_per_iter,
            }
        )
        return params, now

    def _model_input_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.slice_renderer._normalize_points_if_needed(
            self.nerf,
            torch.reshape(points, (-1, points.shape[-1])),
            self.dataset.point_min_dev,
        )

    def _save_dual_freq_preview(self, iteration: int) -> None:
        if not self.dataset.slices_valid or self.nerf.dual_encoder is None:
            return

        with torch.no_grad():
            progress = self.nerf.training_progress
            pts = self._model_input_points(self.dataset.get_slice_valid_points(0))
            decomp = self.nerf.dual_encoder.forward_decomposed(pts, progress)

        height, width = self.dataset.px_height, self.dataset.px_width

        def to_img(tensor: torch.Tensor) -> np.ndarray:
            arr = tensor.reshape(height, width).detach().cpu().float().numpy()
            arr_min = arr.min()
            arr_max = arr.max()
            return (arr - arr_min) / (arr_max - arr_min + 1e-8)

        low_img = to_img(decomp["feat_low"].mean(dim=-1))
        high_img = to_img(decomp["feat_high"].mean(dim=-1))
        gate_img = to_img(decomp["gate"].squeeze(-1))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(low_img, cmap="gray")
        axes[0].set_title("Low-frequency mean")
        axes[1].imshow(high_img, cmap="hot")
        axes[2].imshow(gate_img, cmap="hot")
        axes[2].set_title(f"Gate (progress={progress:.2f})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.run_paths.image_dir / f"dual_freq_{iteration:06d}.png", dpi=100)
        plt.close(fig)

    def _next_random_indices(self) -> torch.Tensor:
        if self.random_permutation is None:
            self.random_permutation = torch.randperm(
                len(self.dataset.points),
                device=self.dataset.points.device,
            )
            self.random_start_index = 0

        stop_index = self.random_start_index + self.points_per_iter
        if stop_index <= len(self.random_permutation):
            indices = self.random_permutation[self.random_start_index:stop_index]
        else:
            first_chunk = self.random_permutation[self.random_start_index:]
            overflow = stop_index - len(self.random_permutation)
            second_chunk = self.random_permutation[:overflow]
            indices = torch.cat((first_chunk, second_chunk), dim=0)

        self.random_start_index = stop_index % len(self.random_permutation)
        return indices

    def _build_training_slice_starts(self) -> torch.Tensor:
        return torch.as_tensor(
            [slice_info.start for slice_info in self.dataset.slices],
            dtype=torch.long,
            device=self.dataset.pixels.device,
        )

    def _patches_per_iter(self, patch_size: int) -> int:
        return self.points_per_iter // (patch_size ** 2)

    def _sample_training_batch(self):
        self._current_points = None
        self._current_viewdirs = None
        self._current_target_slice = None

        if self.training_mode == "Slice":
            slice_index = np.random.randint(len(self.dataset.slices))
            target = self.dataset.get_slice_pixels(slice_index).to(DEVICE)
            self._current_points = self.dataset.get_slice_points(slice_index)
            self._current_viewdirs = self.dataset.get_slice_viewdirs(slice_index)
            self._current_target_slice = target
            density = self.slice_renderer.render_slice_from_dataset(
                self.nerf, slice_index, jitter=self.jitter_training,
            )
            return target, density

        if self.training_mode == "Patch":
            return self._sample_patch_batch()

        if self.training_mode == "Random":
            indices = self._next_random_indices()
            target = self.dataset.get_indices_pixels(indices)
            self._current_points = self.dataset.get_indices_points(indices)
            self._current_viewdirs = self.dataset.get_indices_viewdirs(indices)
            density = self.slice_renderer.query_random_positions(self.nerf, indices, reshaped=False)
            return target, density

        raise ValueError(f"Training mode '{self.training_mode}' is not implemented")

    def _sample_patch_batch(self, patch_size: Optional[int] = None, n_patches: Optional[int] = None):
        patch_size = self.patch_size if patch_size is None else int(patch_size)
        n_patches = self._patches_per_iter(patch_size) if n_patches is None else int(n_patches)
        if n_patches <= 0:
            raise ValueError(
                "Patch mode requires at least one patch per iteration; "
                f"got points_per_iter={self.points_per_iter}, patch_size={patch_size}"
            )
        if self.training_slice_starts is None:
            self.training_slice_starts = self._build_training_slice_starts()

        device = self.dataset.pixels.device
        px_width = int(self.dataset.px_width)
        max_row = int(self.dataset.px_height) - patch_size + 1
        max_col = px_width - patch_size + 1
        if max_row <= 0 or max_col <= 0:
            raise ValueError(
                "patch_size must fit inside one training slice, "
                f"got patch_size={patch_size}, "
                f"slice={self.dataset.px_width}x{self.dataset.px_height}"
            )

        slice_indices = torch.randint(
            len(self.dataset.slices),
            (n_patches,),
            dtype=torch.long,
            device=device,
        )
        patch_rows = torch.randint(max_row, (n_patches,), dtype=torch.long, device=device)
        patch_cols = torch.randint(max_col, (n_patches,), dtype=torch.long, device=device)

        row_offsets = torch.arange(patch_size, dtype=torch.long, device=device).unsqueeze(1) * px_width
        col_offsets = torch.arange(patch_size, dtype=torch.long, device=device).unsqueeze(0)
        patch_offsets = (row_offsets + col_offsets).reshape(1, -1)
        patch_starts = (
            self.training_slice_starts[slice_indices]
            + patch_rows * px_width
            + patch_cols
        ).unsqueeze(1)
        indices = (patch_starts + patch_offsets).reshape(-1)

        target = self.dataset.get_indices_pixels(indices)
        self._current_points = self.dataset.get_indices_points(indices)
        self._current_viewdirs = self.dataset.get_indices_viewdirs(indices)
        density = self.slice_renderer.query_random_positions(
            self.nerf,
            indices,
            reshaped=False,
            jitter=False,
        )
        return target, density

    def _sample_slice_batch(self):
        slice_index = np.random.randint(len(self.dataset.slices))
        target = self.dataset.get_slice_pixels(slice_index).to(DEVICE)
        density = self.slice_renderer.render_slice_from_dataset(
            self.nerf, slice_index, jitter=self.jitter_training,
        )
        return target, density

    def _compute_gradient_loss(self, target: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        target_slice = torch.reshape(
            target,
            (self.dataset.px_height, self.dataset.px_width),
        ).unsqueeze(0).unsqueeze(0)
        density_slice = torch.reshape(
            density,
            (self.dataset.px_height, self.dataset.px_width),
        ).unsqueeze(0).unsqueeze(0)

        target_blurred = self._gaussian_blur(
            target_slice,
            kernel_size=self.grad_blur_kernel_size,
            sigma=self.grad_blur_sigma,
        )
        target_grad_x = torch.nn.functional.conv2d(target_blurred, self.scharr_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target_blurred, self.scharr_y, padding=1)
        density_grad_x = torch.nn.functional.conv2d(density_slice, self.scharr_x, padding=1)
        density_grad_y = torch.nn.functional.conv2d(density_slice, self.scharr_y, padding=1)

        return self.criterion(density_grad_x, target_grad_x) + self.criterion(density_grad_y, target_grad_y)

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int = 6, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
        coords = coords - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        kernel = kernel / kernel.sum()

        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        x_padded = torch.nn.functional.pad(
            x,
            (pad_before, pad_after, pad_before, pad_after),
            mode="replicate",
        )
        return torch.nn.functional.conv2d(x_padded, kernel)

    def _compute_tv_loss(self, density):
        d = torch.reshape(density, (self.dataset.px_height, self.dataset.px_width))
        diff_h = d[1:, :] - d[:-1, :]
        diff_w = d[:, 1:] - d[:, :-1]

        # Huber: 小于threshold用L2（强平滑），大于threshold用L1（保边）
        threshold = 5.0  # 根据你图像灰度范围调s
        tv_h = torch.nn.functional.huber_loss(diff_h, torch.zeros_like(diff_h),
                                            delta=threshold, reduction='mean')
        tv_w = torch.nn.functional.huber_loss(diff_w, torch.zeros_like(diff_w),
                                            delta=threshold, reduction='mean')
        return tv_h + tv_w

    def _compute_patch_tv_loss(self, density: torch.Tensor) -> torch.Tensor:
        patch_area = self.patch_size ** 2
        if density.numel() % patch_area != 0:
            raise ValueError(
                "Patch density count must be divisible by patch_size**2, "
                f"got density.numel()={density.numel()}, patch_size={self.patch_size}"
            )

        density_patches = torch.reshape(
            density,
            (-1, 1, self.patch_size, self.patch_size),
        )
        diff_h = density_patches[:, :, 1:, :] - density_patches[:, :, :-1, :]
        diff_w = density_patches[:, :, :, 1:] - density_patches[:, :, :, :-1]
        tv_h = torch.square(diff_h).sum()
        tv_w = torch.square(diff_w).sum()
        return (tv_h + tv_w) / (density_patches.shape[0] * patch_area)

    def _compute_spatial_smoothness(self, points, viewdirs):
        delta = self.smoothness_delta
        noise = delta * torch.randn_like(points)
        perturbed = points + noise
        d_orig = self.nerf.query(self._model_input_points(points), viewdirs)
        d_perturbed = self.nerf.query(self._model_input_points(perturbed), viewdirs)
        return torch.mean(torch.abs(d_orig - d_perturbed))

    def _compute_gate_boundary_loss(
        self,
        gate: torch.Tensor,
        target_slice: torch.Tensor,
    ) -> torch.Tensor:
        height, width = self.dataset.px_height, self.dataset.px_width
        target = target_slice.reshape(1, 1, height, width).float()

        grad_x = F.conv2d(target, self.scharr_x, padding=1)
        grad_y = F.conv2d(target, self.scharr_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        grad_max = grad_mag.max()
        if grad_max > 1e-8:
            grad_mag = grad_mag / grad_max

        target_gate = grad_mag.reshape(-1, 1)[:gate.shape[0]]
        return F.mse_loss(gate, target_gate.detach())

    def _compute_training_loss(self, target, density, iteration=0):
        mse_loss = self.reconstruction_criterion(density, target)
        loss = mse_loss
        components: dict[str, torch.Tensor] = {"mse": mse_loss}

        # if self.training_mode == "Patch":
        #     tv = self._compute_patch_tv_loss(density)
        #     components["patch_tv"] = tv
        #     loss = loss + self.tv_weight * tv

        # elif self.training_mode == "Slice":
        #     grad = self._compute_gradient_loss(target, density)
        #     tv = self._compute_tv_loss(density)
        #     components["gradient"] = grad
        #     components["tv"] = tv
        #     loss = loss + self.grad_weight * grad + self.tv_weight * tv

        # elif self.training_mode == "Random":
        #     smooth = self._compute_spatial_smoothness(
        #         self._current_points, self._current_viewdirs
        #     )
        #     components["smoothness"] = smooth
        #     loss = loss + self.tv_weight * smooth

        if self.nerf.dual_encoder is not None and self._current_points is not None:
            progress = self.nerf.training_progress
            decomp = self.nerf.dual_encoder.forward_decomposed(
                self._model_input_points(self._current_points),
                progress,
            )
            feat_high = decomp["feat_high"]
            gate = decomp["gate"]

            sparsity = torch.mean(torch.abs(feat_high))
            components["dual_sparsity"] = sparsity
            loss = loss + self.dual_sparsity_weight * sparsity

            if (
                self.training_mode == "Slice"
                and self._current_target_slice is not None
                and self._current_target_slice.numel() == gate.shape[0]
            ):
                gate_loss = self._compute_gate_boundary_loss(
                    gate,
                    self._current_target_slice,
                )
                gate_weight = self.dual_gate_weight * progress
                components["dual_gate"] = gate_loss
                loss = loss + gate_weight * gate_loss

        return loss, components

    def _update_progress_bar(self, progress_bar, params: Optional[dict]) -> None:
        if params is None:
            return

        progress_bar.set_postfix(
            secs_per_iter=f"{params['secs_per_iter']:.4f}",
            loss_train=f"{params['loss_train']:.6f}" if params["loss_train"] is not None else "None",
            loss_valid=f"{params['loss_valid']:.6f}" if params["loss_valid"] is not None else "None",
            loss_gt=f"{params['loss_gt']:.6f}" if params["loss_gt"] is not None else "None",
            elapsed=params["time"],
        )

    def run(self):
        start_time = time.time()
        last_plot_time = start_time
        train_losses: list[float] = []

        progress_bar = tqdm.trange(
            self.start,
            self.N_iters + 1,
            desc="Training",
            dynamic_ncols=True,
        )

        try:
            for iteration in progress_bar:
                if self.nerf.dual_encoder is not None:
                    self.nerf.training_progress = iteration / max(1, self.N_iters)

                if (
                    self.progressive_training
                    and self.nerf.encoding_type == "HASH"
                    and self.nerf.use_encoding
                ):
                    active_levels = min(
                        self.nerf.encode.n_levels,
                        self.progressive_start_levels
                        + iteration // self.progressive_step_interval,
                    )
                    self.nerf._active_levels = active_levels
                    if iteration % 1000 == 0:
                        gates = torch.sigmoid(
                            self.nerf.encode.level_weights
                        ).detach().cpu().numpy()
                        print(f"Active levels: {active_levels}, Gates: {np.round(gates, 3)}")

                if iteration != 0 and iteration % self.i_save == 0:
                    self._save_checkpoint(iteration)

                if iteration % self.i_plot == 0:
                    params, last_plot_time = self._run_validation(
                        iteration,
                        train_losses,
                        start_time,
                        last_plot_time,
                    )
                    self._update_progress_bar(progress_bar, params)
                    if self.nerf.encoding_type == "HASH" and self.nerf.use_encoding:
                        gates = torch.sigmoid(
                            self.nerf.encode.level_weights
                        ).detach().cpu().numpy()
                        for i, gate in enumerate(gates):
                            self.tb_writer.add_scalar(f"gates/level_{i}", gate, iteration)

                        if self.progressive_training:
                            self.tb_writer.add_scalar(
                                "gates/active_levels",
                                getattr(self.nerf, '_active_levels', len(gates)),
                                iteration,
                            )

                target, density = self._sample_training_batch()
                self.optimizer.zero_grad()
                loss, loss_components = self._compute_training_loss(target, density, iteration)
                loss.backward()
                if self.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.nerf.parameters(),
                        max_norm=self.grad_clip_norm,
                    )
                    self.tb_writer.add_scalar(
                        "train/grad_norm",
                        self._to_float(grad_norm),
                        iteration,
                    )
                self.optimizer.step()
                self.scheduler.step()

                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                self.tb_writer.add_scalar("loss/train_iter", loss_value, iteration)
                self.tb_writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    iteration,
                )
                for name, value in loss_components.items():
                    self.tb_writer.add_scalar(
                        f"loss/components/{name}",
                        float(value.detach().cpu()),
                        iteration,
                    )
        finally:
            self.tb_writer.close()
            progress_bar.close()
            tensorboard_dir = self.run_paths.log_dir / "tensorboard"
            event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
            print(f"TensorBoard log dir: {tensorboard_dir.resolve()}")
            if event_files:
                for event_file in event_files:
                    print(f"TensorBoard event file: {event_file.resolve()}")
            else:
                print("TensorBoard event file not found.")

    def getReferences(self):
        references = [
            self._reshape_valid_slice(self.dataset.get_slice_valid_pixels, index)
            for index in range(len(VALIDATION_SLICE_NAMES))
        ]
        return tuple(references)

    def getGT(self):
        if not self.dataset.has_gt:
            return (None, None, None, None)

        gts = [
            self._reshape_valid_slice(self.dataset.get_slice_valid_gt, index)
            for index in range(len(VALIDATION_SLICE_NAMES))
        ]
        return tuple(gts)

    def getEncodingName(self):
        return self.nerf.get_encode_name()

    def getDatasetName(self):
        return self.dataset.name


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NeUF model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Dataset folder or baked dataset path")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to resume from")
    parser.add_argument(
        "--encoding",
        default="Hash",
        choices=["Hash", "Freq", "None", "DUAL_HASH", "DUAL_FREQ"],
    )
    parser.add_argument("--training-mode", default="Random", choices=["Random", "Slice", "Patch"])
    parser.add_argument("--points-per-iter", type=int, default=50000)
    parser.add_argument("--patch-size", type=int, default=32, help="Patch side length in pixels for Patch mode")
    parser.add_argument("--nb-iters-max", type=int, default=10000)
    parser.add_argument("--plot-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19981708)
    parser.add_argument("--grad-weight", type=float, default=0.1)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. Use 0 to disable.",
    )
    parser.add_argument(
        "--grad-blur-kernel-size",
        type=int,
        default=6,
        help="Gaussian blur kernel size for GT gradient loss.",
    )
    parser.add_argument(
        "--grad-blur-sigma",
        type=float,
        default=1.5,
        help="Gaussian blur sigma for GT gradient loss.",
    )
    parser.add_argument("--root", default=".", help="Root folder for logs and latest checkpoint")
    parser.add_argument("--raw-dataset", action="store_true", help="Treat --dataset as an unbaked folder")
    parser.add_argument("--jitter-training", action="store_true", help="Enable point jitter during training")
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--slice-mix-interval", type=int, default=10,
                        help="In Random mode, mix a full slice every N iters for 2D TV loss. 0 to disable.")
    parser.add_argument("--smoothness-delta", type=float, default=0.1,
                        help="Perturbation distance in mm for spatial smoothness loss")

    hash_group = parser.add_argument_group(
        "HashGrid experiment parameters",
        "Experiment 1 changes --hash-n-max with --hash-n-levels 16. "
        "Experiment 2 keeps --hash-n-max 512 and changes --hash-n-levels.",
    )
    hash_group.add_argument(
        "--hash-n-levels",
        "--hash-levels",
        dest="hash_n_levels",
        type=int,
        default=DEFAULT_HASH_N_LEVELS,
        help="HashGrid level count L.",
    )
    hash_group.add_argument(
        "--hash-n-max",
        "--hash-finest-resolution",
        dest="hash_finest_resolution",
        type=int,
        default=DEFAULT_HASH_FINEST_RESOLUTION,
        help="HashGrid finest resolution N_max.",
    )
    hash_group.add_argument(
        "--hash-n-min",
        "--hash-base-resolution",
        dest="hash_base_resolution",
        type=int,
        default=DEFAULT_HASH_BASE_RESOLUTION,
        help="HashGrid base resolution N_min.",
    )
    hash_group.add_argument(
        "--hash-n-features-per-level",
        dest="hash_n_features_per_level",
        type=int,
        default=DEFAULT_HASH_N_FEATURES_PER_LEVEL,
        help="HashGrid feature count per level.",
    )
    hash_group.add_argument(
        "--hash-log2-hashmap-size",
        dest="hash_log2_hashmap_size",
        type=int,
        default=DEFAULT_HASH_LOG2_HASHMAP_SIZE,
        help="HashGrid log2 hashmap size.",
    )
    hash_group.add_argument(
        "--progressive-training",
        action="store_true",
        help="Enable progressive level activation for hash encoder",
    )
    hash_group.add_argument(
        "--progressive-start-levels",
        type=int,
        default=4,
        help="Number of hash levels active from the start",
    )
    hash_group.add_argument(
        "--progressive-step-interval",
        type=int,
        default=1000,
        help="Unlock one more hash level every N iterations",
    )
    dual_group = parser.add_argument_group("Dual-frequency PE parameters")
    dual_group.add_argument(
        "--dual-pe-type",
        dest="dual_pe_type",
        default="hash",
        choices=["hash", "fourier"],
        help="Dual PE implementation: hash or fourier",
    )
    dual_group.add_argument("--dual-n-levels-low", dest="dual_n_levels_low", type=int, default=8)
    dual_group.add_argument("--dual-n-levels-high", dest="dual_n_levels_high", type=int, default=8)
    dual_group.add_argument(
        "--dual-finest-resolution-low",
        dest="dual_finest_resolution_low",
        type=int,
        default=64,
    )
    dual_group.add_argument(
        "--dual-finest-resolution-high",
        dest="dual_finest_resolution_high",
        type=int,
        default=512,
    )
    dual_group.add_argument(
        "--dual-base-resolution-low",
        dest="dual_base_resolution_low",
        type=int,
        default=16,
    )
    dual_group.add_argument(
        "--dual-base-resolution-high",
        dest="dual_base_resolution_high",
        type=int,
        default=64,
    )
    dual_group.add_argument("--dual-sigma-low", dest="dual_sigma_low", type=float, default=1.0)
    dual_group.add_argument("--dual-sigma-high", dest="dual_sigma_high", type=float, default=20.0)
    dual_group.add_argument("--dual-n-freq", dest="dual_n_freq", type=int, default=64)
    dual_group.add_argument(
        "--dual-no-gate",
        dest="dual_use_gate",
        action="store_false",
        default=True,
        help="Disable spatial gate network",
    )
    dual_group.add_argument(
        "--dual-hf-activate-ratio",
        dest="dual_hf_activate_ratio",
        type=float,
        default=0.6,
        help="Training progress ratio where high frequencies start activating",
    )
    dual_group.add_argument(
        "--dual-hf-max-weight",
        dest="dual_hf_max_weight",
        type=float,
        default=1.0,
    )
    dual_group.add_argument(
        "--dual-sparsity-weight",
        dest="dual_sparsity_weight",
        type=float,
        default=0.01,
    )
    dual_group.add_argument(
        "--dual-gate-weight",
        dest="dual_gate_weight",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    neuf = NeUF(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        encoding=args.encoding,
        training_mode=args.training_mode,
        points_per_iter=args.points_per_iter,
        patch_size=args.patch_size,
        nb_iters_max=args.nb_iters_max,
        plot_freq=args.plot_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        grad_weight=args.grad_weight,
        grad_clip_norm=args.grad_clip_norm,
        grad_blur_kernel_size=args.grad_blur_kernel_size,
        grad_blur_sigma=args.grad_blur_sigma,
        root=args.root,
        baked_dataset=not args.raw_dataset,
        jitter_training=args.jitter_training,
        tv_weight=args.tv_weight,
        slice_mix_interval=args.slice_mix_interval,
        smoothness_delta=args.smoothness_delta,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_hashmap_size=args.hash_log2_hashmap_size,
        hash_base_resolution=args.hash_base_resolution,
        hash_finest_resolution=args.hash_finest_resolution,
        progressive_training=args.progressive_training,
        progressive_start_levels=args.progressive_start_levels,
        progressive_step_interval=args.progressive_step_interval,
        dual_pe_type=args.dual_pe_type,
        dual_n_levels_low=args.dual_n_levels_low,
        dual_n_levels_high=args.dual_n_levels_high,
        dual_finest_resolution_low=args.dual_finest_resolution_low,
        dual_finest_resolution_high=args.dual_finest_resolution_high,
        dual_base_resolution_low=args.dual_base_resolution_low,
        dual_base_resolution_high=args.dual_base_resolution_high,
        dual_sigma_low=args.dual_sigma_low,
        dual_sigma_high=args.dual_sigma_high,
        dual_n_freq=args.dual_n_freq,
        dual_use_gate=args.dual_use_gate,
        dual_hf_activate_ratio=args.dual_hf_activate_ratio,
        dual_hf_max_weight=args.dual_hf_max_weight,
        dual_sparsity_weight=args.dual_sparsity_weight,
        dual_gate_weight=args.dual_gate_weight,
    )
    neuf.run()


if __name__ == "__main__":
    main()
