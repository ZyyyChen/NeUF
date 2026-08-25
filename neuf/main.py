from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import subprocess
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

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF
from neuf.pose_refinement import PoseRefiner
from neuf.sagittal_supervision import SagittalSliceSupervisor
from neuf.slice_renderer import SliceRenderer
from neuf.slice_renderer_ultra_nerf import UltraNeRFSliceRenderer
from neuf.ultra_nerf_renderer import REFERENCE_ULTRA_NERF_COMMIT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEFAULT_DATASET_PATH = (
#     "/home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/"
#     "Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl"
# )

DEFAULT_DATASET_PATH = (
    "/home/zchen/Code/NeUF/data/simu_56/us/baked_dataset.pkl"
)

VALIDATION_SLICE_NAMES = ("A1", "B1", "C1", "D1")
DEFAULT_HASH_N_LEVELS = 16
DEFAULT_HASH_N_FEATURES_PER_LEVEL = 2
DEFAULT_HASH_LOG2_HASHMAP_SIZE = 19
DEFAULT_HASH_BASE_RESOLUTION = 16
DEFAULT_HASH_FINEST_RESOLUTION = 256
DEFAULT_KRONECKER_N_LEVELS_LATERAL = 8
DEFAULT_KRONECKER_N_LEVELS_AXIAL = 8
DEFAULT_KRONECKER_FINEST_LATERAL = 128
DEFAULT_KRONECKER_FINEST_AXIAL = 512
DEFAULT_KRONECKER_BASE_RESOLUTION = 16
DEFAULT_KRONECKER_COMBINE = "cat"


@dataclass(frozen=True)
class RunPaths:
    log_dir: Path
    checkpoint_dir: Path
    image_dir: Path
    parameter_map_dir: Path
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
        self.training_mode = kwargs.get("training_mode", "CurriculumRS")
        self.phase_switch_ratio = float(kwargs.get("phase_switch_ratio", 0.5))
        self.curriculum_random_ratio = float(kwargs.get("curriculum_random_ratio", 0.2))
        self.curriculum_patch_ratio = float(kwargs.get("curriculum_patch_ratio", 0.5))
        self.points_per_iter = int(kwargs.get("points_per_iter", 50000))
        self.jitter_training = bool(kwargs.get("jitter_training", False))
        self.patch_size = int(kwargs.get("patch_size", 32))
        self.grad_blur_kernel_size = int(kwargs.get("grad_blur_kernel_size", 6))
        self.grad_blur_sigma = float(kwargs.get("grad_blur_sigma", 1.5))
        self.tv_weight = float(kwargs.get("tv_weight", 0))
        self.ssim_weight = float(kwargs.get("ssim_weight", 0.1))
        self.ssim_window_size = int(kwargs.get("ssim_window_size", 11))
        self.slice_mix_interval = int(kwargs.get("slice_mix_interval", 10))
        self.smoothness_delta = float(kwargs.get("smoothness_delta", 0.3))
        self.use_lateral_perturbation = bool(
            kwargs.get("use_lateral_perturbation", False)
        )
        self.lr = float(kwargs.get("lr", 5e-4))
        self.lr_decay_factor = float(kwargs.get("lr_decay_factor", 0.1))
        self.encoding = kwargs.get("encoding", "None")
        self.renderer_name = str(kwargs.get("renderer", "point")).lower()
        self.output_mode = (
            "ultra_nerf" if self.renderer_name == "ultra_nerf" else "intensity"
        )
        self.ultra_psf_half_size = int(kwargs.get("ultra_psf_half_size", 3))
        self.ultra_psf_lateral_std = float(
            kwargs.get("ultra_psf_lateral_std", 2.0)
        )
        self.ultra_psf_axial_std = float(kwargs.get("ultra_psf_axial_std", 1.0))
        self.ultra_distance_unit = str(
            kwargs.get("ultra_distance_unit", "m")
        ).lower()
        self.ultra_bernoulli_seed = int(kwargs.get("ultra_bernoulli_seed", 0))
        self.ultra_eval_mc_samples = int(kwargs.get("ultra_eval_mc_samples", 1))
        self.ultra_query_chunk = int(kwargs.get("ultra_query_chunk", 65536))
        self.ultra_save_parameter_maps = bool(
            kwargs.get("ultra_save_parameter_maps", False)
        )
        self.ultra_init_attenuation = float(
            kwargs.get("ultra_init_attenuation", 1.0)
        )
        self.ultra_init_reflection = float(
            kwargs.get("ultra_init_reflection", 0.02)
        )
        self.ultra_init_border_probability = float(
            kwargs.get("ultra_init_border_probability", 0.005)
        )
        self.ultra_init_scatter_density = float(
            kwargs.get("ultra_init_scatter_density", 0.2)
        )
        self.ultra_init_scatter_amplitude = float(
            kwargs.get("ultra_init_scatter_amplitude", 0.5)
        )
        self.ultra_init_weight_std = float(
            kwargs.get("ultra_init_weight_std", 1e-4)
        )
        self.ultra_mse_warmup_iters = int(
            kwargs.get("ultra_mse_warmup_iters", 500)
        )
        self.ultra_loss_ramp_iters = int(
            kwargs.get("ultra_loss_ramp_iters", 1500)
        )
        self.ultra_final_ms_ssim_weight = float(
            kwargs.get("ultra_final_ms_ssim_weight", 0.9)
        )
        self.ultra_collapse_threshold = float(
            kwargs.get("ultra_collapse_threshold", 1e-6)
        )
        self.ultra_collapse_patience = int(
            kwargs.get("ultra_collapse_patience", 20)
        )
        self._ultra_collapse_count = 0
        intensity_activation_arg = kwargs.get("intensity_activation", None)
        self._intensity_activation_explicit = intensity_activation_arg is not None
        self.intensity_activation = (
            "sigmoid"
            if intensity_activation_arg is None
            else str(intensity_activation_arg).lower()
        )
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
        self.kronecker_n_levels_lateral = int(
            kwargs.get("kronecker_n_levels_lateral", DEFAULT_KRONECKER_N_LEVELS_LATERAL)
        )
        self.kronecker_n_levels_axial = int(
            kwargs.get("kronecker_n_levels_axial", DEFAULT_KRONECKER_N_LEVELS_AXIAL)
        )
        self.kronecker_finest_lateral = int(
            kwargs.get("kronecker_finest_lateral", DEFAULT_KRONECKER_FINEST_LATERAL)
        )
        self.kronecker_finest_axial = int(
            kwargs.get("kronecker_finest_axial", DEFAULT_KRONECKER_FINEST_AXIAL)
        )
        self.kronecker_n_features_per_level = int(
            kwargs.get("kronecker_n_features_per_level", self.hash_n_features_per_level)
        )
        self.kronecker_log2_hashmap_size = int(
            kwargs.get("kronecker_log2_hashmap_size", self.hash_log2_hashmap_size)
        )
        self.kronecker_base_resolution = int(
            kwargs.get("kronecker_base_resolution", DEFAULT_KRONECKER_BASE_RESOLUTION)
        )
        self.kronecker_combine = kwargs.get("kronecker_combine", DEFAULT_KRONECKER_COMBINE)
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
        self.dual_gate_weight = float(kwargs.get("dual_gate_weight", 0.1))
        self.progressive_training = bool(kwargs.get("progressive_training", False))
        self.progressive_start_levels = int(kwargs.get("progressive_start_levels", 4))
        self.progressive_step_interval = int(kwargs.get("progressive_step_interval", 1000))
        self.noise_sigma_min = float(kwargs.get("noise_sigma_min", 1e-3))
        self.noise_sigma_max = float(kwargs.get("noise_sigma_max", 1.0))
        use_loupas_arg = kwargs.get("use_loupas", None)
        self._use_loupas_explicit = use_loupas_arg is not None
        self.use_loupas = (
            self.renderer_name != "ultra_nerf"
            if use_loupas_arg is None
            else bool(use_loupas_arg)
        )
        self.loupas_gamma = float(kwargs.get("loupas_gamma", 0.5))
        self.loupas_weight = float(kwargs.get("loupas_weight", 0.1))
        optimize_poses_arg = kwargs.get("optimize_poses", None)
        self._optimize_poses_explicit = optimize_poses_arg is not None
        self.optimize_poses = False if optimize_poses_arg is None else bool(optimize_poses_arg)
        pose_anchor_arg = kwargs.get("pose_anchor_first", None)
        self._pose_anchor_explicit = pose_anchor_arg is not None
        self.pose_anchor_first = True if pose_anchor_arg is None else bool(pose_anchor_arg)
        pose_defaults = {
            "pose_lr": (1e-4, float),
            "pose_lr_end": (1e-5, float),
            "pose_warmup_iters": (500, int),
            "pose_start_iter": (0, int),
            "pose_rotation_reg_weight": (0.0, float),
            "pose_translation_reg_weight": (0.0, float),
            "pose_velocity_reg_weight": (0.0, float),
            "pose_acceleration_reg_weight": (0.0, float),
            "pose_grad_clip_norm": (1.0, float),
        }
        self._pose_hparam_explicit: dict[str, bool] = {}
        for name, (default, caster) in pose_defaults.items():
            configured_value = kwargs.get(name, None)
            self._pose_hparam_explicit[name] = configured_value is not None
            setattr(self, name, caster(default if configured_value is None else configured_value))

        sagittal_mat_arg = kwargs.get("sagittal_mat", None)
        self._sagittal_mat_explicit = sagittal_mat_arg is not None
        self.sagittal_mat = "" if sagittal_mat_arg is None else str(sagittal_mat_arg)
        sagittal_variable_arg = kwargs.get("sagittal_variable", None)
        self._sagittal_variable_explicit = sagittal_variable_arg is not None
        self.sagittal_variable = (
            "data_sag" if sagittal_variable_arg is None else str(sagittal_variable_arg)
        )
        optimize_sagittal_pose_arg = kwargs.get("optimize_sagittal_pose", None)
        self._optimize_sagittal_pose_explicit = optimize_sagittal_pose_arg is not None
        self.optimize_sagittal_pose = (
            True if optimize_sagittal_pose_arg is None else bool(optimize_sagittal_pose_arg)
        )
        sagittal_defaults = {
            "sagittal_weight": (1.0, float),
            "sagittal_points_per_iter": (8192, int),
            "sagittal_start_iter": (0, int),
            "sagittal_ramp_iters": (0, int),
            "sagittal_pose_lr": (1e-4, float),
            "sagittal_pose_lr_end": (1e-5, float),
            "sagittal_pose_warmup_iters": (500, int),
            "sagittal_pose_start_iter": (0, int),
            "sagittal_pose_rotation_reg_weight": (0.0, float),
            "sagittal_pose_translation_reg_weight": (0.0, float),
            "sagittal_pose_grad_clip_norm": (1.0, float),
        }
        self._sagittal_hparam_explicit: dict[str, bool] = {}
        for name, (default, caster) in sagittal_defaults.items():
            configured_value = kwargs.get(name, None)
            self._sagittal_hparam_explicit[name] = configured_value is not None
            setattr(self, name, caster(default if configured_value is None else configured_value))

        self._validate_configuration()

        self.dataset: Dataset
        self.nerf: NeRF
        self.optimizer: torch.optim.Optimizer
        self.pose_refiner: Optional[PoseRefiner] = None
        self.pose_optimizer: Optional[torch.optim.Optimizer] = None
        self.sagittal_supervisor: Optional[SagittalSliceSupervisor] = None
        self.sagittal_pose_optimizer: Optional[torch.optim.Optimizer] = None
        self.slice_renderer: SliceRenderer
        self.ultra_slice_renderer: Optional[UltraNeRFSliceRenderer] = None
        self.start = 0

        self.criterion = torch.nn.L1Loss()
        self.reconstruction_criterion = torch.nn.MSELoss()
        self.scharr_x, self.scharr_y = self._build_scharr_kernels()
        self.previous_validation_slices: dict[str, torch.Tensor] = {}
        self._validation_ultra_maps: dict[str, dict[str, torch.Tensor]] = {}
        self.gt_saved = False
        self.sagittal_target_saved = False
        self.tb_reference_images_logged = False
        self._active_mode: str = (
            "Random" if self.training_mode.startswith("Curriculum") else self.training_mode
        )

        self.random_permutation: Optional[torch.Tensor] = None
        self.random_start_index = 0
        self.training_point_count = 0
        self.training_slice_starts: Optional[torch.Tensor] = None
        self._current_points = None
        self._current_viewdirs = None
        self._current_target_slice = None
        self._current_valid_mask = None

        checkpoint = self._load_checkpoint(self.ckptFile)
        if checkpoint is not None:
            self._initialize_from_checkpoint(checkpoint)
        else:
            self._initialize_from_scratch()
        self._print_hash_configuration()

        self.slice_renderer = SliceRenderer(self.dataset)
        if self.renderer_name == "ultra_nerf":
            self.ultra_slice_renderer = UltraNeRFSliceRenderer(
                self.dataset,
                psf_half_size=self.ultra_psf_half_size,
                psf_lateral_std=self.ultra_psf_lateral_std,
                psf_axial_std=self.ultra_psf_axial_std,
                distance_unit=self.ultra_distance_unit,
                bernoulli_seed=self.ultra_bernoulli_seed,
                eval_mc_samples=self.ultra_eval_mc_samples,
                query_chunk=self.ultra_query_chunk,
            )
        self.training_slice_starts = self._build_training_slice_starts()
        self._validate_dataset_configuration()
        self.run_paths = self._create_run_paths()
        self.logPath = str(self.run_paths.log_dir)
        self._write_run_manifest()

        self.tb_writer = SummaryWriter(log_dir=str(self.run_paths.log_dir / "tensorboard"))
        self.loss_csv_path = self.run_paths.log_dir / "train_history.csv"
        self._initialize_loss_csv()

    def _validate_configuration(self) -> None:
        if self.renderer_name not in {"point", "ultra_nerf"}:
            raise ValueError(
                "renderer must be 'point' or 'ultra_nerf', got "
                f"{self.renderer_name}"
            )
        if self.N_iters < 0:
            raise ValueError(f"nb_iters_max must be >= 0, got {self.N_iters}")
        if self.i_plot <= 0:
            raise ValueError(f"plot_freq must be >= 1, got {self.i_plot}")
        if self.i_save <= 0:
            raise ValueError(f"save_freq must be >= 1, got {self.i_save}")
        if self.training_mode not in {
            "Random",
            "Slice",
            "Patch",
            "CurriculumRS",
            "CurriculumRPS",
        }:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        if self.renderer_name == "ultra_nerf" and self.training_mode != "Slice":
            raise ValueError(
                "The strict Ultra-NeRF renderer requires --training-mode Slice so "
                "every update contains complete A-lines and the full PSF neighbourhood"
            )
        if self.renderer_name == "ultra_nerf" and self.jitter_training:
            raise ValueError(
                "The strict Ultra-NeRF renderer does not support per-pixel jitter"
            )
        if self.renderer_name == "ultra_nerf" and self.use_loupas:
            raise ValueError(
                "The strict Ultra-NeRF renderer has no log_sigma output; use --no-loupas"
            )
        if self.ultra_psf_half_size < 0:
            raise ValueError("ultra_psf_half_size must be >= 0")
        if self.ultra_psf_lateral_std <= 0 or self.ultra_psf_axial_std <= 0:
            raise ValueError("Ultra-NeRF PSF standard deviations must be positive")
        if self.ultra_distance_unit not in {"m", "mm"}:
            raise ValueError("ultra_distance_unit must be 'm' or 'mm'")
        if self.ultra_eval_mc_samples < 1:
            raise ValueError("ultra_eval_mc_samples must be >= 1")
        if self.ultra_query_chunk < 1:
            raise ValueError("ultra_query_chunk must be >= 1")
        if self.ultra_init_attenuation <= 0:
            raise ValueError("ultra_init_attenuation must be > 0")
        initial_probabilities = {
            "ultra_init_reflection": self.ultra_init_reflection,
            "ultra_init_border_probability": self.ultra_init_border_probability,
            "ultra_init_scatter_density": self.ultra_init_scatter_density,
            "ultra_init_scatter_amplitude": self.ultra_init_scatter_amplitude,
        }
        invalid_probabilities = {
            name: value
            for name, value in initial_probabilities.items()
            if not 0 < value < 1
        }
        if invalid_probabilities:
            raise ValueError(
                "Ultra-NeRF initial probabilities must be in (0, 1): "
                f"{invalid_probabilities}"
            )
        if self.ultra_init_weight_std < 0:
            raise ValueError("ultra_init_weight_std must be >= 0")
        if self.ultra_mse_warmup_iters < 0 or self.ultra_loss_ramp_iters < 0:
            raise ValueError("Ultra-NeRF loss warm-up and ramp iterations must be >= 0")
        if not 0 <= self.ultra_final_ms_ssim_weight <= 1:
            raise ValueError("ultra_final_ms_ssim_weight must be in [0, 1]")
        if self.ultra_collapse_threshold < 0:
            raise ValueError("ultra_collapse_threshold must be >= 0")
        if self.ultra_collapse_patience < 1:
            raise ValueError("ultra_collapse_patience must be >= 1")
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
        if self.grad_weight < 0:
            raise ValueError(f"grad_weight must be >= 0, got {self.grad_weight}")
        if self.tv_weight < 0:
            raise ValueError(f"tv_weight must be >= 0, got {self.tv_weight}")
        if self.ssim_weight < 0:
            raise ValueError(f"ssim_weight must be >= 0, got {self.ssim_weight}")
        if self.ssim_window_size < 3 or self.ssim_window_size % 2 == 0:
            raise ValueError(
                "ssim_window_size must be an odd integer >= 3, "
                f"got {self.ssim_window_size}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.intensity_activation not in {"identity", "sigmoid"}:
            raise ValueError(
                "intensity_activation must be 'identity' or 'sigmoid', got "
                f"{self.intensity_activation}"
            )
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
        if self.kronecker_n_levels_lateral <= 0 or self.kronecker_n_levels_axial <= 0:
            raise ValueError(
                "kronecker_n_levels_lateral and kronecker_n_levels_axial must be >= 1, "
                f"got {self.kronecker_n_levels_lateral}, {self.kronecker_n_levels_axial}"
            )
        if self.kronecker_n_features_per_level <= 0:
            raise ValueError(
                "kronecker_n_features_per_level must be >= 1, "
                f"got {self.kronecker_n_features_per_level}"
            )
        if self.kronecker_log2_hashmap_size <= 0:
            raise ValueError(
                "kronecker_log2_hashmap_size must be >= 1, "
                f"got {self.kronecker_log2_hashmap_size}"
            )
        if self.kronecker_base_resolution <= 0:
            raise ValueError(
                "kronecker_base_resolution must be >= 1, "
                f"got {self.kronecker_base_resolution}"
            )
        if self.kronecker_finest_lateral < self.kronecker_base_resolution:
            raise ValueError(
                "kronecker_finest_lateral must be >= kronecker_base_resolution, "
                f"got {self.kronecker_finest_lateral} < {self.kronecker_base_resolution}"
            )
        if self.kronecker_finest_axial < self.kronecker_base_resolution:
            raise ValueError(
                "kronecker_finest_axial must be >= kronecker_base_resolution, "
                f"got {self.kronecker_finest_axial} < {self.kronecker_base_resolution}"
            )
        if self.kronecker_combine not in {"cat", "sum", "product"}:
            raise ValueError(
                "kronecker_combine must be one of: cat, sum, product; "
                f"got {self.kronecker_combine}"
            )
        if (
            self.kronecker_combine in {"sum", "product"}
            and self.kronecker_n_levels_lateral != self.kronecker_n_levels_axial
        ):
            raise ValueError(
                "kronecker sum/product require equal lateral and axial dimensions; "
                f"got n_levels_lateral={self.kronecker_n_levels_lateral}, "
                f"n_levels_axial={self.kronecker_n_levels_axial}"
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
        if not (0.0 < self.phase_switch_ratio < 1.0):
            raise ValueError(
                f"phase_switch_ratio must be in (0, 1), got {self.phase_switch_ratio}"
            )
        if not (0.0 < self.curriculum_random_ratio < 1.0):
            raise ValueError(
                "curriculum_random_ratio must be in (0, 1), got "
                f"{self.curriculum_random_ratio}"
            )
        if not (0.0 < self.curriculum_patch_ratio < 1.0):
            raise ValueError(
                "curriculum_patch_ratio must be in (0, 1), got "
                f"{self.curriculum_patch_ratio}"
            )
        if self.curriculum_random_ratio + self.curriculum_patch_ratio >= 1.0:
            raise ValueError(
                "curriculum_random_ratio + curriculum_patch_ratio must be < 1, got "
                f"{self.curriculum_random_ratio + self.curriculum_patch_ratio}"
            )
        if self.noise_sigma_min <= 0:
            raise ValueError(f"noise_sigma_min must be > 0, got {self.noise_sigma_min}")
        if self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError(
                "noise_sigma_max must be greater than noise_sigma_min, "
                f"got {self.noise_sigma_max} <= {self.noise_sigma_min}"
            )
        if self.loupas_weight < 0:
            raise ValueError(f"loupas_weight must be >= 0, got {self.loupas_weight}")
        if self.pose_lr <= 0 or self.pose_lr_end <= 0:
            raise ValueError(
                f"pose_lr and pose_lr_end must be > 0, got {self.pose_lr}, {self.pose_lr_end}"
            )
        if self.pose_warmup_iters < 0:
            raise ValueError(
                f"pose_warmup_iters must be >= 0, got {self.pose_warmup_iters}"
            )
        if self.pose_start_iter < 0:
            raise ValueError(f"pose_start_iter must be >= 0, got {self.pose_start_iter}")
        if self.pose_rotation_reg_weight < 0 or self.pose_translation_reg_weight < 0:
            raise ValueError(
                "pose regularization weights must be >= 0, got "
                f"{self.pose_rotation_reg_weight}, {self.pose_translation_reg_weight}"
            )
        if self.pose_velocity_reg_weight < 0 or self.pose_acceleration_reg_weight < 0:
            raise ValueError(
                "pose trajectory regularization weights must be >= 0, got "
                f"{self.pose_velocity_reg_weight}, {self.pose_acceleration_reg_weight}"
            )
        if self.pose_grad_clip_norm < 0:
            raise ValueError(
                f"pose_grad_clip_norm must be >= 0, got {self.pose_grad_clip_norm}"
            )
        if self.sagittal_mat and not self.sagittal_variable:
            raise ValueError("sagittal_variable cannot be empty when sagittal supervision is enabled")
        if self.sagittal_weight < 0:
            raise ValueError(
                f"sagittal_weight must be >= 0, got {self.sagittal_weight}"
            )
        if self.sagittal_points_per_iter <= 0:
            raise ValueError(
                "sagittal_points_per_iter must be >= 1, got "
                f"{self.sagittal_points_per_iter}"
            )
        if self.sagittal_start_iter < 0 or self.sagittal_ramp_iters < 0:
            raise ValueError(
                "sagittal_start_iter and sagittal_ramp_iters must be >= 0, got "
                f"{self.sagittal_start_iter}, {self.sagittal_ramp_iters}"
            )
        if self.sagittal_pose_lr <= 0 or self.sagittal_pose_lr_end <= 0:
            raise ValueError(
                "sagittal pose learning rates must be > 0, got "
                f"{self.sagittal_pose_lr}, {self.sagittal_pose_lr_end}"
            )
        if self.sagittal_pose_warmup_iters < 0:
            raise ValueError(
                "sagittal_pose_warmup_iters must be >= 0, got "
                f"{self.sagittal_pose_warmup_iters}"
            )
        if self.sagittal_pose_start_iter < 0:
            raise ValueError(
                "sagittal_pose_start_iter must be >= 0, got "
                f"{self.sagittal_pose_start_iter}"
            )
        if (
            self.sagittal_pose_rotation_reg_weight < 0
            or self.sagittal_pose_translation_reg_weight < 0
        ):
            raise ValueError(
                "sagittal pose regularization weights must be >= 0, got "
                f"{self.sagittal_pose_rotation_reg_weight}, "
                f"{self.sagittal_pose_translation_reg_weight}"
            )
        if self.sagittal_pose_grad_clip_norm < 0:
            raise ValueError(
                "sagittal_pose_grad_clip_norm must be >= 0, got "
                f"{self.sagittal_pose_grad_clip_norm}"
            )

    def _current_phase(self, progress: float) -> str:
        if self.training_mode == "CurriculumRPS":
            if progress < self.curriculum_random_ratio:
                return "Random"
            if progress < self.curriculum_random_ratio + self.curriculum_patch_ratio:
                return "Patch"
            return "Slice"
        if self.training_mode != "CurriculumRS":
            return self.training_mode
        return "Random" if progress < self.phase_switch_ratio else "Slice"

    def _validate_dataset_configuration(self) -> None:
        if not self.dataset.slices:
            raise ValueError("Dataset contains no training slices")
        if self.dataset.valid_pixel_count <= 0:
            raise ValueError("Dataset mandatory ultrasound sector mask is empty")

        expected_start = 0
        for slice_index, slice_info in enumerate(self.dataset.slices):
            if int(slice_info.start) != expected_start:
                raise ValueError(
                    "Training slices must be contiguous and ordered for random sampling; "
                    f"slice {slice_index} starts at {slice_info.start}, expected {expected_start}"
                )
            if int(slice_info.end) <= int(slice_info.start):
                raise ValueError(
                    f"Training slice {slice_index} has invalid bounds: "
                    f"[{slice_info.start}, {slice_info.end})"
                )
            expected_start = int(slice_info.end)

        tensor_lengths = {
            "pixels": len(self.dataset.pixels),
            "points": len(self.dataset.points),
            "viewdirs": len(self.dataset.viewdirs),
        }
        too_short = {
            name: length for name, length in tensor_lengths.items() if length < expected_start
        }
        if too_short:
            raise ValueError(
                "Dataset tensors do not cover all declared training slices: "
                f"required={expected_start}, lengths={too_short}"
            )

        trailing_counts = {
            name: length - expected_start
            for name, length in tensor_lengths.items()
            if length > expected_start
        }
        if trailing_counts and self.optimize_poses:
            self.training_point_count = (
                self.dataset.valid_pixel_count * len(self.dataset.slices)
            )
            print(
                "Warning: legacy baked dataset contains trailing samples without "
                "matching slice poses; excluding them from training: "
                f"{trailing_counts}"
            )
        else:
            self.training_point_count = (
                self.dataset.valid_pixel_count * len(self.dataset.slices)
            )

        if self.training_mode not in {"Patch", "CurriculumRPS"}:
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
                "points_per_iter must be at least patch_size**2 when Patch is used, "
                f"got points_per_iter={self.points_per_iter}, patch_size={self.patch_size}"
            )

        if self.jitter_training:
            print("Patch mode disables per-pixel training jitter to preserve patch adjacency.")

    @staticmethod
    def _hash_per_level_scale(n_levels: int, base_resolution: float, finest_resolution: float) -> float:
        if n_levels <= 1:
            return 1.0
        return (finest_resolution / base_resolution) ** (1.0 / (n_levels - 1))

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean()
        weights = mask.to(dtype=values.dtype, device=values.device)
        if weights.shape != values.shape:
            weights = torch.broadcast_to(weights, values.shape)
        denominator = weights.sum()
        if denominator <= 0:
            raise ValueError("Mandatory ultrasound sector mask selected zero loss values")
        return torch.sum(values * weights) / denominator

    def _print_hash_configuration(self) -> None:
        if self.nerf.get_encode_name() == "DUAL_HASH":
            encoder = self.nerf.dual_encoder.enc_high
        elif self.nerf.get_encode_name() == "HASH":
            encoder = self.nerf.encode
        elif self.nerf.get_encode_name() == "KRONECKER":
            encoder = self.nerf.kronecker_encoder
            xy = encoder.enc_xy
            xz = encoder.enc_xz
            lateral_scale = self._hash_per_level_scale(
                int(xy.n_levels),
                self._to_float(xy.base_resolution),
                self._to_float(xy.finest_resolution),
            )
            axial_scale = self._hash_per_level_scale(
                int(xz.n_levels),
                self._to_float(xz.base_resolution),
                self._to_float(xz.finest_resolution),
            )
            print(
                "KroneckerHash config: "
                f"combine={encoder.combine}, "
                f"L_lateral={xy.n_levels}, "
                f"L_axial={xz.n_levels}, "
                f"N_min={self._to_float(xy.base_resolution):g}, "
                f"N_lateral={self._to_float(xy.finest_resolution):g}, "
                f"N_axial={self._to_float(xz.finest_resolution):g}, "
                f"scale_lateral={lateral_scale:.3f}, "
                f"scale_axial={axial_scale:.3f}, "
                f"features_per_level={xy.n_features_per_level}, "
                f"log2_hashmap_size={xy.log2_hashmap_size}"
            )
            return
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
        checkpoint_renderer = str(checkpoint.get("renderer", "point")).lower()
        if checkpoint_renderer != self.renderer_name:
            raise ValueError(
                "Renderer does not match checkpoint: "
                f"requested={self.renderer_name}, checkpoint={checkpoint_renderer}"
            )
        checkpoint_output_mode = str(
            checkpoint.get("output_mode", "intensity")
        ).lower()
        if checkpoint_output_mode != self.output_mode:
            raise ValueError(
                "Network output mode does not match checkpoint: "
                f"requested={self.output_mode}, checkpoint={checkpoint_output_mode}"
            )
        if self.renderer_name == "ultra_nerf":
            ultra_checkpoint_values = {
                "ultra_psf_half_size": self.ultra_psf_half_size,
                "ultra_psf_lateral_std": self.ultra_psf_lateral_std,
                "ultra_psf_axial_std": self.ultra_psf_axial_std,
                "ultra_distance_unit": self.ultra_distance_unit,
                "ultra_bernoulli_seed": self.ultra_bernoulli_seed,
            }
            mismatches = {
                name: (requested, checkpoint.get(name))
                for name, requested in ultra_checkpoint_values.items()
                if name not in checkpoint or checkpoint[name] != requested
            }
            if mismatches:
                raise ValueError(
                    "Ultra-NeRF renderer configuration does not match checkpoint: "
                    f"{mismatches}"
                )
        checkpoint_seed = int(checkpoint["seed"])
        np.random.seed(checkpoint_seed)
        torch.manual_seed(checkpoint_seed)
        self.seed = checkpoint_seed
        if not self._use_loupas_explicit:
            self.use_loupas = bool(checkpoint.get("use_loupas", self.use_loupas))
        if not self._intensity_activation_explicit:
            # Checkpoints created before this option used an unconstrained linear output.
            self.intensity_activation = str(
                checkpoint.get("intensity_activation", "identity")
            ).lower()
        if not self._optimize_poses_explicit:
            self.optimize_poses = bool(checkpoint.get("optimize_poses", self.optimize_poses))
        if not self._pose_anchor_explicit:
            self.pose_anchor_first = bool(
                checkpoint.get("pose_anchor_first", self.pose_anchor_first)
            )
        for name, was_explicit in self._pose_hparam_explicit.items():
            if was_explicit or name not in checkpoint:
                continue
            current_value = getattr(self, name)
            setattr(self, name, type(current_value)(checkpoint[name]))
        if not self._sagittal_mat_explicit:
            self.sagittal_mat = str(checkpoint.get("sagittal_mat", self.sagittal_mat))
        if not self._sagittal_variable_explicit:
            self.sagittal_variable = str(
                checkpoint.get("sagittal_variable", self.sagittal_variable)
            )
        if not self._optimize_sagittal_pose_explicit:
            self.optimize_sagittal_pose = bool(
                checkpoint.get("optimize_sagittal_pose", self.optimize_sagittal_pose)
            )
        for name, was_explicit in self._sagittal_hparam_explicit.items():
            if was_explicit or name not in checkpoint:
                continue
            current_value = getattr(self, name)
            setattr(self, name, type(current_value)(checkpoint[name]))
        self.noise_sigma_min = float(checkpoint.get("noise_sigma_min", self.noise_sigma_min))
        self.noise_sigma_max = float(checkpoint.get("noise_sigma_max", self.noise_sigma_max))
        self.loupas_gamma = float(checkpoint.get("loupas_gamma", self.loupas_gamma))
        self.loupas_weight = float(checkpoint.get("loupas_weight", self.loupas_weight))
        self._validate_configuration()

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

        validate_checkpoint_dataset_geometry(
            checkpoint,
            self.dataset,
            checkpoint_path=self.ckptFile,
        )

        self.nerf = NeRF(
            checkpoint,
            intensity_activation=self.intensity_activation,
            output_mode=self.output_mode,
        )
        self._validate_checkpoint_image_shape(checkpoint)
        if self.renderer_name == "ultra_nerf" and self.nerf.use_direction:
            raise ValueError("Strict Ultra-NeRF checkpoints must have use_directions=false")
        self.optimizer = torch.optim.Adam(
            params=self.nerf.grad_vars(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start = int(checkpoint["start"])
        self.scheduler = self._build_lr_scheduler()
        self._initialize_pose_optimization(checkpoint)
        self._initialize_sagittal_supervision(checkpoint)

    def _initialize_from_scratch(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.dataset = self._load_dataset()
        self.nerf = NeRF(
            intensity_activation=self.intensity_activation,
            output_mode=self.output_mode,
        )
        self._initialize_model_encoding()
        self.nerf.init_model(8, 256)
        if self.renderer_name == "ultra_nerf":
            self.nerf.initialize_ultra_output_head(
                attenuation=self.ultra_init_attenuation,
                reflection=self.ultra_init_reflection,
                border_probability=self.ultra_init_border_probability,
                scatter_density=self.ultra_init_scatter_density,
                scatter_amplitude=self.ultra_init_scatter_amplitude,
                weight_std=self.ultra_init_weight_std,
            )
        self.optimizer = torch.optim.Adam(
            params=self.nerf.grad_vars(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = self._build_lr_scheduler()
        self._initialize_pose_optimization()
        self._initialize_sagittal_supervision()

    def _validate_checkpoint_image_shape(self, checkpoint: dict) -> None:
        checkpoint_shape = checkpoint.get("image_shape_hw")
        if checkpoint_shape is None:
            return
        current_shape = [int(self.dataset.px_height), int(self.dataset.px_width)]
        if list(checkpoint_shape) != current_shape:
            raise ValueError(
                "Dataset image shape does not match checkpoint: "
                f"current={current_shape}, checkpoint={list(checkpoint_shape)}"
            )

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR:
        gamma = self.lr_decay_factor ** (1.0 / max(1, self.N_iters))
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=gamma,
            last_epoch=self.start - 1,
        )

    def _initialize_pose_optimization(self, checkpoint: Optional[dict] = None) -> None:
        self.pose_refiner = None
        self.pose_optimizer = None
        if not self.optimize_poses:
            return

        self.pose_refiner = PoseRefiner.from_slices(
            self.dataset.slices,
            device=DEVICE,
            anchor_first=self.pose_anchor_first,
        )
        pose_state = None if checkpoint is None else checkpoint.get("pose_refiner_state_dict")
        if pose_state is not None:
            try:
                self.pose_refiner.load_state_dict(pose_state)
            except RuntimeError as error:
                raise ValueError(
                    "Pose checkpoint is incompatible with the current training-slice set"
                ) from error

        self.pose_optimizer = torch.optim.Adam(
            self.pose_refiner.parameters(),
            lr=self.pose_lr,
            betas=(0.9, 0.999),
        )
        pose_optimizer_state = (
            None if checkpoint is None else checkpoint.get("pose_optimizer_state_dict")
        )
        if pose_optimizer_state is not None:
            self.pose_optimizer.load_state_dict(pose_optimizer_state)

        print(
            "BARF pose optimization enabled: "
            f"poses={self.pose_refiner.num_poses}, lr={self.pose_lr:g} -> "
            f"{self.pose_lr_end:g}, anchor_first={self.pose_anchor_first}, "
            f"trajectory_weights=(velocity={self.pose_velocity_reg_weight:g}, "
            f"acceleration={self.pose_acceleration_reg_weight:g})"
        )

    def _set_pose_learning_rate(self, iteration: int) -> float:
        if self.pose_optimizer is None:
            return 0.0

        if iteration < self.pose_start_iter:
            learning_rate = 0.0
            for parameter_group in self.pose_optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            return learning_rate

        pose_iteration = iteration - self.pose_start_iter
        pose_duration = max(1, self.N_iters - self.pose_start_iter)
        decay_progress = min(max(pose_iteration / pose_duration, 0.0), 1.0)
        learning_rate = self.pose_lr * (
            (self.pose_lr_end / self.pose_lr) ** decay_progress
        )
        if self.pose_warmup_iters > 0:
            learning_rate *= min(1.0, (pose_iteration + 1) / self.pose_warmup_iters)

        for parameter_group in self.pose_optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return learning_rate

    def _initialize_sagittal_supervision(
        self,
        checkpoint: Optional[dict] = None,
    ) -> None:
        self.sagittal_supervisor = None
        self.sagittal_pose_optimizer = None
        if not self.sagittal_mat:
            return

        self.sagittal_supervisor = SagittalSliceSupervisor.from_dataset(
            self.dataset,
            self.sagittal_mat,
            variable_name=self.sagittal_variable,
            optimize_pose=self.optimize_sagittal_pose,
            device=DEVICE,
        )
        checkpoint_mask = (
            None
            if checkpoint is None
            else checkpoint.get("sagittal_ultrasound_sector_mask")
        )
        if checkpoint_mask is not None:
            current_mask = self.sagittal_supervisor.valid_mask_signature()
            if (
                list(checkpoint_mask.get("shape_hw", [])) != current_mask["shape_hw"]
                or checkpoint_mask.get("sha256") != current_mask["sha256"]
            ):
                raise ValueError(
                    "Sagittal ultrasound sector mask does not match the checkpoint"
                )
        pose_state = (
            None if checkpoint is None else checkpoint.get("sagittal_pose_refiner_state_dict")
        )
        if pose_state is not None:
            try:
                self.sagittal_supervisor.pose_refiner.load_state_dict(pose_state)
            except RuntimeError as error:
                raise ValueError(
                    "Sagittal pose checkpoint is incompatible with the current slice geometry"
                ) from error

        if self.optimize_sagittal_pose:
            self.sagittal_pose_optimizer = torch.optim.Adam(
                self.sagittal_supervisor.pose_parameters(),
                lr=self.sagittal_pose_lr,
                betas=(0.9, 0.999),
            )
            optimizer_state = (
                None
                if checkpoint is None
                else checkpoint.get("sagittal_pose_optimizer_state_dict")
            )
            if optimizer_state is not None:
                self.sagittal_pose_optimizer.load_state_dict(optimizer_state)

        position = self.dataset.slices[
            self.sagittal_supervisor.initial_slice_index
        ].position
        print(
            "Sagittal supervision enabled: "
            f"source={self.sagittal_mat}:{self.sagittal_variable}, "
            f"shape={self.sagittal_supervisor.width}x{self.sagittal_supervisor.height}, "
            f"initial_training_slice={self.sagittal_supervisor.initial_slice_index}, "
            f"initial_position_mm=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
            f"weight={self.sagittal_weight:g}, start={self.sagittal_start_iter}, "
            f"ramp={self.sagittal_ramp_iters}, points/iter={self.sagittal_points_per_iter}, "
            f"optimize_pose={self.optimize_sagittal_pose}, "
            f"valid_sector={self.sagittal_supervisor.valid_point_count}/"
            f"{self.sagittal_supervisor.point_count}"
        )

    def _set_sagittal_pose_learning_rate(self, iteration: int) -> float:
        if self.sagittal_pose_optimizer is None:
            return 0.0

        if iteration < self.sagittal_pose_start_iter:
            learning_rate = 0.0
        else:
            pose_iteration = iteration - self.sagittal_pose_start_iter
            pose_duration = max(1, self.N_iters - self.sagittal_pose_start_iter)
            decay_progress = min(max(pose_iteration / pose_duration, 0.0), 1.0)
            learning_rate = self.sagittal_pose_lr * (
                (self.sagittal_pose_lr_end / self.sagittal_pose_lr) ** decay_progress
            )
            if self.sagittal_pose_warmup_iters > 0:
                learning_rate *= min(
                    1.0,
                    (pose_iteration + 1) / self.sagittal_pose_warmup_iters,
                )

        for parameter_group in self.sagittal_pose_optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return learning_rate

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

        if encoding_name == "kronecker":
            self.nerf.init_kronecker_encoding(
                bounding_box=self.dataset.get_bounding_box(),
                n_levels_lateral=self.kronecker_n_levels_lateral,
                n_levels_axial=self.kronecker_n_levels_axial,
                finest_lateral=self.kronecker_finest_lateral,
                finest_axial=self.kronecker_finest_axial,
                n_features_per_level=self.kronecker_n_features_per_level,
                log2_hashmap_size=self.kronecker_log2_hashmap_size,
                base_resolution=self.kronecker_base_resolution,
                combine=self.kronecker_combine,
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
        image_dir = log_dir / "predictions"
        parameter_map_dir = log_dir / "parameter_maps"
        loss_dir = log_dir / "losses"
        latest_dir = self.rootPoint / "latest"

        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        image_dir.mkdir(parents=True, exist_ok=False)
        parameter_map_dir.mkdir(parents=True, exist_ok=False)
        loss_dir.mkdir(parents=True, exist_ok=False)
        latest_dir.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            image_dir=image_dir,
            parameter_map_dir=parameter_map_dir,
            loss_dir=loss_dir,
            latest_checkpoint=latest_dir / "ckpt.pkl",
        )

    def _write_run_manifest(self) -> None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "unknown"

        manifest = {
            "dataset": str(self.datasetFolder),
            "dataset_name": str(self.dataset.name),
            "train_frame_indices": [
                (
                    -1
                    if getattr(item, "frame_index", None) is None
                    else int(item.frame_index)
                )
                for item in self.dataset.slices
            ],
            "validation_frame_indices": [
                (
                    -1
                    if getattr(item, "frame_index", None) is None
                    else int(item.frame_index)
                )
                for item in self.dataset.slices_valid
            ],
            "roi": getattr(self.dataset, "roi_2d", None),
            "image_shape_hw": [
                int(self.dataset.px_height),
                int(self.dataset.px_width),
            ],
            "world_coordinate_unit": "mm",
            "ultra_distance_unit": self.ultra_distance_unit,
            "renderer": self.renderer_name,
            "network_output_mode": self.output_mode,
            "use_directions": bool(self.nerf.use_direction),
            "encoding": self.nerf.get_encode_name(),
            "psf": {
                "half_size": self.ultra_psf_half_size,
                "lateral_std": self.ultra_psf_lateral_std,
                "axial_std": self.ultra_psf_axial_std,
            },
            "bernoulli_seed": self.ultra_bernoulli_seed,
            "eval_mc_samples": self.ultra_eval_mc_samples,
            "model_query_chunk": self.ultra_query_chunk,
            "ultra_head_initialization": {
                "attenuation": self.ultra_init_attenuation,
                "reflection": self.ultra_init_reflection,
                "border_probability": self.ultra_init_border_probability,
                "scatter_density": self.ultra_init_scatter_density,
                "scatter_amplitude": self.ultra_init_scatter_amplitude,
                "weight_std": self.ultra_init_weight_std,
            },
            "ultra_loss_schedule": {
                "mse_warmup_iterations": self.ultra_mse_warmup_iters,
                "ms_ssim_ramp_iterations": self.ultra_loss_ramp_iters,
                "final_mse_weight": 1.0 - self.ultra_final_ms_ssim_weight,
                "final_ms_ssim_weight": self.ultra_final_ms_ssim_weight,
            },
            "ultra_collapse_monitor": {
                "mean_echo_threshold": self.ultra_collapse_threshold,
                "patience_iterations": self.ultra_collapse_patience,
            },
            "training_seed": self.seed,
            "training_mode": self.training_mode,
            "dataset_physical_calibration": (
                self.dataset.physical_calibration_signature()
            ),
            "ultrasound_sector_mask": self.dataset.sector_mask_signature(),
            "sagittal": {
                "enabled": self.sagittal_supervisor is not None,
                "source": self.sagittal_mat or None,
                "variable": self.sagittal_variable,
                "weight": self.sagittal_weight,
                "start_iteration": self.sagittal_start_iter,
                "ramp_iterations": self.sagittal_ramp_iters,
                "optimize_pose": self.optimize_sagittal_pose,
                "pose_start_iteration": self.sagittal_pose_start_iter,
                "calibration_constraint": "none",
                "ultrasound_sector_mask": (
                    None
                    if self.sagittal_supervisor is None
                    else self.sagittal_supervisor.valid_mask_signature()
                ),
            },
            "loss": (
                (
                    "MSE"
                    if self.ultra_final_ms_ssim_weight == 0
                    else "MSE warm-up -> configured MS-SSIM/MSE blend"
                )
                if self.renderer_name == "ultra_nerf"
                else "NeUF configured loss"
            ),
            "optimizer": "Adam",
            "learning_rate": self.lr,
            "iterations": self.N_iters,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": git_commit,
            "reference_ultra_nerf_commit": REFERENCE_ULTRA_NERF_COMMIT,
        }
        with (self.run_paths.log_dir / "run_manifest.json").open("w") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
        (self.run_paths.log_dir / "git_commit.txt").write_text(git_commit + "\n")
        (self.run_paths.log_dir / "reference_ultra_nerf_commit.txt").write_text(
            REFERENCE_ULTRA_NERF_COMMIT + "\n"
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
                "renderer": self.renderer_name,
                "output_mode": self.output_mode,
                "image_shape_hw": [
                    int(self.dataset.px_height),
                    int(self.dataset.px_width),
                ],
                "ultra_psf_half_size": self.ultra_psf_half_size,
                "ultra_psf_lateral_std": self.ultra_psf_lateral_std,
                "ultra_psf_axial_std": self.ultra_psf_axial_std,
                "ultra_distance_unit": self.ultra_distance_unit,
                "ultra_bernoulli_seed": self.ultra_bernoulli_seed,
                "ultra_eval_mc_samples": self.ultra_eval_mc_samples,
                "ultra_query_chunk": self.ultra_query_chunk,
                "ultra_init_attenuation": self.ultra_init_attenuation,
                "ultra_init_reflection": self.ultra_init_reflection,
                "ultra_init_border_probability": (
                    self.ultra_init_border_probability
                ),
                "ultra_init_scatter_density": self.ultra_init_scatter_density,
                "ultra_init_scatter_amplitude": self.ultra_init_scatter_amplitude,
                "ultra_init_weight_std": self.ultra_init_weight_std,
                "ultra_mse_warmup_iters": self.ultra_mse_warmup_iters,
                "ultra_loss_ramp_iters": self.ultra_loss_ramp_iters,
                "ultra_final_ms_ssim_weight": self.ultra_final_ms_ssim_weight,
                "ultra_collapse_threshold": self.ultra_collapse_threshold,
                "ultra_collapse_patience": self.ultra_collapse_patience,
                "reference_ultra_nerf_commit": REFERENCE_ULTRA_NERF_COMMIT,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "start": iteration,
                "bounding_box": self.dataset.get_bounding_box(),
                "dataset_physical_calibration": (
                    self.dataset.physical_calibration_signature()
                ),
                "ultrasound_sector_mask": self.dataset.sector_mask_signature(),
                "training_mode": self.training_mode,
                "phase_switch_ratio": self.phase_switch_ratio,
                "curriculum_random_ratio": self.curriculum_random_ratio,
                "curriculum_patch_ratio": self.curriculum_patch_ratio,
                "use_loupas": self.use_loupas,
                "noise_sigma_min": self.noise_sigma_min,
                "noise_sigma_max": self.noise_sigma_max,
                "loupas_gamma": self.loupas_gamma,
                "loupas_weight": self.loupas_weight,
                "optimize_poses": self.optimize_poses,
                "pose_anchor_first": self.pose_anchor_first,
                "pose_lr": self.pose_lr,
                "pose_lr_end": self.pose_lr_end,
                "pose_warmup_iters": self.pose_warmup_iters,
                "pose_start_iter": self.pose_start_iter,
                "pose_rotation_reg_weight": self.pose_rotation_reg_weight,
                "pose_translation_reg_weight": self.pose_translation_reg_weight,
                "pose_velocity_reg_weight": self.pose_velocity_reg_weight,
                "pose_acceleration_reg_weight": self.pose_acceleration_reg_weight,
                "pose_grad_clip_norm": self.pose_grad_clip_norm,
                "sagittal_mat": self.sagittal_mat,
                "sagittal_variable": self.sagittal_variable,
                "sagittal_weight": self.sagittal_weight,
                "sagittal_points_per_iter": self.sagittal_points_per_iter,
                "sagittal_start_iter": self.sagittal_start_iter,
                "sagittal_ramp_iters": self.sagittal_ramp_iters,
                "optimize_sagittal_pose": self.optimize_sagittal_pose,
                "sagittal_pose_lr": self.sagittal_pose_lr,
                "sagittal_pose_lr_end": self.sagittal_pose_lr_end,
                "sagittal_pose_warmup_iters": self.sagittal_pose_warmup_iters,
                "sagittal_pose_start_iter": self.sagittal_pose_start_iter,
                "sagittal_pose_rotation_reg_weight": (
                    self.sagittal_pose_rotation_reg_weight
                ),
                "sagittal_pose_translation_reg_weight": (
                    self.sagittal_pose_translation_reg_weight
                ),
                "sagittal_pose_grad_clip_norm": self.sagittal_pose_grad_clip_norm,
            }
        )
        if self.pose_refiner is not None and self.pose_optimizer is not None:
            params.update(
                {
                    "pose_refiner_state_dict": self.pose_refiner.state_dict(),
                    "pose_optimizer_state_dict": self.pose_optimizer.state_dict(),
                    "pose_corrections_se3": self.pose_refiner.corrections().detach().cpu(),
                    "refined_train_poses": self.pose_refiner.refined_poses().detach().cpu(),
                    "pose_source_frame_indices": torch.as_tensor(
                        [
                            getattr(slice_info, "frame_index", None)
                            if getattr(slice_info, "frame_index", None) is not None
                            else -1
                            for slice_info in self.dataset.slices
                        ],
                        dtype=torch.long,
                    ),
                }
            )
        if self.sagittal_supervisor is not None:
            params.update(
                {
                    "sagittal_initial_training_slice": (
                        self.sagittal_supervisor.initial_slice_index
                    ),
                    "sagittal_pose_refiner_state_dict": (
                        self.sagittal_supervisor.pose_refiner.state_dict()
                    ),
                    "sagittal_pose_correction_se3": (
                        self.sagittal_supervisor.pose_refiner.corrections()
                        .detach()
                        .cpu()
                    ),
                    "refined_sagittal_pose": (
                        self.sagittal_supervisor.refined_pose().detach().cpu()
                    ),
                    "sagittal_ultrasound_sector_mask": (
                        self.sagittal_supervisor.valid_mask_signature()
                    ),
                }
            )
            if self.sagittal_pose_optimizer is not None:
                params["sagittal_pose_optimizer_state_dict"] = (
                    self.sagittal_pose_optimizer.state_dict()
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
        self._validation_ultra_maps = {}

        with torch.no_grad():
            for index in self._iter_validation_indices():
                name = VALIDATION_SLICE_NAMES[index]
                if self.renderer_name == "ultra_nerf":
                    if self.ultra_slice_renderer is None:
                        raise RuntimeError("Ultra-NeRF slice renderer is not initialized")
                    source_frame_index = getattr(
                        self.dataset.slices_valid[index],
                        "frame_index",
                        None,
                    )
                    if source_frame_index is None:
                        source_frame_index = index
                    eval_seed = (
                        self.ultra_bernoulli_seed
                        + 1_000_003 * int(source_frame_index)
                    )
                    maps = self.ultra_slice_renderer.render_slice_from_dataset_valid(
                        self.nerf,
                        index,
                        eval_seed=eval_seed,
                    )
                    rendered[name] = maps["intensity_map"].detach()
                    self._validation_ultra_maps[name] = {
                        map_name: value.detach() for map_name, value in maps.items()
                    }
                else:
                    rendered[name] = self.slice_renderer.render_slice_from_dataset_valid(
                        self.nerf,
                        index,
                        reshaped=True,
                    ).detach()
                reference = self._reshape_valid_slice(self.dataset.get_slice_valid_pixels, index)
                if reference is not None:
                    references[name] = reference.detach()

        return rendered, references

    def _render_sagittal_preview(self) -> Optional[torch.Tensor]:
        if self.sagittal_supervisor is None:
            return None

        if self.renderer_name == "ultra_nerf":
            if self.ultra_slice_renderer is None:
                raise RuntimeError("Ultra-NeRF slice renderer is not initialized")
            with torch.no_grad():
                points, viewdirs = self.sagittal_supervisor.refined_geometry()
                maps = self.ultra_slice_renderer.render_points(
                    self.nerf,
                    points,
                    viewdirs,
                    height=self.sagittal_supervisor.height,
                    width=self.sagittal_supervisor.width,
                    eval_seed=self.ultra_bernoulli_seed + 2_000_000_033,
                    valid_mask=self.sagittal_supervisor.valid_mask,
                )
            return maps["intensity_map"].detach()

        prediction = torch.zeros_like(self.sagittal_supervisor.target)
        valid_indices = self.sagittal_supervisor.valid_flat_indices
        chunk_size = 65536
        with torch.no_grad():
            for start in range(0, len(valid_indices), chunk_size):
                indices = valid_indices[start : start + chunk_size]
                points, viewdirs = self.sagittal_supervisor.refined_geometry(indices)
                prediction[indices] = self.slice_renderer.query_points(
                    self.nerf,
                    points,
                    viewdirs,
                    return_sigma=False,
                ).detach()

        return prediction.reshape(
            self.sagittal_supervisor.height,
            self.sagittal_supervisor.width,
        )

    def _write_sagittal_preview(
        self,
        iteration: int,
        prediction: Optional[torch.Tensor],
    ) -> None:
        if prediction is None or self.sagittal_supervisor is None:
            return

        target = self.sagittal_supervisor.target_image().detach()
        abs_diff = torch.abs(prediction - target)
        preview_mse = self._masked_mean(
            torch.square(prediction - target),
            self.sagittal_supervisor.valid_mask,
        )
        self.tb_writer.add_scalar(
            "loss/sagittal_full_mse",
            self._to_float(preview_mse),
            iteration,
        )
        self.tb_writer.add_image(
            "sagittal/prediction",
            self._prepare_tensorboard_image(prediction),
            iteration,
            dataformats="HW",
        )
        self.tb_writer.add_image(
            "sagittal/abs_diff",
            self._prepare_tensorboard_image(abs_diff, normalize=True),
            iteration,
            dataformats="HW",
        )
        if not self.sagittal_target_saved:
            self._save_tensor_and_image("sagittal_target", target)
            self.tb_writer.add_image(
                "sagittal/target",
                self._prepare_tensorboard_image(target),
                iteration,
                dataformats="HW",
            )
            self.sagittal_target_saved = True
        self._save_tensor_and_image(f"sagittal_{iteration}", prediction)

    def _compute_validation_loss(
        self,
        rendered: dict[str, torch.Tensor],
        references: dict[str, torch.Tensor],
    ) -> tuple[Optional[float], Optional[float]]:
        if not rendered or not references:
            return None, None

        losses: list[float] = []
        tv_values: list[float] = []
        valid = self.dataset.get_sector_mask(device=DEVICE)
        valid_h = valid[1:, :] & valid[:-1, :]
        valid_w = valid[:, 1:] & valid[:, :-1]
        for name, prediction in rendered.items():
            if name in references:
                losses.append(
                    self._masked_mean(
                        torch.abs(prediction - references[name]),
                        valid,
                    ).item()
                )

            tv = (
                self._masked_mean(
                    torch.abs(prediction[1:, :] - prediction[:-1, :]), valid_h
                )
                + self._masked_mean(
                    torch.abs(prediction[:, 1:] - prediction[:, :-1]), valid_w
                )
            )
            tv_values.append(tv.item())

        loss_valid = float(np.mean(losses)) if losses else None
        tv_valid = float(np.mean(tv_values)) if tv_values else None
        return loss_valid, tv_valid

    def _compute_temporal_validation_loss(self, rendered: dict[str, torch.Tensor]) -> Optional[float]:
        if not self.previous_validation_slices:
            return None

        deltas = []
        valid = self.dataset.get_sector_mask(device=DEVICE)
        for name, current_slice in rendered.items():
            previous_slice = self.previous_validation_slices.get(name)
            if previous_slice is None:
                continue
            deltas.append(
                self._masked_mean(
                    torch.square(previous_slice - current_slice),
                    valid,
                ).item()
            )

        if not deltas:
            return None
        return float(np.mean(deltas))

    def _save_tensor_and_image(self, base_name: str, tensor: torch.Tensor) -> None:
        cpu_tensor = tensor.detach().cpu()
        torch.save(cpu_tensor, self.run_paths.image_dir / f"{base_name}.pt")
        plt.imsave(
            self.run_paths.image_dir / f"{base_name}.png",
            cpu_tensor.numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    def _save_ultra_parameter_maps(self, iteration: int) -> None:
        if not self.ultra_save_parameter_maps or not self._validation_ultra_maps:
            return
        iteration_dir = self.run_paths.parameter_map_dir / str(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        seed_manifest: dict[str, int] = {}
        for index, (frame_name, maps) in enumerate(self._validation_ultra_maps.items()):
            source_frame_index = getattr(
                self.dataset.slices_valid[index],
                "frame_index",
                None,
            )
            if source_frame_index is None:
                source_frame_index = index
            seed_manifest[frame_name] = (
                self.ultra_bernoulli_seed
                + 1_000_003 * int(source_frame_index)
            )
            frame_dir = iteration_dir / frame_name
            frame_dir.mkdir(parents=True, exist_ok=True)
            for map_name, tensor in maps.items():
                cpu_tensor = tensor.detach().cpu()
                torch.save(cpu_tensor, frame_dir / f"{map_name}.pt")
                plt.imsave(
                    frame_dir / f"{map_name}.png",
                    cpu_tensor.numpy(),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
        with (iteration_dir / "seeds.json").open("w") as output:
            json.dump(seed_manifest, output, indent=2, sort_keys=True)
            output.write("\n")

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
            image = torch.clamp(image, 0.0, 1.0)

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
        sagittal_prediction = self._render_sagittal_preview()

        loss_train = float(np.mean(train_losses)) if train_losses else None
        loss_valid, loss_valid_tv = self._compute_validation_loss(rendered, references)
        loss_gt = self._compute_temporal_validation_loss(rendered)

        self._save_ground_truth_images(references)
        self._save_preview_images(iteration, rendered)
        self._save_ultra_parameter_maps(iteration)
        if self.nerf.dual_encoder is not None:
            self._save_dual_freq_preview(iteration)
        self._write_tensorboard_images(iteration, rendered, references)
        self._write_sagittal_preview(iteration, sagittal_prediction)
        self._write_dual_freq_tensorboard(iteration)
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
        axes[1].set_title("High-frequency mean")
        axes[2].imshow(gate_img, cmap="hot")
        axes[2].set_title(f"Gate (progress={progress:.2f})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.run_paths.image_dir / f"dual_freq_{iteration:06d}.png", dpi=100)
        plt.close(fig)

    def _write_dual_freq_tensorboard(self, iteration: int) -> None:
        if self.nerf.dual_encoder is None or not self.dataset.slices_valid:
            return

        encoder = self.nerf.dual_encoder
        progress = float(getattr(self.nerf, "training_progress", 1.0))
        hf_weight = float(
            encoder.global_hf_weight(progress, device=DEVICE, dtype=torch.float32)
            .detach()
            .cpu()
        )
        self.tb_writer.add_scalar("dual/hf_weight", hf_weight, iteration)

        with torch.no_grad():
            points = self._model_input_points(self.dataset.get_slice_valid_points(0))
            decomp = encoder.forward_decomposed(points, progress)

        height, width = self.dataset.px_height, self.dataset.px_width
        if points.shape[0] != height * width:
            return

        def to_hw_image(tensor: torch.Tensor, clamp_unit: bool = False) -> torch.Tensor:
            image = tensor.detach().cpu().float().reshape(height, width)
            if clamp_unit:
                return torch.clamp(image, 0.0, 1.0)
            image_min = image.min()
            image_max = image.max()
            if float(image_max - image_min) > 1e-8:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = torch.zeros_like(image)
            return image

        name = VALIDATION_SLICE_NAMES[0]
        gate_img = to_hw_image(decomp["gate"].squeeze(-1), clamp_unit=True)
        feat_low_img = to_hw_image(decomp["feat_low"].mean(dim=-1))
        feat_high_img = to_hw_image(decomp["feat_high"].mean(dim=-1))
        weighted_high_img = to_hw_image(decomp["weighted_high"].mean(dim=-1))

        self.tb_writer.add_image(f"dual/gate_map/{name}", gate_img, iteration, dataformats="HW")
        self.tb_writer.add_image(
            f"dual/feat_low_mean/{name}",
            feat_low_img,
            iteration,
            dataformats="HW",
        )
        self.tb_writer.add_image(
            f"dual/feat_high_mean/{name}",
            feat_high_img,
            iteration,
            dataformats="HW",
        )
        self.tb_writer.add_image(
            f"dual/weighted_high_mean/{name}",
            weighted_high_img,
            iteration,
            dataformats="HW",
        )

        gate_flat = gate_img.reshape(-1)
        self.tb_writer.add_scalar("dual/gate_mean", float(gate_flat.mean()), iteration)
        self.tb_writer.add_scalar("dual/gate_std", float(gate_flat.std()), iteration)
        self.tb_writer.add_histogram("dual/gate_histogram", gate_flat, iteration)

    def _next_random_indices(self) -> torch.Tensor:
        if self.random_permutation is None:
            self.random_permutation = torch.randperm(
                self.training_point_count,
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
        return self.dataset.map_valid_training_ranks(indices)

    def _build_training_slice_starts(self) -> torch.Tensor:
        return torch.as_tensor(
            [slice_info.start for slice_info in self.dataset.slices],
            dtype=torch.long,
            device=self.dataset.pixels.device,
        )

    def _patches_per_iter(self, patch_size: int) -> int:
        return self.points_per_iter // (patch_size ** 2)

    def _refine_training_points(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        frame_indices: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pose_refiner is None:
            return points, viewdirs
        return self.pose_refiner(points, viewdirs, frame_indices)

    def _query_training_points(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
    ):
        if self.renderer_name == "ultra_nerf":
            raise RuntimeError(
                "Ultra-NeRF training must query and render a complete frame"
            )
        return self.slice_renderer.query_points(
            self.nerf,
            points,
            viewdirs,
            return_sigma=self.use_loupas,
        )

    def _sample_training_batch(self):
        self._current_points = None
        self._current_viewdirs = None
        self._current_target_slice = None
        self._current_valid_mask = None
        active_mode = self._active_mode

        if self.renderer_name == "ultra_nerf":
            return self._sample_slice_batch()

        if active_mode == "Slice":
            return self._sample_slice_batch()

        if active_mode == "Patch":
            return self._sample_patch_batch()

        if active_mode == "Random":
            return self._sample_random_batch()

        raise ValueError(f"Training mode '{active_mode}' is not implemented")

    def _sample_random_batch(self):
        indices = self._next_random_indices()
        target = self.dataset.get_indices_pixels(indices)
        points, viewdirs = self._refine_training_points(
            self.dataset.get_indices_points(indices),
            self.dataset.get_indices_viewdirs(indices),
            self.dataset.get_indices_frame_indices(indices),
        )
        self._current_points = points
        self._current_viewdirs = viewdirs
        prediction = self._query_training_points(points, viewdirs)
        density, log_sigma = prediction if self.use_loupas else (prediction, None)
        return target, density, log_sigma

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

        valid_origins = self.dataset.get_valid_patch_origins(patch_size)
        if valid_origins.numel() == 0:
            raise ValueError(
                f"No fully valid {patch_size}x{patch_size} patches exist inside "
                "the mandatory ultrasound sector mask"
            )
        slice_indices = torch.randint(
            len(self.dataset.slices),
            (n_patches,),
            dtype=torch.long,
            device=device,
        )
        origin_indices = torch.randint(
            len(valid_origins),
            (n_patches,),
            dtype=torch.long,
            device=device,
        )
        patch_rows = valid_origins[origin_indices, 0]
        patch_cols = valid_origins[origin_indices, 1]

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
        points, viewdirs = self._refine_training_points(
            self.dataset.get_indices_points(indices),
            self.dataset.get_indices_viewdirs(indices),
            self.dataset.get_indices_frame_indices(indices),
        )
        self._current_points = points
        self._current_viewdirs = viewdirs
        prediction = self._query_training_points(points, viewdirs)
        density, log_sigma = prediction if self.use_loupas else (prediction, None)
        return target, density, log_sigma

    def _sample_slice_batch(self):
        slice_index = np.random.randint(len(self.dataset.slices))
        target = self.dataset.get_slice_pixels(slice_index).to(DEVICE)
        valid_mask = self.dataset.get_sector_mask(flatten=True, device=DEVICE)
        points, viewdirs = self._refine_training_points(
            self.dataset.get_slice_points(slice_index),
            self.dataset.get_slice_viewdirs(slice_index),
            slice_index,
        )
        self._current_points = points
        self._current_viewdirs = viewdirs
        self._current_target_slice = target
        self._current_valid_mask = valid_mask
        query_points = points
        if self.jitter_training:
            query_points = self.slice_renderer._apply_jitter(
                query_points,
                self.slice_renderer.width_px,
                self.slice_renderer.height_px,
            )
        if self.renderer_name == "ultra_nerf":
            if self.ultra_slice_renderer is None:
                raise RuntimeError("Ultra-NeRF slice renderer is not initialized")
            maps = self.ultra_slice_renderer.render_points(
                self.nerf,
                query_points,
                viewdirs,
                height=self.dataset.px_height,
                width=self.dataset.px_width,
                valid_mask=valid_mask,
            )
            density = maps["intensity_map"].reshape(-1, 1)
            return target, density, None

        prediction = self.slice_renderer.query_points_masked(
            self.nerf,
            query_points,
            viewdirs,
            valid_mask,
            return_sigma=self.use_loupas,
        )
        density, log_sigma = prediction if self.use_loupas else (prediction, None)
        return target, density, log_sigma

    def _compute_gradient_loss(
        self,
        target: torch.Tensor,
        density: torch.Tensor,
        active_mode: str,
    ) -> Optional[torch.Tensor]:
        structured = self._reshape_for_ssim(target, density, active_mode)
        if structured is None:
            return None

        target_images, density_images, valid_images = structured
        target_grad_x = F.conv2d(target_images, self.scharr_x, padding=1)
        target_grad_y = F.conv2d(target_images, self.scharr_y, padding=1)
        density_grad_x = F.conv2d(density_images, self.scharr_x, padding=1)
        density_grad_y = F.conv2d(density_images, self.scharr_y, padding=1)

        neighbourhood = F.max_pool2d(
            (~valid_images).float(),
            kernel_size=3,
            stride=1,
            padding=1,
        ) == 0
        return self._masked_mean(
            torch.abs(density_grad_x - target_grad_x), neighbourhood
        ) + self._masked_mean(
            torch.abs(density_grad_y - target_grad_y), neighbourhood
        )

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

    def _reshape_for_ssim(
        self,
        target: torch.Tensor,
        density: torch.Tensor,
        active_mode: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if active_mode == "Slice":
            height, width = self.dataset.px_height, self.dataset.px_width
            if target.numel() != height * width or density.numel() != height * width:
                return None
            target_images = target.reshape(1, 1, height, width)
            density_images = density.reshape(1, 1, height, width)
            if self._current_valid_mask is None:
                valid_images = torch.ones_like(target_images, dtype=torch.bool)
            else:
                valid_images = self._current_valid_mask.reshape(1, 1, height, width)
            return target_images, density_images, valid_images

        if active_mode == "Patch":
            patch_area = self.patch_size ** 2
            if target.numel() % patch_area != 0 or density.numel() % patch_area != 0:
                return None
            target_images = target.reshape(-1, 1, self.patch_size, self.patch_size)
            density_images = density.reshape(-1, 1, self.patch_size, self.patch_size)
            valid_images = torch.ones_like(target_images, dtype=torch.bool)
            return target_images, density_images, valid_images

        return None

    def _ssim_window_size_for(self, height: int, width: int) -> int:
        window_size = min(self.ssim_window_size, int(height), int(width))
        if window_size % 2 == 0:
            window_size -= 1
        return window_size

    def _compute_ssim_loss(
        self,
        target: torch.Tensor,
        density: torch.Tensor,
        active_mode: str,
    ) -> Optional[torch.Tensor]:
        structured = self._reshape_for_ssim(target, density, active_mode)
        if structured is None:
            return None

        target_images, density_images, valid_images = structured
        _, _, height, width = target_images.shape
        window_size = self._ssim_window_size_for(height, width)
        if window_size < 3:
            return None

        data_range = torch.clamp(
            target_images.detach().amax() - target_images.detach().amin(),
            min=1.0,
        )
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        padding = window_size // 2

        valid_weights = valid_images.to(dtype=target_images.dtype)

        def local_sum(image: torch.Tensor) -> torch.Tensor:
            padded = F.pad(image, (padding, padding, padding, padding), mode="constant")
            return F.avg_pool2d(padded, kernel_size=window_size, stride=1)

        local_weight = torch.clamp(local_sum(valid_weights), min=1e-8)

        def local_mean(image: torch.Tensor) -> torch.Tensor:
            return local_sum(image * valid_weights) / local_weight

        mu_pred = local_mean(density_images)
        mu_target = local_mean(target_images)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = local_mean(density_images ** 2) - mu_pred_sq
        sigma_target_sq = local_mean(target_images ** 2) - mu_target_sq
        sigma_pred_target = local_mean(density_images * target_images) - mu_pred_target

        numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (
            sigma_pred_sq + sigma_target_sq + c2
        )
        ssim = numerator / torch.clamp(denominator, min=1e-12)
        return 1.0 - self._masked_mean(ssim, valid_images)

    def _compute_tv_loss(self, density):
        d = torch.reshape(density, (self.dataset.px_height, self.dataset.px_width))
        diff_h = d[1:, :] - d[:-1, :]
        diff_w = d[:, 1:] - d[:, :-1]
        valid = self.dataset.get_sector_mask(device=d.device)
        valid_h = valid[1:, :] & valid[:-1, :]
        valid_w = valid[:, 1:] & valid[:, :-1]

        # Huber: 小于threshold用L2（强平滑），大于threshold用L1（保边）
        threshold = 5.0  # 根据你图像灰度范围调s
        tv_h = torch.nn.functional.huber_loss(
            diff_h[valid_h], torch.zeros_like(diff_h[valid_h]),
            delta=threshold, reduction='mean'
        )
        tv_w = torch.nn.functional.huber_loss(
            diff_w[valid_w], torch.zeros_like(diff_w[valid_w]),
            delta=threshold, reduction='mean'
        )
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

    def _compute_lateral_consistency_loss(self, points, viewdirs):
        noise_x = self.smoothness_delta * torch.randn(
            points.shape[0],
            dtype=points.dtype,
            device=points.device,
        )
        perturbed = points.clone()
        perturbed[:, 0] = perturbed[:, 0] + noise_x
        d_orig = self.nerf.query(self._model_input_points(points), viewdirs)
        d_perturbed = self.nerf.query(self._model_input_points(perturbed), viewdirs)
        return torch.mean(torch.abs(d_orig - d_perturbed))

    def _compute_spatial_smoothness(self, points, viewdirs):
        noise = self.smoothness_delta * torch.randn_like(points)
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
        valid_mask = self._current_valid_mask
        return self._masked_mean(
            torch.square(gate - target_gate.detach()),
            valid_mask,
        )

    def _sagittal_weight_at(self, iteration: int) -> float:
        if iteration < self.sagittal_start_iter:
            return 0.0
        if self.sagittal_ramp_iters <= 0:
            return self.sagittal_weight
        ramp_progress = min(
            1.0,
            (iteration - self.sagittal_start_iter) / self.sagittal_ramp_iters,
        )
        return self.sagittal_weight * ramp_progress

    def _compute_sagittal_supervision_loss(
        self,
        iteration: int,
    ) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        if self.sagittal_supervisor is None:
            return None, {}

        effective_weight = self._sagittal_weight_at(iteration)
        if effective_weight <= 0:
            return None, {}

        if self.renderer_name == "ultra_nerf":
            if self.ultra_slice_renderer is None:
                raise RuntimeError("Ultra-NeRF slice renderer is not initialized")
            points, viewdirs = self.sagittal_supervisor.refined_geometry()
            target = self.sagittal_supervisor.target
            maps = self.ultra_slice_renderer.render_points(
                self.nerf,
                points,
                viewdirs,
                height=self.sagittal_supervisor.height,
                width=self.sagittal_supervisor.width,
                valid_mask=self.sagittal_supervisor.valid_mask,
            )
            density = maps["intensity_map"].reshape(-1, 1)
            sagittal_valid_mask = self.sagittal_supervisor.valid_mask.reshape(-1, 1)
        else:
            target, points, viewdirs = self.sagittal_supervisor.sample(
                self.sagittal_points_per_iter
            )
            density = self.slice_renderer.query_points(
                self.nerf,
                points,
                viewdirs,
                return_sigma=False,
            )
            sagittal_valid_mask = None
        target = target.reshape(density.shape)
        sagittal_mse = self._masked_mean(
            torch.square(density - target),
            sagittal_valid_mask,
        )
        if self.renderer_name == "ultra_nerf":
            sagittal_ms_ssim = self._compute_ms_ssim_loss(
                target,
                density,
                valid_mask=sagittal_valid_mask,
            )
            mse_weight, ms_ssim_weight = self._ultra_loss_weights(iteration)
            sagittal_reconstruction = (
                ms_ssim_weight * sagittal_ms_ssim
                + mse_weight * sagittal_mse
            )
        else:
            sagittal_ms_ssim = None
            sagittal_reconstruction = sagittal_mse
        auxiliary_loss = effective_weight * sagittal_reconstruction
        components = {
            "sagittal_mse": sagittal_mse,
            "sagittal_weight": sagittal_mse.new_tensor(effective_weight),
        }
        if sagittal_ms_ssim is not None:
            components["sagittal_ms_ssim"] = sagittal_ms_ssim
            components["sagittal_reconstruction"] = sagittal_reconstruction
            components["sagittal_mse_weight"] = sagittal_mse.new_tensor(mse_weight)
            components["sagittal_ms_ssim_weight"] = sagittal_mse.new_tensor(
                ms_ssim_weight
            )

        if (
            self.sagittal_pose_rotation_reg_weight > 0
            or self.sagittal_pose_translation_reg_weight > 0
        ):
            rotation_reg, translation_reg = (
                self.sagittal_supervisor.pose_refiner.regularization_terms()
            )
            components["sagittal_pose_rotation_reg"] = rotation_reg
            components["sagittal_pose_translation_reg"] = translation_reg
            auxiliary_loss = (
                auxiliary_loss
                + self.sagittal_pose_rotation_reg_weight * rotation_reg
                + self.sagittal_pose_translation_reg_weight * translation_reg
            )

        return auxiliary_loss, components

    def _compute_ms_ssim_loss(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable five-scale MS-SSIM loss for normalized full frames."""
        height, width = int(self.dataset.px_height), int(self.dataset.px_width)
        target_image = target.reshape(1, 1, height, width)
        prediction_image = prediction.reshape(1, 1, height, width)
        if valid_mask is None:
            valid_image = torch.ones_like(target_image)
        else:
            valid_image = valid_mask.reshape(1, 1, height, width).to(
                dtype=target_image.dtype,
                device=target_image.device,
            )
        standard_weights = prediction.new_tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        )
        contrast_terms: list[torch.Tensor] = []
        structure_terms: list[torch.Tensor] = []

        for level in range(len(standard_weights)):
            current_height, current_width = target_image.shape[-2:]
            window_size = self._ssim_window_size_for(
                current_height,
                current_width,
            )
            if window_size < 3:
                break
            padding = window_size // 2

            def local_sum(image: torch.Tensor) -> torch.Tensor:
                padded = F.pad(
                    image,
                    (padding, padding, padding, padding),
                    mode="constant",
                )
                return F.avg_pool2d(padded, kernel_size=window_size, stride=1)

            local_weight = torch.clamp(local_sum(valid_image), min=1e-8)

            def local_mean(image: torch.Tensor) -> torch.Tensor:
                return local_sum(image * valid_image) / local_weight

            mu_prediction = local_mean(prediction_image)
            mu_target = local_mean(target_image)
            prediction_variance = torch.clamp(
                local_mean(prediction_image ** 2) - mu_prediction ** 2,
                min=0.0,
            )
            target_variance = torch.clamp(
                local_mean(target_image ** 2) - mu_target ** 2,
                min=0.0,
            )
            covariance = (
                local_mean(prediction_image * target_image)
                - mu_prediction * mu_target
            )
            c1 = prediction.new_tensor(0.01 ** 2)
            c2 = prediction.new_tensor(0.03 ** 2)
            luminance = (
                2.0 * mu_prediction * mu_target + c1
            ) / (mu_prediction ** 2 + mu_target ** 2 + c1)
            contrast_structure = (
                2.0 * covariance + c2
            ) / (prediction_variance + target_variance + c2)
            contrast_terms.append(
                torch.clamp(
                    self._masked_mean(contrast_structure, valid_image),
                    min=1e-6,
                    max=1.0,
                )
            )
            structure_terms.append(
                torch.clamp(
                    self._masked_mean(
                        luminance * contrast_structure,
                        valid_image,
                    ),
                    min=1e-6,
                    max=1.0,
                )
            )

            if level == len(standard_weights) - 1 or min(
                current_height,
                current_width,
            ) < 4:
                break
            pooled_valid = F.avg_pool2d(valid_image, kernel_size=2, stride=2)
            safe_pooled_valid = torch.clamp(pooled_valid, min=1e-8)
            target_image = F.avg_pool2d(
                target_image * valid_image, kernel_size=2, stride=2
            ) / safe_pooled_valid
            prediction_image = F.avg_pool2d(
                prediction_image * valid_image, kernel_size=2, stride=2
            ) / safe_pooled_valid
            valid_image = pooled_valid

        if not structure_terms:
            return self._masked_mean(
                torch.square(prediction - target),
                valid_mask,
            )
        weights = standard_weights[: len(structure_terms)]
        weights = weights / weights.sum()
        if len(structure_terms) == 1:
            ms_ssim = structure_terms[0]
        else:
            contrast_stack = torch.stack(contrast_terms[:-1])
            ms_ssim = torch.prod(contrast_stack ** weights[:-1])
            ms_ssim = ms_ssim * structure_terms[-1] ** weights[-1]
        return 1.0 - ms_ssim

    def _ultra_loss_weights(self, iteration: int) -> tuple[float, float]:
        """Return MSE and MS-SSIM weights with a stable MSE-first schedule."""
        if iteration < self.ultra_mse_warmup_iters:
            ms_ssim_weight = 0.0
        elif self.ultra_loss_ramp_iters == 0:
            ms_ssim_weight = self.ultra_final_ms_ssim_weight
        else:
            ramp = min(
                1.0,
                max(
                    0.0,
                    (iteration - self.ultra_mse_warmup_iters)
                    / self.ultra_loss_ramp_iters,
                ),
            )
            ms_ssim_weight = self.ultra_final_ms_ssim_weight * ramp
        return 1.0 - ms_ssim_weight, ms_ssim_weight

    def _monitor_ultra_output(self, density: torch.Tensor, iteration: int) -> None:
        """Abort instead of spending a full GPU allocation in a black-output dead zone."""
        if self.renderer_name != "ultra_nerf":
            return
        mean_echo = NeUF._masked_mean(
            density.detach().float(),
            getattr(self, "_current_valid_mask", None),
        )
        if not torch.isfinite(mean_echo):
            raise RuntimeError(
                f"Ultra-NeRF output became non-finite at iteration {iteration}"
            )
        mean_echo_value = float(mean_echo.cpu())
        self.tb_writer.add_scalar("ultra/mean_training_echo", mean_echo_value, iteration)
        if (
            self.ultra_collapse_threshold > 0
            and mean_echo_value < self.ultra_collapse_threshold
        ):
            self._ultra_collapse_count += 1
        else:
            self._ultra_collapse_count = 0
        if self._ultra_collapse_count >= self.ultra_collapse_patience:
            raise RuntimeError(
                "Ultra-NeRF black-output collapse detected: "
                f"mean echo remained below {self.ultra_collapse_threshold:g} for "
                f"{self._ultra_collapse_count} iterations (iteration={iteration})."
            )

    def _compute_training_loss(self, target, density, log_sigma=None, iteration=0):
        target = torch.reshape(target, density.shape)
        valid_mask = self._current_valid_mask
        mse_loss = self._masked_mean(torch.square(density - target), valid_mask)

        if self.renderer_name == "ultra_nerf":
            ms_ssim_loss = self._compute_ms_ssim_loss(
                target,
                density,
                valid_mask=valid_mask,
            )
            mse_weight, ms_ssim_weight = self._ultra_loss_weights(iteration)
            loss = ms_ssim_weight * ms_ssim_loss + mse_weight * mse_loss
            components: dict[str, torch.Tensor] = {
                "mse": mse_loss,
                "ms_ssim": ms_ssim_loss,
                "mse_weight": mse_loss.new_tensor(mse_weight),
                "ms_ssim_weight": mse_loss.new_tensor(ms_ssim_weight),
            }
        elif not self.use_loupas:
            loss = mse_loss
            components: dict[str, torch.Tensor] = {
                "mse": mse_loss,
            }
        else:
            if log_sigma is None:
                raise ValueError("Loupas NLL is enabled but log_sigma was not returned by the model.")
            log_sigma = torch.reshape(log_sigma, density.shape)
            log_sigma_min = torch.log(
                torch.as_tensor(self.noise_sigma_min, dtype=log_sigma.dtype, device=log_sigma.device)
            )
            log_sigma_max = torch.log(
                torch.as_tensor(self.noise_sigma_max, dtype=log_sigma.dtype, device=log_sigma.device)
            )
            stable_log_sigma = torch.clamp(log_sigma, min=log_sigma_min, max=log_sigma_max)
            sigma = torch.exp(stable_log_sigma)

            nll = ((density - target) ** 2) / (2.0 * sigma ** 2) + stable_log_sigma
            nll_loss = self._masked_mean(nll, valid_mask)
            loupas_target = self.loupas_gamma * torch.log(
                density.detach().clamp(min=self.noise_sigma_min, max=1.0)
            )
            loupas_reg = self._masked_mean(
                torch.square(log_sigma - loupas_target),
                valid_mask,
            )

            loss = nll_loss + self.loupas_weight * loupas_reg
            components = {
                "nll": nll_loss,
                "mse": mse_loss,
                "loupas_reg": loupas_reg,
                "sigma_mean": sigma.mean(),
            }
        active_mode = self._active_mode

        if self.renderer_name == "point" and self.ssim_weight > 0:
            ssim_loss = self._compute_ssim_loss(target, density, active_mode)
            if ssim_loss is not None:
                components["ssim"] = ssim_loss
                loss = loss + self.ssim_weight * ssim_loss

        if self.renderer_name == "point" and self.grad_weight > 0:
            gradient_loss = self._compute_gradient_loss(target, density, active_mode)
            if gradient_loss is not None:
                components["gradient"] = gradient_loss
                loss = loss + self.grad_weight * gradient_loss

        if self.renderer_name == "point" and self.tv_weight > 0:
            if active_mode == "Patch":
                tv = self._compute_patch_tv_loss(density)
                components["patch_tv"] = tv
                loss = loss + self.tv_weight * tv

            elif active_mode == "Slice":
                tv = self._compute_tv_loss(density)
                components["tv"] = tv
                loss = loss + self.tv_weight * tv

            elif active_mode == "Random" and self.training_mode == "Random":
                if self._current_points is not None and self._current_viewdirs is not None:
                    if self.use_lateral_perturbation:
                        smooth = self._compute_lateral_consistency_loss(
                            self._current_points, self._current_viewdirs
                        )
                        components["lateral_smoothness"] = smooth
                    else:
                        smooth = self._compute_spatial_smoothness(
                            self._current_points, self._current_viewdirs
                        )
                        components["spatial_smoothness"] = smooth
                    loss = loss + self.tv_weight * smooth

        if (
            self.renderer_name == "point"
            and self.nerf.dual_encoder is not None
            and self._current_points is not None
        ):
            progress = self.nerf.training_progress
            decomp = self.nerf.dual_encoder.forward_decomposed(
                self._model_input_points(self._current_points),
                progress,
            )
            feat_high = decomp["feat_high"]
            gate = decomp["gate"]

            if self._current_valid_mask is None:
                sparsity = torch.mean(torch.abs(feat_high))
            else:
                sparsity = self._masked_mean(
                    torch.abs(feat_high),
                    self._current_valid_mask,
                )
            components["dual_sparsity"] = sparsity
            # loss = loss + self.dual_sparsity_weight * sparsity

            if (
                active_mode == "Slice"
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

        if self.pose_refiner is not None and (
            self.pose_rotation_reg_weight > 0 or self.pose_translation_reg_weight > 0
        ):
            rotation_reg, translation_reg = self.pose_refiner.regularization_terms()
            components["pose_rotation_reg"] = rotation_reg
            components["pose_translation_reg"] = translation_reg
            loss = loss + self.pose_rotation_reg_weight * rotation_reg
            loss = loss + self.pose_translation_reg_weight * translation_reg

        if self.pose_refiner is not None and (
            self.pose_velocity_reg_weight > 0 or self.pose_acceleration_reg_weight > 0
        ):
            velocity_reg, acceleration_reg = (
                self.pose_refiner.trajectory_regularization_terms()
            )
            components["pose_velocity_reg"] = velocity_reg
            components["pose_acceleration_reg"] = acceleration_reg
            loss = loss + self.pose_velocity_reg_weight * velocity_reg
            loss = loss + self.pose_acceleration_reg_weight * acceleration_reg

        sagittal_loss, sagittal_components = self._compute_sagittal_supervision_loss(iteration)
        if sagittal_loss is not None:
            loss = loss + sagittal_loss
            components.update(sagittal_components)

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
                self.nerf.training_progress = iteration / max(1, self.N_iters)
                next_mode = self._current_phase(self.nerf.training_progress)
                if next_mode != self._active_mode:
                    tqdm.tqdm.write(
                        f"[{self.training_mode}] Switched from {self._active_mode} "
                        f"to {next_mode} at iteration={iteration}, "
                        f"progress={self.nerf.training_progress:.3f}"
                    )
                self._active_mode = next_mode

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

                target, density, log_sigma = self._sample_training_batch()
                self._monitor_ultra_output(density, iteration)
                self.optimizer.zero_grad()
                pose_learning_rate = self._set_pose_learning_rate(iteration)
                if self.pose_optimizer is not None:
                    self.pose_optimizer.zero_grad()
                sagittal_pose_learning_rate = (
                    self._set_sagittal_pose_learning_rate(iteration)
                )
                if self.sagittal_pose_optimizer is not None:
                    self.sagittal_pose_optimizer.zero_grad()
                loss, loss_components = self._compute_training_loss(
                    target,
                    density,
                    log_sigma,
                    iteration,
                )
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
                if (
                    self.pose_refiner is not None
                    and self.pose_optimizer is not None
                    and iteration >= self.pose_start_iter
                ):
                    if self.pose_grad_clip_norm > 0:
                        pose_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.pose_refiner.parameters(),
                            max_norm=self.pose_grad_clip_norm,
                        )
                        self.tb_writer.add_scalar(
                            "pose/grad_norm",
                            self._to_float(pose_grad_norm),
                            iteration,
                        )
                if (
                    self.sagittal_supervisor is not None
                    and self.sagittal_pose_optimizer is not None
                    and iteration >= self.sagittal_pose_start_iter
                    and self.sagittal_pose_grad_clip_norm > 0
                ):
                    sagittal_pose_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.sagittal_supervisor.pose_parameters(),
                        max_norm=self.sagittal_pose_grad_clip_norm,
                    )
                    self.tb_writer.add_scalar(
                        "sagittal_pose/grad_norm",
                        self._to_float(sagittal_pose_grad_norm),
                        iteration,
                    )
                self.optimizer.step()
                if self.pose_optimizer is not None and iteration >= self.pose_start_iter:
                    self.pose_optimizer.step()
                if (
                    self.sagittal_pose_optimizer is not None
                    and iteration >= self.sagittal_pose_start_iter
                ):
                    self.sagittal_pose_optimizer.step()
                self.scheduler.step()

                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                self.tb_writer.add_scalar("loss/train_iter", loss_value, iteration)
                self.tb_writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    iteration,
                )
                if self.pose_refiner is not None:
                    self.tb_writer.add_scalar("pose/lr", pose_learning_rate, iteration)
                    pose_log_interval = min(self.i_plot, 100)
                    if iteration % pose_log_interval == 0 or iteration == self.N_iters:
                        pose_statistics = self.pose_refiner.statistics()
                        for name, value in pose_statistics.items():
                            self.tb_writer.add_scalar(f"pose/{name}", value, iteration)
                if self.sagittal_supervisor is not None:
                    self.tb_writer.add_scalar(
                        "sagittal_pose/lr",
                        sagittal_pose_learning_rate,
                        iteration,
                    )
                    pose_log_interval = min(self.i_plot, 100)
                    if iteration % pose_log_interval == 0 or iteration == self.N_iters:
                        sagittal_pose_statistics = (
                            self.sagittal_supervisor.pose_refiner.statistics()
                        )
                        for name, value in sagittal_pose_statistics.items():
                            self.tb_writer.add_scalar(
                                f"sagittal_pose/{name}",
                                value,
                                iteration,
                            )
                phase_value = {"Random": 0.0, "Slice": 1.0, "Patch": 2.0}.get(
                    self._active_mode,
                    -1.0,
                )
                self.tb_writer.add_scalar("train/phase", phase_value, iteration)
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
    parser = argparse.ArgumentParser(prog="neuf", description="Train a NeUF model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Dataset folder or baked dataset path")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to resume from")
    parser.add_argument(
        "--encoding",
        default="Hash",
        choices=["Hash", "Freq", "None", "DUAL_HASH", "DUAL_FREQ", "KRONECKER", "Kronecker"],
    )
    parser.add_argument(
        "--intensity-activation",
        choices=["identity", "sigmoid"],
        default=None,
        help=(
            "Output activation for normalized B-mode intensity. New training defaults "
            "to sigmoid; legacy checkpoints without metadata retain identity."
        ),
    )
    renderer_group = parser.add_argument_group("Ultrasound renderer")
    renderer_group.add_argument(
        "--renderer",
        choices=["point", "ultra_nerf"],
        default="point",
        help="Legacy point-intensity renderer or strict full-frame Ultra-NeRF physics.",
    )
    renderer_group.add_argument("--ultra-psf-half-size", type=int, default=3)
    renderer_group.add_argument("--ultra-psf-lateral-std", type=float, default=2.0)
    renderer_group.add_argument("--ultra-psf-axial-std", type=float, default=1.0)
    renderer_group.add_argument(
        "--ultra-distance-unit",
        choices=["m", "mm"],
        default="m",
        help=(
            "Unit passed to the physics renderer. Dataset millimetres are converted "
            "when 'm' is selected; attenuation is always evaluated in metres."
        ),
    )
    renderer_group.add_argument("--ultra-bernoulli-seed", type=int, default=0)
    renderer_group.add_argument("--ultra-eval-mc-samples", type=int, default=1)
    renderer_group.add_argument("--ultra-query-chunk", type=int, default=65536)
    renderer_group.add_argument(
        "--ultra-save-parameter-maps",
        action="store_true",
        help="Save all validation physics maps as PT and fixed-window PNG files.",
    )
    renderer_group.add_argument(
        "--ultra-init-attenuation",
        type=float,
        default=1.0,
        help="Initial positive attenuation coefficient in inverse metres.",
    )
    renderer_group.add_argument(
        "--ultra-init-reflection",
        type=float,
        default=0.02,
        help="Initial reflection coefficient beta in (0, 1).",
    )
    renderer_group.add_argument(
        "--ultra-init-border-probability",
        type=float,
        default=0.005,
        help="Initial Bernoulli border probability in (0, 1).",
    )
    renderer_group.add_argument(
        "--ultra-init-scatter-density",
        type=float,
        default=0.2,
        help="Initial Bernoulli scattering density in (0, 1).",
    )
    renderer_group.add_argument(
        "--ultra-init-scatter-amplitude",
        type=float,
        default=0.5,
        help="Initial scattering amplitude in (0, 1).",
    )
    renderer_group.add_argument(
        "--ultra-init-weight-std",
        type=float,
        default=1e-4,
        help="Standard deviation of the stabilized five-parameter output weights.",
    )
    renderer_group.add_argument(
        "--ultra-mse-warmup-iters",
        type=int,
        default=500,
        help="Iterations trained with pure MSE before introducing MS-SSIM.",
    )
    renderer_group.add_argument(
        "--ultra-loss-ramp-iters",
        type=int,
        default=1500,
        help="Iterations ramping MS-SSIM weight from 0 to 0.9 after warm-up.",
    )
    renderer_group.add_argument(
        "--ultra-final-ms-ssim-weight",
        type=float,
        default=0.9,
        help=(
            "Final MS-SSIM weight after ramping; the complementary weight is "
            "assigned to MSE. Use 0 for deterministic reconstruction-quality MSE."
        ),
    )
    renderer_group.add_argument(
        "--ultra-collapse-threshold",
        type=float,
        default=1e-6,
        help="Mean echo below which an iteration counts as black-output collapse.",
    )
    renderer_group.add_argument(
        "--ultra-collapse-patience",
        type=int,
        default=20,
        help="Consecutive black-output iterations allowed before aborting.",
    )
    parser.add_argument(
        "--training-mode",
        default="Random",
        choices=["Random", "Slice", "Patch", "CurriculumRS", "CurriculumRPS"],
    )
    parser.add_argument("--points-per-iter", type=int, default=50000)
    parser.add_argument("--patch-size", type=int, default=32, help="Patch side length in pixels for Patch mode")
    parser.add_argument("--nb-iters-max", type=int, default=10000)
    parser.add_argument("--plot-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19981708)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
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
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=0.1,
        help="Weight for SSIM loss on Slice and Patch batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=11,
        help="Odd local window size for SSIM loss.",
    )
    parser.add_argument("--slice-mix-interval", type=int, default=10,
                        help="In Random mode, mix a full slice every N iters for 2D TV loss. 0 to disable.")
    parser.add_argument("--smoothness-delta", type=float, default=0.1,
                        help="Lateral perturbation distance in mm for consistency loss")
    parser.add_argument(
        "--use-lateral-perturbation",
        action="store_true",
        help="Add lateral perturbation consistency loss in Random training mode.",
    )
    parser.add_argument(
        "--phase-switch-ratio",
        dest="phase_switch_ratio",
        type=float,
        default=0.4,
        help="Training progress threshold for CurriculumRS to switch from Random to Slice.",
    )
    parser.add_argument(
        "--curriculum-random-ratio",
        type=float,
        default=0.2,
        help="Fraction of CurriculumRPS spent in Random mode (default: 0.2).",
    )
    parser.add_argument(
        "--curriculum-patch-ratio",
        type=float,
        default=0.5,
        help="Fraction of CurriculumRPS spent in Patch mode (default: 0.5).",
    )

    pose_group = parser.add_argument_group("BARF-style pose optimization")
    pose_toggle = pose_group.add_mutually_exclusive_group()
    pose_toggle.add_argument(
        "--optimize-poses",
        dest="optimize_poses",
        action="store_true",
        default=None,
        help="Jointly optimize one 6-DoF SE(3) correction per training slice.",
    )
    pose_toggle.add_argument(
        "--no-optimize-poses",
        dest="optimize_poses",
        action="store_false",
        help="Disable pose optimization, including when resuming a pose-aware checkpoint.",
    )
    anchor_toggle = pose_group.add_mutually_exclusive_group()
    anchor_toggle.add_argument(
        "--pose-anchor-first",
        dest="pose_anchor_first",
        action="store_true",
        default=None,
        help="Keep the first training pose fixed to remove global gauge freedom (default).",
    )
    anchor_toggle.add_argument(
        "--no-pose-anchor-first",
        dest="pose_anchor_first",
        action="store_false",
        help="Allow all training poses, including the first, to move.",
    )
    pose_group.add_argument(
        "--pose-lr",
        type=float,
        default=None,
        help="Initial SE(3) learning rate (default: 1e-4; restored from checkpoint).",
    )
    pose_group.add_argument(
        "--pose-lr-end",
        type=float,
        default=None,
        help="Final geometrically-decayed pose learning rate (default: 1e-5).",
    )
    pose_group.add_argument(
        "--pose-warmup-iters",
        type=int,
        default=None,
        help="Linear pose-LR warmup length (default: 500); 0 disables it.",
    )
    pose_group.add_argument(
        "--pose-start-iter",
        type=int,
        default=None,
        help="Iteration at which pose updates start (default: 0).",
    )
    pose_group.add_argument(
        "--pose-rotation-reg-weight",
        type=float,
        default=None,
        help="L2 weight for rotation corrections in radians (default: 0).",
    )
    pose_group.add_argument(
        "--pose-translation-reg-weight",
        type=float,
        default=None,
        help="L2 weight for translation corrections in millimetres (default: 0).",
    )
    pose_group.add_argument(
        "--pose-velocity-reg-weight",
        type=float,
        default=None,
        help="L2 weight for first differences of adjacent SE(3) corrections (default: 0).",
    )
    pose_group.add_argument(
        "--pose-acceleration-reg-weight",
        type=float,
        default=None,
        help="L2 weight for second differences of adjacent SE(3) corrections (default: 0).",
    )
    pose_group.add_argument(
        "--pose-grad-clip-norm",
        type=float,
        default=None,
        help="Max pose-gradient norm (default: 1); use 0 to disable clipping.",
    )

    sagittal_group = parser.add_argument_group(
        "Sagittal auxiliary-slice supervision",
        "The image is initialized at the tracked centre slice and can learn an "
        "independent 6-DoF SE(3) correction.",
    )
    sagittal_source = sagittal_group.add_mutually_exclusive_group()
    sagittal_source.add_argument(
        "--sagittal-mat",
        dest="sagittal_mat",
        default=None,
        help="MATLAB file containing the auxiliary sagittal image.",
    )
    sagittal_source.add_argument(
        "--no-sagittal",
        dest="sagittal_mat",
        action="store_const",
        const="",
        help="Disable sagittal supervision, including when resuming a sagittal checkpoint.",
    )
    sagittal_group.add_argument(
        "--sagittal-variable",
        default=None,
        help="Image variable in --sagittal-mat (default: data_sag).",
    )
    sagittal_group.add_argument(
        "--sagittal-weight",
        type=float,
        default=None,
        help="Weight of sampled sagittal MSE in the total loss (default: 1).",
    )
    sagittal_group.add_argument(
        "--sagittal-points-per-iter",
        type=int,
        default=None,
        help="Sagittal pixels sampled at every training iteration (default: 8192).",
    )
    sagittal_group.add_argument(
        "--sagittal-start-iter",
        type=int,
        default=None,
        help="Iteration at which sagittal image supervision begins (default: 0).",
    )
    sagittal_group.add_argument(
        "--sagittal-ramp-iters",
        type=int,
        default=None,
        help="Iterations used to ramp sagittal image weight from zero (default: 0).",
    )
    sagittal_pose_toggle = sagittal_group.add_mutually_exclusive_group()
    sagittal_pose_toggle.add_argument(
        "--optimize-sagittal-pose",
        dest="optimize_sagittal_pose",
        action="store_true",
        default=None,
        help="Learn a 6-DoF correction for the sagittal centre pose (default).",
    )
    sagittal_pose_toggle.add_argument(
        "--no-optimize-sagittal-pose",
        dest="optimize_sagittal_pose",
        action="store_false",
        help="Use a fixed tracked centre pose for the sagittal image.",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-lr",
        type=float,
        default=None,
        help="Initial sagittal SE(3) learning rate (default: 1e-4).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-lr-end",
        type=float,
        default=None,
        help="Final geometrically-decayed sagittal pose LR (default: 1e-5).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-warmup-iters",
        type=int,
        default=None,
        help="Linear sagittal pose-LR warmup length (default: 500).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-start-iter",
        type=int,
        default=None,
        help="Iteration at which sagittal pose updates start (default: 0).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-rotation-reg-weight",
        type=float,
        default=None,
        help="L2 weight for sagittal rotation correction in radians (default: 0).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-translation-reg-weight",
        type=float,
        default=None,
        help="L2 weight for sagittal translation correction in mm (default: 0).",
    )
    sagittal_group.add_argument(
        "--sagittal-pose-grad-clip-norm",
        type=float,
        default=None,
        help="Max sagittal pose-gradient norm (default: 1); 0 disables clipping.",
    )

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
    kronecker_group = parser.add_argument_group("Kronecker tri-plane Hash PE parameters")
    kronecker_group.add_argument(
        "--kronecker-n-levels-lateral",
        dest="kronecker_n_levels_lateral",
        type=int,
        default=DEFAULT_KRONECKER_N_LEVELS_LATERAL,
        help="Number of levels for the low-resolution lateral XY plane.",
    )
    kronecker_group.add_argument(
        "--kronecker-n-levels-axial",
        dest="kronecker_n_levels_axial",
        type=int,
        default=DEFAULT_KRONECKER_N_LEVELS_AXIAL,
        help="Number of levels for axial XZ/YZ planes.",
    )
    kronecker_group.add_argument(
        "--kronecker-finest-lateral",
        dest="kronecker_finest_lateral",
        type=int,
        default=DEFAULT_KRONECKER_FINEST_LATERAL,
        help="Finest resolution for the lateral XY plane.",
    )
    kronecker_group.add_argument(
        "--kronecker-finest-axial",
        dest="kronecker_finest_axial",
        type=int,
        default=DEFAULT_KRONECKER_FINEST_AXIAL,
        help="Finest resolution for axial XZ/YZ planes.",
    )
    kronecker_group.add_argument(
        "--kronecker-n-features-per-level",
        dest="kronecker_n_features_per_level",
        type=int,
        default=DEFAULT_HASH_N_FEATURES_PER_LEVEL,
        help="Feature count per Kronecker hash level.",
    )
    kronecker_group.add_argument(
        "--kronecker-log2-hashmap-size",
        dest="kronecker_log2_hashmap_size",
        type=int,
        default=DEFAULT_HASH_LOG2_HASHMAP_SIZE,
        help="Kronecker hash table log2 size.",
    )
    kronecker_group.add_argument(
        "--kronecker-base-resolution",
        dest="kronecker_base_resolution",
        type=int,
        default=DEFAULT_KRONECKER_BASE_RESOLUTION,
        help="Base resolution shared by all Kronecker planes.",
    )
    kronecker_group.add_argument(
        "--kronecker-combine",
        dest="kronecker_combine",
        choices=["cat", "sum", "product"],
        default=DEFAULT_KRONECKER_COMBINE,
        help="How to combine XY/XZ/YZ plane features.",
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
    noise_group = parser.add_argument_group("Heteroscedastic noise model parameters")
    noise_group.add_argument(
        "--use-loupas",
        dest="use_loupas",
        action="store_true",
        default=None,
        help="Use the Loupas-inspired heteroscedastic model and NLL loss.",
    )
    noise_group.add_argument(
        "--no-loupas",
        dest="use_loupas",
        action="store_false",
        help="Disable the Loupas model and train with plain MSE loss.",
    )
    noise_group.add_argument(
        "--noise-sigma-min",
        dest="noise_sigma_min",
        type=float,
        default=1e-3,
        help="Minimum predicted sigma used in the heteroscedastic NLL.",
    )
    noise_group.add_argument(
        "--noise-sigma-max",
        dest="noise_sigma_max",
        type=float,
        default=1.0,
        help="Maximum predicted sigma used in the heteroscedastic NLL.",
    )
    noise_group.add_argument(
        "--loupas-gamma",
        dest="loupas_gamma",
        type=float,
        default=0.5,
        help="Exponent gamma for log_sigma ~= gamma * log(I_clean).",
    )
    noise_group.add_argument(
        "--loupas-weight",
        dest="loupas_weight",
        type=float,
        default=0.1,
        help="Weight for the Loupas-inspired sigma-density regularization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    neuf = NeUF(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        encoding=args.encoding,
        renderer=args.renderer,
        ultra_psf_half_size=args.ultra_psf_half_size,
        ultra_psf_lateral_std=args.ultra_psf_lateral_std,
        ultra_psf_axial_std=args.ultra_psf_axial_std,
        ultra_distance_unit=args.ultra_distance_unit,
        ultra_bernoulli_seed=args.ultra_bernoulli_seed,
        ultra_eval_mc_samples=args.ultra_eval_mc_samples,
        ultra_query_chunk=args.ultra_query_chunk,
        ultra_save_parameter_maps=args.ultra_save_parameter_maps,
        ultra_init_attenuation=args.ultra_init_attenuation,
        ultra_init_reflection=args.ultra_init_reflection,
        ultra_init_border_probability=args.ultra_init_border_probability,
        ultra_init_scatter_density=args.ultra_init_scatter_density,
        ultra_init_scatter_amplitude=args.ultra_init_scatter_amplitude,
        ultra_init_weight_std=args.ultra_init_weight_std,
        ultra_mse_warmup_iters=args.ultra_mse_warmup_iters,
        ultra_loss_ramp_iters=args.ultra_loss_ramp_iters,
        ultra_final_ms_ssim_weight=args.ultra_final_ms_ssim_weight,
        ultra_collapse_threshold=args.ultra_collapse_threshold,
        ultra_collapse_patience=args.ultra_collapse_patience,
        intensity_activation=args.intensity_activation,
        training_mode=args.training_mode,
        points_per_iter=args.points_per_iter,
        patch_size=args.patch_size,
        nb_iters_max=args.nb_iters_max,
        plot_freq=args.plot_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        grad_weight=args.grad_weight,
        grad_clip_norm=args.grad_clip_norm,
        grad_blur_kernel_size=args.grad_blur_kernel_size,
        grad_blur_sigma=args.grad_blur_sigma,
        root=args.root,
        baked_dataset=not args.raw_dataset,
        jitter_training=args.jitter_training,
        tv_weight=args.tv_weight,
        ssim_weight=args.ssim_weight,
        ssim_window_size=args.ssim_window_size,
        slice_mix_interval=args.slice_mix_interval,
        smoothness_delta=args.smoothness_delta,
        use_lateral_perturbation=args.use_lateral_perturbation,
        phase_switch_ratio=args.phase_switch_ratio,
        curriculum_random_ratio=args.curriculum_random_ratio,
        curriculum_patch_ratio=args.curriculum_patch_ratio,
        optimize_poses=args.optimize_poses,
        pose_anchor_first=args.pose_anchor_first,
        pose_lr=args.pose_lr,
        pose_lr_end=args.pose_lr_end,
        pose_warmup_iters=args.pose_warmup_iters,
        pose_start_iter=args.pose_start_iter,
        pose_rotation_reg_weight=args.pose_rotation_reg_weight,
        pose_translation_reg_weight=args.pose_translation_reg_weight,
        pose_velocity_reg_weight=args.pose_velocity_reg_weight,
        pose_acceleration_reg_weight=args.pose_acceleration_reg_weight,
        pose_grad_clip_norm=args.pose_grad_clip_norm,
        sagittal_mat=args.sagittal_mat,
        sagittal_variable=args.sagittal_variable,
        sagittal_weight=args.sagittal_weight,
        sagittal_points_per_iter=args.sagittal_points_per_iter,
        sagittal_start_iter=args.sagittal_start_iter,
        sagittal_ramp_iters=args.sagittal_ramp_iters,
        optimize_sagittal_pose=args.optimize_sagittal_pose,
        sagittal_pose_lr=args.sagittal_pose_lr,
        sagittal_pose_lr_end=args.sagittal_pose_lr_end,
        sagittal_pose_warmup_iters=args.sagittal_pose_warmup_iters,
        sagittal_pose_start_iter=args.sagittal_pose_start_iter,
        sagittal_pose_rotation_reg_weight=(
            args.sagittal_pose_rotation_reg_weight
        ),
        sagittal_pose_translation_reg_weight=(
            args.sagittal_pose_translation_reg_weight
        ),
        sagittal_pose_grad_clip_norm=args.sagittal_pose_grad_clip_norm,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_hashmap_size=args.hash_log2_hashmap_size,
        hash_base_resolution=args.hash_base_resolution,
        hash_finest_resolution=args.hash_finest_resolution,
        kronecker_n_levels_lateral=args.kronecker_n_levels_lateral,
        kronecker_n_levels_axial=args.kronecker_n_levels_axial,
        kronecker_finest_lateral=args.kronecker_finest_lateral,
        kronecker_finest_axial=args.kronecker_finest_axial,
        kronecker_n_features_per_level=args.kronecker_n_features_per_level,
        kronecker_log2_hashmap_size=args.kronecker_log2_hashmap_size,
        kronecker_base_resolution=args.kronecker_base_resolution,
        kronecker_combine=args.kronecker_combine,
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
        noise_sigma_min=args.noise_sigma_min,
        noise_sigma_max=args.noise_sigma_max,
        use_loupas=args.use_loupas,
        loupas_gamma=args.loupas_gamma,
        loupas_weight=args.loupas_weight,
    )
    neuf.run()


if __name__ == "__main__":
    main()
