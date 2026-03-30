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
        self.seed = int(kwargs.get("seed", 19981708))
        self.N_iters = int(kwargs.get("nb_iters_max", 8500))
        self.i_plot = int(kwargs.get("plot_freq", 100))
        self.i_save = int(kwargs.get("save_freq", 100))
        self.baked_dataset = bool(kwargs.get("baked_dataset", True))
        self.training_mode = kwargs.get("training_mode", "Random")
        self.points_per_iter = int(kwargs.get("points_per_iter", 50000))
        self.jitter_training = bool(kwargs.get("jitter_training", False))
        self.encoding = kwargs.get("encoding", "None")
        self.datasetFolder = kwargs.get("dataset", DEFAULT_DATASET_PATH)
        self.ckptFile = kwargs.get("checkpoint", "")
        self.rootPoint = Path(kwargs.get("root", ".")).expanduser()

        self._validate_configuration()

        self.dataset: Dataset
        self.nerf: NeRF
        self.optimizer: torch.optim.Optimizer
        self.slice_renderer: SliceRenderer
        self.start = 0

        self.mse = torch.nn.MSELoss()
        self.scharr_x, self.scharr_y = self._build_scharr_kernels()
        self.previous_validation_slices: dict[str, torch.Tensor] = {}
        self.gt_saved = False
        self.tb_reference_images_logged = False

        self.random_permutation: Optional[torch.Tensor] = None
        self.random_start_index = 0

        checkpoint = self._load_checkpoint(self.ckptFile)
        if checkpoint is not None:
            self._initialize_from_checkpoint(checkpoint)
        else:
            self._initialize_from_scratch()

        self.slice_renderer = SliceRenderer(self.dataset)
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
        if self.points_per_iter <= 0:
            raise ValueError(f"points_per_iter must be >= 1, got {self.points_per_iter}")

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
            lr=5e-4,
            betas=(0.9, 0.999),
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start = int(checkpoint["start"])

    def _initialize_from_scratch(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.dataset = self._load_dataset()
        self.nerf = NeRF()
        self._initialize_model_encoding()
        self.nerf.init_model(8, 256)
        self.optimizer = torch.optim.Adam(
            params=self.nerf.grad_vars(),
            lr=5e-4,
            betas=(0.9, 0.999),
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
                use_encoding=True,
                use_directions=False,
            )
            return

        if encoding_name == "none":
            self.nerf.init_hash_encoding(
                bounding_box=self.dataset.get_bounding_box(),
                use_encoding=False,
                use_directions=False,
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
            writer.writerow(["iteration", "loss_train", "loss_valid", "loss_gt"])

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
    ) -> Optional[float]:
        if not rendered or not references:
            return None

        losses = [self.mse(rendered[name], references[name]).item() for name in rendered if name in references]
        if not losses:
            return None
        return float(np.mean(losses))

    def _compute_temporal_validation_loss(self, rendered: dict[str, torch.Tensor]) -> Optional[float]:
        if not self.previous_validation_slices:
            return None

        deltas = []
        for name, current_slice in rendered.items():
            previous_slice = self.previous_validation_slices.get(name)
            if previous_slice is None:
                continue
            deltas.append(torch.sum(torch.square(previous_slice - current_slice)).item())

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
    ) -> None:
        if loss_train is not None:
            self.tb_writer.add_scalar("loss/train", loss_train, iteration)
        if loss_valid is not None:
            self.tb_writer.add_scalar("loss/valid", loss_valid, iteration)
        if loss_gt is not None:
            self.tb_writer.add_scalar("loss/gt", loss_gt, iteration)

        with self.loss_csv_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([iteration, loss_train, loss_valid, loss_gt])

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
        loss_valid = self._compute_validation_loss(rendered, references)
        loss_gt = self._compute_temporal_validation_loss(rendered)

        self._save_ground_truth_images(references)
        self._save_preview_images(iteration, rendered)
        self._write_tensorboard_images(iteration, rendered, references)
        self._write_metrics(iteration, loss_train, loss_valid, loss_gt)
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

    def _sample_training_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_mode == "Slice":
            slice_index = np.random.randint(len(self.dataset.slices))
            target = self.dataset.get_slice_pixels(slice_index).to(DEVICE)
            density = self.slice_renderer.render_slice_from_dataset(
                self.nerf,
                slice_index,
                jitter=self.jitter_training,
            )
            return target, density

        if self.training_mode == "Random":
            indices = self._next_random_indices()
            target = self.dataset.get_indices_pixels(indices)
            density = self.slice_renderer.query_random_positions(self.nerf, indices, reshaped=False)
            return target, density

        raise ValueError(f"Training mode '{self.training_mode}' is not implemented")

    def _compute_gradient_loss(self, target: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        target_slice = torch.reshape(
            target,
            (self.dataset.px_height, self.dataset.px_width),
        ).unsqueeze(0).unsqueeze(0)
        density_slice = torch.reshape(
            density,
            (self.dataset.px_height, self.dataset.px_width),
        ).unsqueeze(0).unsqueeze(0)

        target_grad_x = torch.nn.functional.conv2d(target_slice, self.scharr_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target_slice, self.scharr_y, padding=1)
        density_grad_x = torch.nn.functional.conv2d(density_slice, self.scharr_x, padding=1)
        density_grad_y = torch.nn.functional.conv2d(density_slice, self.scharr_y, padding=1)

        return self.mse(density_grad_x, target_grad_x) + self.mse(density_grad_y, target_grad_y)

    def _compute_training_loss(self, target: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        loss = self.mse(density, target)
        if self.training_mode == "Slice":
            loss = loss + self.grad_weight * self._compute_gradient_loss(target, density)
        return loss

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

                target, density = self._sample_training_batch()
                self.optimizer.zero_grad()
                loss = self._compute_training_loss(target, density)
                loss.backward()
                self.optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
        finally:
            self.tb_writer.close()
            progress_bar.close()

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
    parser.add_argument("--encoding", default="Hash", choices=["Hash", "Freq", "None"])
    parser.add_argument("--training-mode", default="Random", choices=["Random", "Slice"])
    parser.add_argument("--points-per-iter", type=int, default=50000)
    parser.add_argument("--nb-iters-max", type=int, default=8500)
    parser.add_argument("--plot-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19981708)
    parser.add_argument("--grad-weight", type=float, default=0.1)
    parser.add_argument("--root", default=".", help="Root folder for logs and latest checkpoint")
    parser.add_argument("--raw-dataset", action="store_true", help="Treat --dataset as an unbaked folder")
    parser.add_argument("--jitter-training", action="store_true", help="Enable point jitter during training")
    return parser.parse_args()


def main():
    args = parse_args()
    neuf = NeUF(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        encoding=args.encoding,
        training_mode=args.training_mode,
        points_per_iter=args.points_per_iter,
        nb_iters_max=args.nb_iters_max,
        plot_freq=args.plot_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        grad_weight=args.grad_weight,
        root=args.root,
        baked_dataset=not args.raw_dataset,
        jitter_training=args.jitter_training,
    )
    neuf.run()


if __name__ == "__main__":
    main()
