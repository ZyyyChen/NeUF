"""Full-frame NeUF adapter for the strict Ultra-NeRF renderer."""

from __future__ import annotations

from typing import Optional

import torch

from neuf.slice_renderer_base import DEVICE, BaseSliceRenderer
from neuf.ultra_nerf_renderer import UltraNeRFRenderer, make_generator


class UltraNeRFSliceRenderer(BaseSliceRenderer):
    """Query a NeUF field in chunks, then render complete ultrasound A-lines."""

    def __init__(
        self,
        dataset,
        *,
        psf_half_size: int = 3,
        psf_lateral_std: float = 2.0,
        psf_axial_std: float = 1.0,
        distance_unit: str = "m",
        bernoulli_seed: int = 0,
        eval_mc_samples: int = 1,
        query_chunk: int = 65536,
    ) -> None:
        super().__init__(dataset)
        if eval_mc_samples < 1:
            raise ValueError(
                f"eval_mc_samples must be >= 1, got {eval_mc_samples}"
            )
        if query_chunk < 1:
            raise ValueError(f"query_chunk must be >= 1, got {query_chunk}")

        self.physics = UltraNeRFRenderer(
            psf_half_size=psf_half_size,
            psf_lateral_std=psf_lateral_std,
            psf_axial_std=psf_axial_std,
            distance_unit=distance_unit,
        ).to(DEVICE)
        self.distance_unit = str(distance_unit).lower()
        self.bernoulli_seed = int(bernoulli_seed)
        self.eval_mc_samples = int(eval_mc_samples)
        self.query_chunk = int(query_chunk)
        self._training_generator_device: Optional[torch.device] = None
        self._border_generator: Optional[torch.Generator] = None
        self._scatter_generator: Optional[torch.Generator] = None

    @staticmethod
    def validate_axial_layout(points_hw3: torch.Tensor) -> None:
        """Check that every stored image column is a complete shallow-to-deep A-line."""
        if points_hw3.ndim != 3 or points_hw3.shape[-1] != 3:
            raise ValueError(
                f"points_hw3 must have shape [H, W, 3], got {tuple(points_hw3.shape)}"
            )
        if points_hw3.shape[0] < 2:
            raise ValueError("An A-line requires at least two axial pixels")
        axial_steps = torch.linalg.vector_norm(
            points_hw3[1:, :, :] - points_hw3[:-1, :, :],
            dim=-1,
        )
        if not torch.all(torch.isfinite(axial_steps)):
            raise ValueError("A-line coordinates contain non-finite axial steps")
        if torch.any(axial_steps <= 0):
            raise ValueError(
                "A-line coordinates must advance strictly from shallow to deep"
            )

    def _z_values(self, points_hw3: torch.Tensor) -> torch.Tensor:
        self.validate_axial_layout(points_hw3)
        axial_steps_mm = torch.linalg.vector_norm(
            points_hw3[1:, :, :] - points_hw3[:-1, :, :],
            dim=-1,
        ).transpose(0, 1)
        z_vals_wh = torch.cat(
            [
                torch.zeros(
                    (points_hw3.shape[1], 1),
                    dtype=points_hw3.dtype,
                    device=points_hw3.device,
                ),
                torch.cumsum(axial_steps_mm, dim=1),
            ],
            dim=1,
        )
        if self.distance_unit == "m":
            z_vals_wh = z_vals_wh * 1e-3
        return z_vals_wh

    def _query_raw(
        self,
        model,
        points_hw3: torch.Tensor,
        viewdirs_hw3: torch.Tensor,
        valid_mask_hw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_points = points_hw3.reshape(-1, 3)
        flat_viewdirs = viewdirs_hw3.reshape(-1, 3)
        if valid_mask_hw is None:
            flat_valid = torch.ones(
                (flat_points.shape[0],),
                dtype=torch.bool,
                device=flat_points.device,
            )
        else:
            flat_valid = torch.as_tensor(
                valid_mask_hw,
                dtype=torch.bool,
                device=flat_points.device,
            ).reshape(-1)
            if flat_valid.numel() != flat_points.shape[0]:
                raise ValueError(
                    "valid_mask must contain one value per frame pixel, got "
                    f"{flat_valid.numel()} and {flat_points.shape[0]}"
                )
        query_points = self._normalize_points_if_needed(
            model,
            flat_points[flat_valid],
            self._require_dataset().point_min_dev,
        )
        raw_valid = model.query(
            query_points,
            flat_viewdirs[flat_valid],
            netchunk=self.query_chunk,
            return_sigma=False,
        )
        if raw_valid.shape != (int(flat_valid.sum()), 5):
            raise ValueError(
                "Ultra-NeRF field must return [valid_pixels, 5], got "
                f"{tuple(raw_valid.shape)}"
            )
        raw_flat = torch.zeros(
            (flat_points.shape[0], 5),
            dtype=raw_valid.dtype,
            device=raw_valid.device,
        )
        raw_flat[flat_valid] = raw_valid
        raw_hw5 = raw_flat.reshape(
            points_hw3.shape[0],
            points_hw3.shape[1],
            5,
        )
        return raw_hw5.permute(1, 0, 2).contiguous()

    def _training_generators(
        self,
        device: torch.device,
    ) -> tuple[torch.Generator, torch.Generator]:
        if self._training_generator_device != device:
            self._border_generator = make_generator(device, self.bernoulli_seed)
            self._scatter_generator = make_generator(device, self.bernoulli_seed + 1)
            self._training_generator_device = device
        if self._border_generator is None or self._scatter_generator is None:
            raise RuntimeError("Failed to initialize Ultra-NeRF random generators")
        return self._border_generator, self._scatter_generator

    def render_points(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        *,
        height: int,
        width: int,
        eval_seed: Optional[int] = None,
        eval_mc_samples: Optional[int] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Render one full frame and return all maps in project ``[H, W]`` layout."""
        height = int(height)
        width = int(width)
        if points.numel() != height * width * 3:
            raise ValueError(
                f"points do not fill a {height}x{width} frame: {tuple(points.shape)}"
            )
        if viewdirs.numel() != height * width * 3:
            raise ValueError(
                f"viewdirs do not fill a {height}x{width} frame: {tuple(viewdirs.shape)}"
            )

        points_hw3 = points.reshape(height, width, 3)
        viewdirs_hw3 = viewdirs.reshape(height, width, 3)
        if valid_mask is None:
            valid_mask_hw = None
            valid_mask_wh = None
        else:
            valid_mask_hw = torch.as_tensor(
                valid_mask,
                dtype=torch.bool,
                device=points_hw3.device,
            ).reshape(height, width)
            valid_mask_wh = valid_mask_hw.transpose(0, 1).contiguous()
        raw_wh5 = self._query_raw(
            model,
            points_hw3,
            viewdirs_hw3,
            valid_mask_hw,
        )
        z_vals_wh = self._z_values(points_hw3)

        if eval_seed is None:
            border_generator, scatter_generator = self._training_generators(
                raw_wh5.device
            )
            rendered_wh = self.physics(
                raw_wh5,
                z_vals_wh,
                border_generator=border_generator,
                scatter_generator=scatter_generator,
                valid_mask=valid_mask_wh,
            )
        else:
            sample_count = (
                self.eval_mc_samples
                if eval_mc_samples is None
                else int(eval_mc_samples)
            )
            if sample_count < 1:
                raise ValueError(
                    f"eval_mc_samples must be >= 1, got {sample_count}"
                )
            samples = []
            for sample_index in range(sample_count):
                sample_seed = int(eval_seed) + 2 * sample_index
                samples.append(
                    self.physics(
                        raw_wh5,
                        z_vals_wh,
                        border_generator=make_generator(raw_wh5.device, sample_seed),
                        scatter_generator=make_generator(
                            raw_wh5.device,
                            sample_seed + 1,
                        ),
                        valid_mask=valid_mask_wh,
                    )
                )
            rendered_wh = {
                name: torch.stack([sample[name] for sample in samples], dim=0).mean(0)
                for name in samples[0]
            }

        return {
            name: value.transpose(0, 1).contiguous()
            for name, value in rendered_wh.items()
        }

    def render_slice_from_dataset(
        self,
        model,
        slice_number: int,
        *,
        eval_seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        dataset = self._require_dataset()
        return self.render_points(
            model,
            dataset.get_slice_points(slice_number),
            dataset.get_slice_viewdirs(slice_number),
            height=dataset.px_height,
            width=dataset.px_width,
            eval_seed=eval_seed,
            valid_mask=dataset.get_sector_mask(device=dataset.points.device),
        )

    def render_slice_from_dataset_valid(
        self,
        model,
        slice_number: int,
        *,
        eval_seed: int,
    ) -> dict[str, torch.Tensor]:
        dataset = self._require_dataset()
        return self.render_points(
            model,
            dataset.get_slice_valid_points(slice_number),
            dataset.get_slice_valid_viewdirs(slice_number),
            height=dataset.px_height,
            width=dataset.px_width,
            eval_seed=eval_seed,
            valid_mask=dataset.get_sector_mask(device=dataset.points_valid.device),
        )
