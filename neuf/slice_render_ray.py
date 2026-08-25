from __future__ import annotations

import torch
import torch.nn.functional as F

from neuf.slice_renderer_base import DEVICE, BaseSliceRenderer
from neuf.utils import get_oriented_points_and_views


class SliceRenderer(BaseSliceRenderer):
    """Ray-marching renderer with the same public API as ``slice_renderer``."""

    def __init__(
        self,
        dataset=None,
        px_width=0,
        px_height=0,
        width=0,
        height=0,
        point_min=None,
        point_max=None,
        decimation=1,
        n_samples: int = 16,
        ray_length: float | None = None,
        near: float | None = None,
        far: float | None = None,
        chunk: int = 1024 * 32,
        integration: str = "mean",
        density_scale: float = 1.0,
    ):
        self.n_samples = max(1, int(n_samples))
        self.ray_length = None if ray_length is None else float(ray_length)
        self.near = None if near is None else float(near)
        self.far = None if far is None else float(far)
        self.chunk = int(chunk)
        self.integration = integration
        self.density_scale = float(density_scale)

        if self.integration not in {"mean", "sum", "alpha"}:
            raise ValueError("integration must be one of: 'mean', 'sum', 'alpha'")

        super().__init__(
            dataset=dataset,
            px_width=px_width,
            px_height=px_height,
            width=width,
            height=height,
            point_min=point_min,
            point_max=point_max,
            decimation=decimation,
            prefer_dataset_offsets=True,
        )
        self._initialize_ray_bounds()

    def _initialize_ray_bounds(self) -> None:
        if self.ray_length is None:
            if self.dataset is not None and getattr(self.dataset, "roi_px_size_height_mm", 0.0) > 0:
                self.ray_length = float(self.dataset.roi_px_size_height_mm)
            else:
                self.ray_length = self.height / max(1, self.height_px)

        if self.near is None and self.far is None:
            self.near = -0.5 * self.ray_length
            self.far = 0.5 * self.ray_length
        elif self.near is None:
            self.near = 0.0
        elif self.far is None:
            self.far = self.near + self.ray_length

    def _ray_offsets(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.n_samples == 1:
            return torch.zeros((1,), dtype=dtype, device=device)
        return torch.linspace(float(self.near), float(self.far), self.n_samples, dtype=dtype, device=device)

    def _sample_distances(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offsets = self._ray_offsets(device, dtype)
        if offsets.numel() <= 1:
            return torch.ones_like(offsets)
        distances = torch.abs(offsets[1:] - offsets[:-1])
        return torch.cat([distances, distances[-1:]], dim=0)

    def _build_ray_samples(
        self,
        origins: torch.Tensor,
        viewdirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origins = torch.reshape(origins, (-1, origins.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        viewdirs = F.normalize(viewdirs, dim=-1, eps=1e-8)

        offsets = self._ray_offsets(origins.device, origins.dtype)
        sample_points = origins[:, None, :] + viewdirs[:, None, :] * offsets[None, :, None]
        sample_viewdirs = viewdirs[:, None, :].expand_as(sample_points)
        return sample_points, sample_viewdirs

    def _query_flat_samples(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        bb_min_dev: torch.Tensor,
        return_sigma: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        points = torch.reshape(points, (-1, points.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        query_points = self._normalize_points_if_needed(model, points, bb_min_dev)
        prediction = model.query(query_points, viewdirs, return_sigma=return_sigma)
        if return_sigma:
            density, log_sigma = prediction
            return density.to(points.device), log_sigma.to(points.device)
        return prediction.to(points.device)

    def _query_ray_samples(
        self,
        model,
        sample_points: torch.Tensor,
        sample_viewdirs: torch.Tensor,
        bb_min_dev: torch.Tensor,
        return_sigma: bool = False,
    ):
        n_rays, n_samples = sample_points.shape[:2]
        flat_points = sample_points.reshape(-1, 3)
        flat_viewdirs = sample_viewdirs.reshape(-1, 3)
        valid_mask = self._points_in_scan_bounds(flat_points)

        densities = torch.zeros((flat_points.shape[0], 1), dtype=flat_points.dtype, device=flat_points.device)
        log_sigmas = torch.zeros_like(densities) if return_sigma else None

        if torch.any(valid_mask):
            prediction = self._query_flat_samples(
                model,
                flat_points[valid_mask],
                flat_viewdirs[valid_mask],
                bb_min_dev,
                return_sigma=return_sigma,
            )
            if return_sigma:
                density_values, log_sigma_values = prediction
                densities[valid_mask] = density_values
                log_sigmas[valid_mask] = log_sigma_values
            else:
                densities[valid_mask] = prediction

        densities = densities.reshape(n_rays, n_samples, 1)
        valid_weights = valid_mask.reshape(n_rays, n_samples, 1).to(densities.dtype)
        if return_sigma:
            return densities, log_sigmas.reshape(n_rays, n_samples, 1), valid_weights
        return densities, valid_weights

    def _integrate_samples(
        self,
        samples: torch.Tensor,
        valid_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.n_samples == 1:
            return samples[:, 0, :]

        if self.integration == "sum":
            return torch.sum(samples * valid_weights, dim=1)

        if self.integration == "alpha":
            positive = F.softplus(samples.squeeze(-1)) * self.density_scale
            distances = self._sample_distances(samples.device, samples.dtype)
            alpha = 1.0 - torch.exp(-positive * distances[None, :])
            alpha = alpha * valid_weights.squeeze(-1)
            transmittance = torch.cumprod(
                torch.cat(
                    [
                        torch.ones((alpha.shape[0], 1), dtype=alpha.dtype, device=alpha.device),
                        1.0 - alpha + 1e-10,
                    ],
                    dim=1,
                ),
                dim=1,
            )[:, :-1]
            weights = (alpha * transmittance).unsqueeze(-1)
            return torch.sum(samples * weights, dim=1)

        weight_sum = torch.sum(valid_weights, dim=1).clamp_min(1.0)
        return torch.sum(samples * valid_weights, dim=1) / weight_sum

    def _integrate_log_sigma(
        self,
        log_sigma_samples: torch.Tensor,
        valid_weights: torch.Tensor,
    ) -> torch.Tensor:
        sigma = torch.exp(log_sigma_samples).clamp_min(1e-8)
        weight_sum = torch.sum(valid_weights, dim=1).clamp_min(1.0)
        sigma_mean = torch.sum(sigma * valid_weights, dim=1) / weight_sum
        return torch.log(sigma_mean.clamp_min(1e-8))

    def _render_rays(
        self,
        model,
        origins: torch.Tensor,
        viewdirs: torch.Tensor,
        bb_min_dev: torch.Tensor,
        return_sigma: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        origins = torch.reshape(origins, (-1, origins.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        densities = []
        log_sigmas = []

        for start in range(0, origins.shape[0], self.chunk):
            stop = min(start + self.chunk, origins.shape[0])
            sample_points, sample_viewdirs = self._build_ray_samples(origins[start:stop], viewdirs[start:stop])
            queried = self._query_ray_samples(
                model,
                sample_points,
                sample_viewdirs,
                bb_min_dev,
                return_sigma=return_sigma,
            )

            if return_sigma:
                density_samples, log_sigma_samples, valid_weights = queried
                densities.append(self._integrate_samples(density_samples, valid_weights))
                log_sigmas.append(self._integrate_log_sigma(log_sigma_samples, valid_weights))
                continue

            density_samples, valid_weights = queried
            densities.append(self._integrate_samples(density_samples, valid_weights))

        density = torch.cat(densities, dim=0).to(DEVICE)
        if return_sigma:
            return density, torch.cat(log_sigmas, dim=0).to(DEVICE)
        return density

    def render_slice_from_dataset_valid(self, model, slice_number, reshaped=False, jitter=False):
        dataset = self._require_dataset()
        points = dataset.get_slice_valid_points(slice_number)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self._render_rays(
            model,
            points,
            dataset.get_slice_valid_viewdirs(slice_number),
            dataset.point_min_dev,
        )
        return self._reshape_density(density, reshaped, (self.height_px, self.width_px))

    def render_slice_from_dataset(
        self,
        model,
        slice_number,
        reshaped=False,
        jitter=False,
        scalefactor=None,
        return_sigma: bool = False,
    ):
        dataset = self._require_dataset()
        points = dataset.get_slice_points(slice_number)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        prediction = self._render_rays(
            model,
            points,
            dataset.get_slice_viewdirs(slice_number),
            dataset.point_min_dev,
            return_sigma=return_sigma,
        )
        if return_sigma:
            density, log_sigma = prediction
            density = self._reshape_density(density, reshaped, (self.height_px, self.width_px))
            log_sigma = self._reshape_density(log_sigma, reshaped, (self.height_px, self.width_px))
            return density, log_sigma

        density = prediction
        reshaped_density = self._reshape_density(density, reshaped, (self.height_px, self.width_px))
        if not reshaped or scalefactor is None:
            return reshaped_density

        image = reshaped_density.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            image,
            size=(self.height_px * scalefactor, self.width_px * scalefactor),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).squeeze(0).to(DEVICE)

    def render_slice(self, model, pos, rot, reshaped=False, jitter=False):
        points, viewdirs = get_oriented_points_and_views(self.X, self.Y, pos, rot)
        points = torch.as_tensor(points, dtype=torch.float32, device=DEVICE)
        viewdirs = torch.as_tensor(viewdirs, dtype=torch.float32, device=DEVICE)

        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self._render_rays(model, points, viewdirs, self.bb_min_dev)
        return self._reshape_density(density, reshaped, (self.height_px, self.width_px))

    def render_slice_for_chosen_grid(
        self,
        model,
        X,
        Y,
        pos,
        rot,
        reshaped=False,
        jitter=False,
        grid_shape=None,
    ):
        points, viewdirs = get_oriented_points_and_views(X, Y, pos, rot)
        points = torch.as_tensor(points, dtype=torch.float32, device=DEVICE)
        viewdirs = torch.as_tensor(viewdirs, dtype=torch.float32, device=DEVICE)

        if grid_shape is None:
            grid_shape = (self.height_px, self.width_px)

        if jitter:
            points = self._apply_jitter(points, int(grid_shape[1]), int(grid_shape[0]))

        density = self._render_rays(model, points, viewdirs, self.bb_min_dev)
        return self._reshape_density(density, reshaped, grid_shape)

    def query_random_positions(
        self,
        model,
        indices,
        reshaped=False,
        jitter=False,
        return_sigma: bool = False,
    ):
        del reshaped

        dataset = self._require_dataset()
        points = dataset.get_indices_points(indices)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        return self._render_rays(
            model,
            points,
            dataset.get_indices_viewdirs(indices),
            dataset.point_min_dev,
            return_sigma=return_sigma,
        )


class SliceRenderRay(SliceRenderer):
    """Explicit class name for code that should not shadow ``SliceRenderer``."""
