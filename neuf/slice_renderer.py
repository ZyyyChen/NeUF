from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from neuf.slice_renderer_base import DEVICE, BaseSliceRenderer
from neuf.utils import get_oriented_points_and_views


class SliceRenderer(BaseSliceRenderer):
    """Point-sampling slice renderer used by the current NeUF training loop."""

    def query_points(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        *,
        return_sigma: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Query caller-provided world points, including differentiable poses."""
        dataset = self._require_dataset()
        return self._query_points(
            model,
            points,
            viewdirs,
            dataset.point_min_dev,
            return_sigma=return_sigma,
        )

    def _query_points(
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
            return density.to(DEVICE), log_sigma.to(DEVICE)
        return prediction.to(DEVICE)

    def _query_with_scan_mask(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        bb_min_dev: torch.Tensor,
    ) -> torch.Tensor:
        points = torch.reshape(points, (-1, points.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        valid_mask = self._points_in_scan_bounds(points)

        if torch.all(valid_mask):
            return self._query_points(model, points, viewdirs, bb_min_dev)

        densities = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
        if torch.any(valid_mask):
            densities[valid_mask] = self._query_points(
                model,
                points[valid_mask],
                viewdirs[valid_mask],
                bb_min_dev,
            )

        return densities.to(DEVICE)

    def query_points_masked(
        self,
        model,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_sigma: bool = False,
    ):
        """Query only valid fan pixels and scatter results into the full frame."""
        dataset = self._require_dataset()
        flat_points = torch.reshape(points, (-1, points.shape[-1]))
        flat_viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        flat_mask = torch.as_tensor(
            valid_mask,
            dtype=torch.bool,
            device=flat_points.device,
        ).reshape(-1)
        if flat_mask.numel() != flat_points.shape[0]:
            raise ValueError(
                "valid_mask must contain one value per query point, got "
                f"{flat_mask.numel()} and {flat_points.shape[0]}"
            )

        prediction = self._query_points(
            model,
            flat_points[flat_mask],
            flat_viewdirs[flat_mask],
            dataset.point_min_dev,
            return_sigma=return_sigma,
        )
        if return_sigma:
            density_valid, log_sigma_valid = prediction
            density = torch.zeros(
                (flat_points.shape[0], density_valid.shape[-1]),
                dtype=density_valid.dtype,
                device=density_valid.device,
            )
            log_sigma = torch.zeros_like(density)
            density[flat_mask] = density_valid
            log_sigma[flat_mask] = log_sigma_valid
            return density, log_sigma

        density_valid = prediction
        density = torch.zeros(
            (flat_points.shape[0], density_valid.shape[-1]),
            dtype=density_valid.dtype,
            device=density_valid.device,
        )
        density[flat_mask] = density_valid
        return density

    def render_slice_from_dataset_valid(self, model, slice_number, reshaped=False, jitter=False):
        dataset = self._require_dataset()
        points = dataset.get_slice_valid_points(slice_number)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self.query_points_masked(
            model,
            points,
            dataset.get_slice_valid_viewdirs(slice_number),
            dataset.get_sector_mask(flatten=True, device=points.device),
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

        prediction = self.query_points_masked(
            model,
            points,
            dataset.get_slice_viewdirs(slice_number),
            dataset.get_sector_mask(flatten=True, device=points.device),
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
        points = torch.from_numpy(points.astype(np.float32)).to(DEVICE)
        viewdirs = torch.from_numpy(viewdirs.astype(np.float32)).to(DEVICE)

        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        valid_mask = self._sector_mask_for_grid(
            self.height_px,
            self.width_px,
            device=points.device,
        )
        density = self.query_points_masked(
            model,
            points,
            viewdirs,
            valid_mask,
        )
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
        points = torch.from_numpy(points.astype(np.float32)).to(DEVICE)
        viewdirs = torch.from_numpy(viewdirs.astype(np.float32)).to(DEVICE)

        if grid_shape is None:
            grid_shape = (self.height_px, self.width_px)

        if jitter:
            points = self._apply_jitter(points, int(grid_shape[1]), int(grid_shape[0]))

        density = self._query_with_scan_mask(model, points, viewdirs, self.bb_min_dev)
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
        del jitter

        dataset = self._require_dataset()
        return self._query_points(
            model,
            dataset.get_indices_points(indices),
            dataset.get_indices_viewdirs(indices),
            dataset.point_min_dev,
            return_sigma=return_sigma,
        )
