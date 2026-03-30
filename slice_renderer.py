from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import Resize

from utils import get_base_points, get_oriented_points_and_views

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_BBOX_EXTENT = 1e-6


class SliceRenderer:
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
    ):
        self.dataset = None
        self.X = np.array([], dtype=np.float32)
        self.Y = np.array([], dtype=np.float32)

        if dataset is not None:
            self.dataset = dataset
            self._init_slice_renderer(
                dataset.px_width,
                dataset.px_height,
                dataset.width,
                dataset.height,
                dataset.point_min,
                dataset.point_max,
                decimation,
            )
            return

        if px_width and px_height and width and height and point_min is not None and point_max is not None:
            self._init_slice_renderer(
                px_width,
                px_height,
                width,
                height,
                point_min,
                point_max,
                decimation,
            )
            return

        raise ValueError("Invalid SliceRenderer initialization parameters")

    def _init_slice_renderer(self, px_width, px_height, width, height, point_min, point_max, decimation):
        self.width_px = max(1, int(px_width) // max(1, int(decimation)))
        self.height_px = max(1, int(px_height) // max(1, int(decimation)))
        self.width = float(width)
        self.height = float(height)

        self.bb_min = np.asarray(point_min, dtype=np.float32)
        self.bb_max = np.asarray(point_max, dtype=np.float32)
        self.bb_min_dev = torch.as_tensor(self.bb_min, dtype=torch.float32, device=DEVICE)
        bbox_extent = torch.as_tensor(self.bb_max - self.bb_min, dtype=torch.float32, device=DEVICE)
        self.max_coord = torch.clamp(bbox_extent, min=MIN_BBOX_EXTENT)

        self.X, self.Y = get_base_points(
            width,
            height,
            self.width_px,
            self.height_px,
            offset_x_mm=point_min[0],
            offset_y_mm=point_min[1],
        )

    def _require_dataset(self):
        if self.dataset is None:
            raise ValueError("SliceRenderer requires a dataset for this operation")
        return self.dataset

    def _points_in_scan_bounds(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.reshape(points, (-1, points.shape[-1]))
        mask = (
            (points[:, 0] >= self.bb_min[0]) & (points[:, 0] <= self.bb_max[0]) &
            (points[:, 1] >= self.bb_min[1]) & (points[:, 1] <= self.bb_max[1]) &
            (points[:, 2] >= self.bb_min[2]) & (points[:, 2] <= self.bb_max[2])
        )

        if self.dataset and self.dataset.front_plane_point is not None and self.dataset.back_plane_point is not None:
            front_point = torch.as_tensor(self.dataset.front_plane_point, dtype=points.dtype, device=points.device)
            back_point = torch.as_tensor(self.dataset.back_plane_point, dtype=points.dtype, device=points.device)
            scan_axis = torch.as_tensor(self.dataset.scan_axis, dtype=points.dtype, device=points.device)

            front_proj = torch.sum((points - front_point) * scan_axis, dim=1)
            back_proj = torch.sum((points - back_point) * scan_axis, dim=1)
            mask = mask & (front_proj >= 0) & (back_proj <= 0)

        return mask

    def _normalize_points_if_needed(self, model, points: torch.Tensor, bb_min_dev: torch.Tensor) -> torch.Tensor:
        if model.encoding_type == "HASH" and model.use_encoding:
            return points

        return torch.add(
            torch.multiply(torch.divide(torch.add(points, -bb_min_dev), self.max_coord), 2),
            -1,
        )

    def _query_points(self, model, points: torch.Tensor, viewdirs: torch.Tensor, bb_min_dev: torch.Tensor) -> torch.Tensor:
        points = torch.reshape(points, (-1, points.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        query_points = self._normalize_points_if_needed(model, points, bb_min_dev)
        return model.query(query_points, viewdirs).to(DEVICE)

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

    def _apply_jitter(self, points: torch.Tensor, grid_width: int, grid_height: int) -> torch.Tensor:
        jitter_shape = points.shape[:-1]
        pixel_width = self.width / (3 * max(1, grid_width))
        pixel_height = self.height / (3 * max(1, grid_height))
        thickness = min(pixel_width, pixel_height)
        jitter = torch.stack(
            (
                pixel_width * torch.randn(jitter_shape, device=points.device),
                thickness * torch.randn(jitter_shape, device=points.device),
                pixel_height * torch.randn(jitter_shape, device=points.device),
            ),
            dim=-1,
        )
        return points + jitter

    def _reshape_density(self, density: torch.Tensor, reshaped: bool, grid_shape) -> torch.Tensor:
        if not reshaped:
            return density.to(DEVICE)
        return torch.reshape(density, (int(grid_shape[0]), int(grid_shape[1]))).to(DEVICE)

    def render_slice_from_dataset_valid(self, model, slice_number, reshaped=False, jitter=False):
        dataset = self._require_dataset()
        points = dataset.get_slice_valid_points(slice_number)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self._query_points(
            model,
            points,
            dataset.get_slice_valid_viewdirs(slice_number),
            dataset.point_min_dev,
        )
        return self._reshape_density(density, reshaped, (self.height_px, self.width_px))

    def render_slice_from_dataset(self, model, slice_number, reshaped=False, jitter=False, scalefactor=None):
        dataset = self._require_dataset()
        points = dataset.get_slice_points(slice_number)
        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self._query_points(
            model,
            points,
            dataset.get_slice_viewdirs(slice_number),
            dataset.point_min_dev,
        )
        reshaped_density = self._reshape_density(density, reshaped, (self.height_px, self.width_px))
        if not reshaped or scalefactor is None:
            return reshaped_density

        resize = Resize((self.height_px * scalefactor, self.width_px * scalefactor))
        resized = resize(torch.unsqueeze(reshaped_density, dim=0))
        return torch.squeeze(resized).to(DEVICE)

    def render_slice(self, model, pos, rot, reshaped=False, jitter=False):
        points, viewdirs = get_oriented_points_and_views(self.X, self.Y, pos, rot)
        points = torch.from_numpy(points.astype(np.float32)).to(DEVICE)
        viewdirs = torch.from_numpy(viewdirs.astype(np.float32)).to(DEVICE)

        if jitter:
            points = self._apply_jitter(points, self.width_px, self.height_px)

        density = self._query_points(model, points, viewdirs, self.bb_min_dev)
        return self._reshape_density(density, reshaped, (self.height_px, self.width_px))

    def render_slice_for_chosen_grid(self, model, X, Y, pos, rot, reshaped=False, jitter=False, grid_shape=None):
        points, viewdirs = get_oriented_points_and_views(X, Y, pos, rot)
        points = torch.from_numpy(points.astype(np.float32)).to(DEVICE)
        viewdirs = torch.from_numpy(viewdirs.astype(np.float32)).to(DEVICE)

        if grid_shape is None:
            grid_shape = (self.height_px, self.width_px)

        if jitter:
            points = self._apply_jitter(points, int(grid_shape[1]), int(grid_shape[0]))

        density = self._query_with_scan_mask(model, points, viewdirs, self.bb_min_dev)
        return self._reshape_density(density, reshaped, grid_shape)

    def query_random_positions(self, model, indices, reshaped=False, jitter=False):
        del reshaped
        del jitter

        dataset = self._require_dataset()
        points = dataset.get_indices_points(indices)
        viewdirs = dataset.get_indices_viewdirs(indices)
        return self._query_points(model, points, viewdirs, dataset.point_min_dev)
