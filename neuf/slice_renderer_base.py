from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from neuf.utils import get_base_points

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_BBOX_EXTENT = 1e-6


class BaseSliceRenderer:
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
        prefer_dataset_offsets=False,
    ):
        self.dataset = None
        self.X = np.array([], dtype=np.float32)
        self.Y = np.array([], dtype=np.float32)
        self.prefer_dataset_offsets = bool(prefer_dataset_offsets)

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

        if self.prefer_dataset_offsets:
            offset_x_mm = getattr(self.dataset, "roi_offset_x_mm", self.bb_min[0])
            offset_y_mm = getattr(self.dataset, "roi_offset_y_mm", self.bb_min[1])
        else:
            offset_x_mm = self.bb_min[0]
            offset_y_mm = self.bb_min[1]

        self.X, self.Y = get_base_points(
            width,
            height,
            self.width_px,
            self.height_px,
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
        )

    def _require_dataset(self):
        if self.dataset is None:
            raise ValueError("SliceRenderer requires a dataset for this operation")
        return self.dataset

    def _sector_mask_for_grid(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        dataset = self._require_dataset()
        mask = dataset.get_sector_mask(device=device)
        if tuple(mask.shape) == (int(height), int(width)):
            return mask
        resized = F.interpolate(
            mask.float()[None, None],
            size=(int(height), int(width)),
            mode="nearest",
        )
        return resized[0, 0] > 0.5

    def _points_in_scan_bounds(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.reshape(points, (-1, points.shape[-1]))
        mask = (
            (points[:, 0] >= self.bb_min[0])
            & (points[:, 0] <= self.bb_max[0])
            & (points[:, 1] >= self.bb_min[1])
            & (points[:, 1] <= self.bb_max[1])
            & (points[:, 2] >= self.bb_min[2])
            & (points[:, 2] <= self.bb_max[2])
        )

        if (
            self.dataset
            and self.dataset.front_plane_point is not None
            and self.dataset.back_plane_point is not None
        ):
            front_point = torch.as_tensor(
                self.dataset.front_plane_point,
                dtype=points.dtype,
                device=points.device,
            )
            back_point = torch.as_tensor(
                self.dataset.back_plane_point,
                dtype=points.dtype,
                device=points.device,
            )
            scan_axis = torch.as_tensor(
                self.dataset.scan_axis,
                dtype=points.dtype,
                device=points.device,
            )

            front_proj = torch.sum((points - front_point) * scan_axis, dim=1)
            back_proj = torch.sum((points - back_point) * scan_axis, dim=1)
            mask = mask & (front_proj >= 0) & (back_proj <= 0)

        return mask

    def _normalize_points_if_needed(
        self,
        model,
        points: torch.Tensor,
        bb_min_dev: torch.Tensor,
    ) -> torch.Tensor:
        if model.encoding_type in {"HASH", "DUAL_HASH", "KRONECKER"} and model.use_encoding:
            return points

        bb_min_dev = bb_min_dev.to(points.device)
        max_coord = self.max_coord.to(points.device)
        return torch.add(
            torch.multiply(torch.divide(torch.add(points, -bb_min_dev), max_coord), 2),
            -1,
        )

    def _apply_jitter(
        self,
        points: torch.Tensor,
        grid_width: int,
        grid_height: int,
    ) -> torch.Tensor:
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

    @staticmethod
    def _reshape_density(
        density: torch.Tensor,
        reshaped: bool,
        grid_shape,
    ) -> torch.Tensor:
        if not reshaped:
            return density.to(DEVICE)
        return torch.reshape(density, (int(grid_shape[0]), int(grid_shape[1]))).to(DEVICE)
