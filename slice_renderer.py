import numpy as np
import torch
from utils import get_base_points, get_oriented_points_and_views
import torch.nn.functional as F
from torchvision.transforms import Resize

import dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SliceRenderer:
    def __init__(self, dataset=None, px_width=0, px_height=0, width=0, height=0, point_min=None, point_max=None, 
                 decimation=1):
        self.dataset = None
        self.X, self.Y = ([], [])
        if dataset:
            self.dataset = dataset
            self._init_slice_renderer(dataset.px_width,
                                      dataset.px_height,
                                      dataset.width,
                                      dataset.height,
                                      dataset.point_min,
                                      dataset.point_max,
                                      
                                      decimation)
        elif px_width and px_height and width and height and point_min is not None and point_max is not None:
            self._init_slice_renderer(px_width,
                                      px_height,
                                      width,
                                      height,
                                      point_min,
                                      point_max,
                                      decimation)
        else:
            print("Parameter error when creating SliceRenderer")
            exit(-1)

    def _init_slice_renderer(self, px_width, px_height, width, height, point_min, point_max, decimation):
        self.width_px = int(px_width)
        self.height_px = int(px_height)

        self.width = width
        self.height = height

        self.bb_min = point_min
        self.bb_min_dev = torch.FloatTensor(point_min).to(device)
        self.bb_max = point_max
        # self.max_coord = torch.max(torch.FloatTensor(
        #     (self.bb_max[0] - self.bb_min[0], self.bb_max[1] - self.bb_min[1], self.bb_max[2] - self.bb_min[2]))).to(
        #     device)
        self.max_coord = torch.FloatTensor(
            (self.bb_max[0] - self.bb_min[0], self.bb_max[1] - self.bb_min[1], self.bb_max[2] - self.bb_min[2])).to(
            device)

        self.X, self.Y = get_base_points(width, height, self.width_px, self.height_px, offset_x_mm=point_min[0], offset_y_mm=point_min[1])

    def _points_in_scan_bounds(self, points):
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

    def _query_with_scan_mask(self, model, points, viewdirs, bb_min_dev):
        points = torch.reshape(points, (-1, points.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))
        valid_mask = self._points_in_scan_bounds(points)
        if torch.all(valid_mask):
            query_points = points
            query_viewdirs = viewdirs
            if model.encoding_type != "HASH" or not model.use_encoding:
                query_points = torch.add(
                    torch.multiply(torch.divide(torch.add(query_points, -bb_min_dev), self.max_coord), 2), -1
                )
            return model.query(query_points, query_viewdirs).to(device)

        dens = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
        if torch.any(valid_mask):
            query_points = points[valid_mask]
            query_viewdirs = viewdirs[valid_mask]
            if model.encoding_type != "HASH" or not model.use_encoding:
                query_points = torch.add(
                    torch.multiply(torch.divide(torch.add(query_points, -bb_min_dev), self.max_coord), 2), -1
                )
            dens[valid_mask] = model.query(query_points, query_viewdirs)

        return dens.to(device)

    def _query_points(self, model, points, viewdirs, bb_min_dev):
        points = torch.reshape(points, (-1, points.shape[-1]))
        viewdirs = torch.reshape(viewdirs, (-1, viewdirs.shape[-1]))

        if model.encoding_type != "HASH" or not model.use_encoding:
            points = torch.add(
                torch.multiply(torch.divide(torch.add(points, -bb_min_dev), self.max_coord), 2), -1
            )

        return model.query(points, viewdirs).to(device)

    def render_slice_from_dataset_valid(self, model, slice_number, reshaped=False, jitter=False):
        if not self.dataset:
            print("Cannot render slice from dataset if slice_renderer was not initialized with a dataset")
            exit(-1)

        # points = self.dataset.slices_valid[slice_number].points
        points = self.dataset.get_slice_valid_points(slice_number)

        if jitter:
            w = self.width / (3 * self.width_px)
            h = self.height / (3 * self.height_px)
            thickness = min(w, h)
            jit = torch.stack((w * torch.randn((points.shape[0], 1)), thickness * torch.randn((points.shape[0], 1)),
                               h * torch.randn((points.shape[0], 1)))).to(device)
            points = torch.add(points, jit)

        # print(points)
        # print(self.dataset.point_min)
        # print(self.max_coord.cpu())

        dens = self._query_points(
            model,
            points,
            self.dataset.get_slice_valid_viewdirs(slice_number),
            self.dataset.point_min_dev,
        )

        if  not reshaped :
            return dens.to(device)
        return torch.reshape(dens, (self.height_px, self.width_px)).to(device)
    def render_slice_from_dataset(self, model, slice_number, reshaped=False, jitter=False, scalefactor=None):
        if not self.dataset:
            print("Cannot render slice from dataset if slice_renderer was not initialized with a dataset")
            exit(-1)

        # points = self.dataset.slices[slice_number].points
        points = self.dataset.get_slice_points(slice_number)

        if jitter :
            w = self.width / (3 * self.width_px)
            h = self.height / (3 * self.height_px)
            thickness = min(w,h)
            jit = torch.stack( (w*torch.randn((points.shape[0],1)),thickness*torch.randn((points.shape[0],1)),h*torch.randn((points.shape[0],1)) ) ).to(device)
            points = torch.add(points,jit)

        # print(points)
        # print(self.dataset.point_min)
        # print(self.max_coord.cpu())

        dens = self._query_points(
            model,
            points,
            self.dataset.get_slice_viewdirs(slice_number),
            self.dataset.point_min_dev,
        )
        if not reshaped :
            return dens.to(device)
        if scalefactor is None:
            return torch.reshape(dens, (self.height_px, self.width_px)).to(device)
        else:
            resize = Resize((self.height_px * scalefactor, self.width_px * scalefactor))
            return torch.squeeze(resize(torch.unsqueeze(torch.reshape(dens, (self.height_px, self.width_px)),dim=0))).to(device)

    def render_slice(self, model, pos, rot, reshaped=False, jitter=False):
        points, viewdirs = get_oriented_points_and_views(self.X, self.Y, pos, rot)

        points = torch.from_numpy(points.astype(dtype=np.float32)).to(device)
        viewdirs = torch.from_numpy(viewdirs.astype(dtype=np.float32)).to(device)

        if jitter :
            w = self.width / (3 * self.width_px)
            h = self.height / (3 * self.height_px)
            thickness = min(w,h)
            jit = torch.stack( (w*torch.randn((points.shape[0],1)),thickness*torch.randn((points.shape[0],1)),h*torch.randn((points.shape[0],1)) ) ).to(device)
            points = torch.add(points,jit)

        raw = self._query_points(model, points, viewdirs, self.bb_min_dev)

        # dens = F.relu(raw)
        dens = raw
        if not reshaped :
            return dens.to(device)
        return torch.reshape(dens, (self.height_px, self.width_px)).to(device)

    def render_slice_for_chosen_grid(self, model, X, Y, pos, rot, reshaped=False, jitter=False, grid_shape=None):
        points, viewdirs = get_oriented_points_and_views(X, Y, pos, rot)

        points = torch.from_numpy(points.astype(dtype=np.float32)).to(device)
        viewdirs = torch.from_numpy(viewdirs.astype(dtype=np.float32)).to(device)

        if jitter :
            if grid_shape is None:
                grid_height, grid_width = self.height_px, self.width_px
            else:
                grid_height, grid_width = int(grid_shape[0]), int(grid_shape[1])
            w = self.width / (3 * grid_width)
            h = self.height / (3 * grid_height)
            thickness = min(w,h)
            jit = torch.stack( (w*torch.randn((points.shape[0],1)),thickness*torch.randn((points.shape[0],1)),h*torch.randn((points.shape[0],1)) ) ).to(device)
            points = torch.add(points,jit)

        raw = self._query_with_scan_mask(model, points, viewdirs, self.bb_min_dev)

        # dens = F.relu(raw)
        dens = raw
        if not reshaped :
            return dens.to(device)
        if grid_shape is None:
            grid_shape = (self.height_px, self.width_px)
        return torch.reshape(dens, (int(grid_shape[0]), int(grid_shape[1]))).to(device)


    def query_random_positions(self, model, indices, reshaped=False, jitter=False):
        if not self.dataset:
            print("Cannot query random position if slice_renderer was not initialized with a dataset")
            exit(-1)

        points = self.dataset.get_indices_points(indices)
        if model.encoding_type != "HASH" or not model.use_encoding:
            points = torch.add(
                torch.multiply(torch.divide(torch.add(points, -self.dataset.point_min_dev), self.max_coord), 2), -1)
        viewdirs = points #Todo viewdirs not implemented

        raw = model.query(points, viewdirs)

        return raw
