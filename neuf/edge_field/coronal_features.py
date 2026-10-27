"""原生毫米坐标上的 coronal 特征采样与固定 sagittal 平面重建。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CoronalFeatures(nn.Module):
    """共享 2D CNN 编码全部帧；沿法向距离融合查询点的局部特征。"""

    def __init__(self, *, frames, height=96, width=128, channels=32, sigma_mm=.75):
        super().__init__()
        self.sigma_mm = float(sigma_mm)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, channels, 3, stride=2, padding=1), nn.SiLU(),
        )
        # 输入、标定和当前位姿随模型保存，独立推理不需要重新打开原始数据。
        self.register_buffer("images", torch.zeros(frames, 1, height, width))
        self.register_buffer("mask", torch.zeros(1, 1, height, width))
        self.register_buffer("matrices", torch.eye(4).repeat(frames, 1, 1))
        self.register_buffer("local_origin", torch.zeros(2))
        self.register_buffer("local_extent", torch.ones(2))
        self.feature_maps = None
        self.query_matrices = None

    @torch.no_grad()
    def initialize(self, data):
        size = self.images.shape[-2:]
        self.images.copy_(F.interpolate(data.images[:, None] * data.mask, size=size,
                                       mode="bilinear", align_corners=True))
        self.mask.copy_(F.interpolate(data.mask[None, None].float(), size=size,
                                     mode="bilinear", align_corners=True))
        self.local_origin.copy_(data.local[0, 0, :2])
        self.local_extent.copy_(data.local[-1, -1, :2] - data.local[0, 0, :2])
        self.matrices.copy_(data.initial)

    def prepare(self, matrices):
        # 每个 optimizer 阶段重新编码；禁止跨更新复用旧的计算图或旧权重特征。
        self.feature_maps = self.encoder(self.images)
        self.query_matrices = matrices
        with torch.no_grad():
            self.matrices.copy_(matrices.detach())

    def project(self, xyz, matrices=None):
        matrices = self.matrices if matrices is None else matrices
        local = torch.einsum("npi,nij->npj", xyz[None] - matrices[:, None, :3, 3],
                             matrices[:, :3, :3])
        uv = 2 * (local[..., :2] - self.local_origin) / self.local_extent - 1
        return local, uv.flip(-1)

    def forward(self, xyz):
        if self.feature_maps is None:
            self.prepare(self.matrices)
        shape = xyz.shape[:-1]
        outputs = []
        for points in xyz.reshape(-1, 3).split(2048):
            local, grid = self.project(points, self.query_matrices)
            # 两层 stride=2 的卷积中心落在输入 0,4,8,... 像素；不能把末中心
            # 错当成输入末像素，否则低分辨率特征会产生系统性坐标偏移。
            scale = grid.new_tensor([(self.images.shape[-1] - 1) / (4 * (self.feature_maps.shape[-1] - 1)),
                                     (self.images.shape[-2] - 1) / (4 * (self.feature_maps.shape[-2] - 1))])
            feature_grid = (grid + 1) * scale - 1
            sampled = F.grid_sample(self.feature_maps, feature_grid[:, :, None],
                                    align_corners=True, padding_mode="border")[..., 0]
            valid = F.grid_sample(self.mask.expand(len(local), -1, -1, -1),
                                  grid[:, :, None], align_corners=True)[..., 0][:, 0]
            weight = torch.exp(-.5 * (local[..., 2] / self.sigma_mm).square()) * (valid > .999)
            result = (sampled * weight[:, None]).sum(0) / weight.sum(0).clamp_min(1e-8)
            outputs.append(result.T)
        return torch.cat(outputs).reshape(*shape, -1)


def coronal_projection(xyz, matrices, local_grid, mask, *, neighbors=2, sigma_mm=.75):
    """选择覆盖查询点的最近物理平面；离面距离仍参与可微权重与投影。"""
    flat = xyz.reshape(-1, 3)
    local = torch.einsum("npi,nij->npj", flat[None] - matrices[:, None, :3, 3],
                         matrices[:, :3, :3])
    uv = 2 * (local[..., :2] - local_grid[0, 0, :2]) / (
        local_grid[-1, -1, :2] - local_grid[0, 0, :2]) - 1
    sampled_mask = F.grid_sample(mask[None, None].float().expand(len(matrices), -1, -1, -1),
                                 uv.flip(-1)[:, :, None], align_corners=True)[..., 0][:, 0]
    valid = sampled_mask > .999
    distance = local[..., 2].abs()
    ranked = distance.detach().masked_fill(~valid, float("inf"))
    chosen = ranked.topk(min(neighbors, len(matrices)), dim=0, largest=False).indices
    point_ids = torch.arange(len(flat), device=flat.device)[None].expand_as(chosen)
    selected_distance = local[chosen, point_ids, 2]
    normals = matrices[chosen, :3, 2]
    projected = flat[None] - selected_distance[..., None] * normals
    selected_valid = valid[chosen, point_ids] & (selected_distance.abs() <= 3 * sigma_mm)
    weight = torch.exp(-.5 * (selected_distance / sigma_mm).square()) * selected_valid
    support = weight.sum(0) > 1e-8
    weight = weight / weight.sum(0).clamp_min(1e-8)
    return projected, weight, support.reshape(xyz.shape[:-1])


def render_coronal_field(model, xyz, matrices, local_grid, mask, progress=1.):
    """固定 sag 世界点投影到 coronal 平面后查询 INR，保留两组共同的位姿梯度。"""
    predictions, supports = [], []
    for points in xyz.reshape(-1, 3).split(2048):
        projected, weight, support = coronal_projection(points, matrices, local_grid, mask)
        values = model(projected, progress)[..., 0]
        predictions.append((values * weight).sum(0))
        supports.append(support)
    return (torch.cat(predictions).reshape(xyz.shape[:-1]),
            torch.cat(supports).reshape(xyz.shape[:-1]))


def pose_update_rate(progress):
    """每次 INR 更新对应的几何更新率，20%/40%/70% 为精确转折点。"""
    if progress <= .2 or progress >= .7:
        return 0.
    return (progress - .2) / .2 if progress <= .4 else (.7 - progress) / .3
