from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """[...,6] 的前三项为平移生成元 mm，后三项为旋转向量 rad。"""
    v, w = twist[..., :3], twist[..., 3:]
    zero = torch.zeros_like(w[..., 0])
    wx, wy, wz = w.unbind(-1)
    skew = torch.stack((zero, -wz, wy, wz, zero, -wx, -wy, wx, zero), -1)
    upper = torch.cat((skew.reshape(*w.shape[:-1], 3, 3), v.unsqueeze(-1)), -1)
    algebra = torch.cat((upper, torch.zeros_like(upper[..., :1, :])), -2)
    # matrix_exp 在零旋转处也有正确有限梯度，避免 Rodrigues 的 0/0 分支。
    return torch.matrix_exp(algebra)


class EdgeField(nn.Module):
    def __init__(self, bounds, *, bands=10, width=128, layers=4, response_only=False,
                 plane_resolutions=(), plane_channels=4):
        super().__init__()
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        self.register_buffer("center", bounds.mean(0))
        self.register_buffer("extent", (bounds[1] - bounds[0]).clamp_min(1))
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands) * math.pi)
        self.config = dict(bands=bands, width=width, layers=layers, response_only=response_only,
                           plane_resolutions=list(plane_resolutions), plane_channels=plane_channels)
        self.response_only = response_only
        # 每层 XY/XZ/YZ 三张可学习特征平面；坐标仍是统一的毫米世界坐标。
        self.planes = nn.ParameterList([nn.Parameter(.01*torch.randn(3, plane_channels, r, r))
                                        for r in plane_resolutions])
        self.progress = 1.0
        modules = []
        for index in range(layers):
            inputs = 3 + 6*bands + 3*plane_channels*len(plane_resolutions)
            modules.extend((nn.Linear(inputs if index == 0 else width, width), nn.SiLU()))
        self.trunk = nn.Sequential(*modules)
        if not response_only:
            self.gray = nn.Linear(width, 1)
            nn.init.constant_(self.gray.bias, -1.5)
        self.edge = nn.Linear(width, 1)
        nn.init.constant_(self.edge.bias, -2.0)

    def forward(self, xyz, progress=None):
        progress = self.progress if progress is None else progress
        x = 2 * (xyz - self.center) / self.extent
        alpha = 2 + (self.config["bands"] - 2) * min(1.0, max(0.0, progress))
        weight = (1 - torch.cos(math.pi * (alpha - torch.arange(
            self.config["bands"], device=x.device)).clamp(0, 1))) / 2
        phase = x[..., :, None] * self.frequencies
        encoded = torch.cat((x, (phase.sin() * weight).flatten(-2),
                             (phase.cos() * weight).flatten(-2)), -1)
        if self.planes:
            flat = x.reshape(-1, 3)
            grid = torch.stack((flat[:, [0, 1]], flat[:, [0, 2]], flat[:, [1, 2]]))[:, :, None]
            features = []
            for level, plane in enumerate(self.planes):
                values = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
                # 高频平面逐步开放；位姿阶段保留双线性采样的坐标导数。
                active = min(1., max(0., 2 + (len(self.planes)-2)*progress - level))
                features.append(values[..., 0].permute(2, 0, 1).reshape(*x.shape[:-1], -1)*active)
            encoded = torch.cat((encoded, *features), -1)
        features = self.trunk(encoded)
        if self.response_only:
            return self.edge(features).sigmoid()
        return torch.cat((self.gray(features).sigmoid(), self.edge(features).sigmoid()), -1)


class PoseRefiner(nn.Module):
    def __init__(self, initial, train_indices, center, *, translation_mm=2.0, rotation_deg=2.0):
        super().__init__()
        initial = torch.as_tensor(initial, dtype=torch.float32)
        self.register_buffer("initial", initial)
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("limits", torch.tensor([translation_mm] * 3 +
                                                    [math.radians(rotation_deg)] * 3))
        active = torch.zeros(len(initial), 1)
        active[train_indices] = 1
        active[train_indices[0]] = 0
        self.register_buffer("active", active)
        self.raw = nn.Parameter(torch.zeros(len(initial), 6))

    def twists(self):
        return self.raw.tanh() * self.limits * self.active

    def matrices(self):
        # 在固定场景中心附近做左乘修正，避免世界原点很远时旋转引入巨大平移。
        centered = self.initial.clone()
        centered[:, :3, 3] -= self.center
        corrected = se3_exp(self.twists()) @ centered
        result = torch.cat((torch.cat((corrected[:, :3, :3],
                            (corrected[:, :3, 3] + self.center).unsqueeze(-1)), -1),
                            corrected[:, 3:4]), -2)
        return torch.where(self.active[:, :, None].bool(), result, self.initial)

    def forward(self, frame_indices, local_points):
        matrices = self.matrices()[frame_indices]
        return torch.einsum("bij,bpj->bpi", matrices[:, :3, :3], local_points) + matrices[:, None, :3, 3]

    def prior(self, train_indices, frame_ids):
        # 平滑的是修正而不是探头实际运动；跨缺失帧按原始帧间距缩放。
        normalized = self.twists() / self.limits
        selected = normalized[train_indices]
        gaps = (frame_ids[train_indices][1:] - frame_ids[train_indices][:-1]).clamp_min(1)
        smooth = ((selected[1:] - selected[:-1]) / gaps[:, None]).square().mean()
        return normalized.square().mean() + smooth


class InPlanePoseRefiner(nn.Module):
    """在图像中心修正面内平移和旋转；每张物理平面的位置、法向保持不变。"""

    def __init__(self, initial, active, local_center, *, translation_mm=1., rotation_deg=1.):
        super().__init__()
        self.register_buffer("initial", initial.detach().clone())
        self.register_buffer("active", active.to(initial.device).bool())
        self.register_buffer("local_center", local_center.detach().clone())
        self.register_buffer("limits", initial.new_tensor([translation_mm, translation_mm,
                                                           math.radians(rotation_deg)]))
        self.raw = nn.Parameter(initial.new_zeros((len(initial), 3)))

    def corrections(self):
        # 平移限制作用于二维向量范数，而非分别限制后得到 sqrt(2) 倍上限。
        xy = self.raw[:, :2] / (1 + self.raw[:, :2].square().sum(-1, keepdim=True)).sqrt()
        return torch.cat((xy, self.raw[:, 2:].tanh()), -1) * self.limits * self.active[:, None]

    def matrices(self):
        delta = self.corrections()
        angle = delta[:, 2]
        c, s, z = angle.cos(), angle.sin(), torch.zeros_like(angle)
        rotation = torch.stack((c, -s, z, s, c, z, z, z, torch.ones_like(z)), -1).reshape(-1, 3, 3)
        translation = self.local_center - rotation @ self.local_center + torch.cat((delta[:, :2], z[:, None]), -1)
        result = self.initial.clone()
        result[:, :3, :3] = self.initial[:, :3, :3] @ rotation
        result[:, :3, 3] = self.initial[:, :3, 3] + (self.initial[:, :3, :3] @ translation[..., None])[..., 0]
        return torch.where(self.active[:, None, None], result, self.initial)

    def prior(self, train_indices, frame_ids):
        correction = self.corrections() / self.limits
        selected = correction[train_indices]
        gaps = (frame_ids[train_indices][1:] - frame_ids[train_indices][:-1]).clamp_min(1)
        smooth = ((selected[1:] - selected[:-1]) / gaps[:, None]).square().mean()
        return correction.square().mean(), smooth


def load_field(path: str | Path, device="cpu"):
    """推理只依赖 checkpoint，不读取 teacher 或原始数据。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["schema"] not in ("nlstv_edge_field_v1", "nlstv_response_field_v1"):
        raise ValueError("不是 NLSTV edge field checkpoint")
    model = EdgeField(checkpoint["bounds"], **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.progress = checkpoint["frequency_progress"]
    return model.eval(), checkpoint
