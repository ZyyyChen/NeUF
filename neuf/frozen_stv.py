"""冻结二维 Neural STV，为三维分量场提供观测切片上的 S/B/R 监督。

这里采用 standalone 定义 S=A-B、R=I-A；R 包含散斑和其他残差，不能称为
物理上独立的 speckle。Teacher 仅属于训练辅助对象，不注册到 NeUF 模型中。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class FrozenSTVTeacher:
    """复用独立 STV 包推理；仅支持当前像素域 geometry=none checkpoint。"""

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        tile_size: int = 128,
    ) -> None:
        if isinstance(tile_size, bool) or not isinstance(tile_size, Integral) or tile_size <= 0:
            raise ValueError("tile_size 必须为正整数")
        checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
        if not checkpoint_path.is_file():
            raise ValueError("Neural STV checkpoint 必须为普通文件")
        try:
            from stv.cli.infer import context_tiled_prediction, load_student_checkpoint
        except ImportError as exc:
            raise ImportError(
                "冻结 Neural STV 监督需要独立的 neural-stv 包；"
                "请在训练环境安装已有 NLSTV/Code/STV 项目。"
            ) from exc

        model, payload, model_config = load_student_checkpoint(checkpoint_path, torch.device(device))
        if payload.get("geometry_mode", "none") != "none":
            raise ValueError("当前冻结监督仅接受 geometry_mode='none'，不推测毫米或 beam 几何")
        stride = int(model.stv.grid_stride)
        if tile_size % stride:
            raise ValueError(f"tile_size 必须可被 STV grid_stride={stride} 整除")
        # 使用 checkpoint 重建后的像素单位；不拿 NeUF 的体素 spacing 替换它。
        spacing = tuple(float(value) for value in model.stv.spacing_mm)
        if len(spacing) != 2 or not all(math.isfinite(value) and value > 0 for value in spacing):
            raise ValueError("checkpoint spacing 必须为两个有限正数")
        self._model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.tile_size = int(tile_size)
        self._spacing = spacing
        self._predict = context_tiled_prediction
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self._metadata = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": digest.hexdigest(),
            "component_schema": payload["component_schema"],
            "model_config": model_config,
            "geometry_mode": "none",
            "spacing": list(spacing),
            "spacing_interpretation": "checkpoint_pixel_space_proxy",
            "intensity_domain": "display_0_1",
            "required_halo": int(model.required_halo()),
            "grid_stride": stride,
            "tile_size": self.tile_size,
            "decomposition": "standalone_sbr_v1: S=A_raw-B_raw; B=B_raw; R=I-A_raw",
            "residual_semantics": "speckle_and_other_residual",
        }

    def metadata(self) -> dict[str, Any]:
        """仅返回配置和来源标识，不含 teacher 参数或 optimizer 状态。"""

        return deepcopy(self._metadata)

    def predict_frame(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """输入真实观测 [1,1,H,W]，输出同形的普通无梯度张量。

        保留 raw signed 分量，不做逐帧归一化或裁值。valid_mask 为观测 mask；
        它不同于仅用于 STV 张量诊断的、更窄的邻域有效域。
        """

        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[:2] != (1, 1)
            or min(image.shape[-2:]) < 1
            or not image.is_floating_point()
        ):
            raise ValueError("image 必须是非空浮点 [1,1,H,W] 观测切片")
        if image.requires_grad:
            raise ValueError("teacher 输入必须是无梯度的真实观测，不能使用 NeUF 预测")
        if image.device != self.device:
            raise ValueError("image 必须与冻结 teacher 位于同一 device")
        _validate_mask(mask, image)
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError("image 必须有限且位于 display [0,1]，不得自动重新归一化")

        self._model.eval()
        with torch.inference_mode():
            result = self._predict(
                self._model,
                image.float(),
                mask,
                geometry_mode="none",
                spacing_mm=self._spacing,
                tile_size=self.tile_size,
                amp=False,
            )
            for name, value in (
                ("anatomy", result.anatomy),
                ("boundary", result.boundary),
                ("support_probability", result.support_probability),
                ("edge_confidence", result.stv.confidence),
            ):
                _validate_value(value, image, name)
            anatomy = result.anatomy.float() * mask
            boundary = result.boundary.float() * mask
            structure = anatomy - boundary
            residual = image.float() * mask - anatomy
            targets = {
                "structure": structure,
                "boundary": boundary,
                "residual": residual,
                "anatomy": anatomy,
                "support_probability": result.support_probability.float() * mask,
                "edge_confidence": result.stv.confidence.float() * mask,
                "valid_mask": mask,
            }
            closure = (structure + boundary + residual - image.float()).abs()[mask]
            if not torch.isfinite(closure).all() or closure.max() >= 1e-6:
                raise RuntimeError("冻结 STV 的 S+B+R 在观测 mask 内未满足 1e-6 闭合误差")
        # inference tensor 不能被某些训练损失保存供 backward 使用，退出后明确复制。
        with torch.inference_mode(False):
            return {name: value.detach().clone() for name, value in targets.items()}


def _validate_mask(mask: torch.Tensor, reference: torch.Tensor) -> None:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.shape != reference.shape
        or mask.device != reference.device
    ):
        raise ValueError("mask 必须为与分量同 shape、同 device 的 bool Tensor")
    if not mask.any():
        raise ValueError("监督 mask 不能为空")


def _validate_value(value: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != reference.shape
        or value.device != reference.device
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} 必须为同 shape、同 device 的浮点 Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN/Inf")


def frozen_stv_component_loss(
    components: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    structure_weight: float = 1.0,
    boundary_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """三分量 masked MSE；接受整帧或采样点，总图像重建项由训练器负责。"""

    names = ("structure", "boundary", "residual")
    if not isinstance(components, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("components 与 targets 必须是分量名到 Tensor 的映射")
    missing = [name for name in names if name not in components or name not in targets]
    if missing:
        raise ValueError(f"分量监督缺少键: {missing}")
    weights = (structure_weight, boundary_weight, residual_weight)
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights
    ):
        raise ValueError("分量权重必须为有限非负数")
    reference = components["structure"]
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise ValueError("structure 必须为浮点 Tensor")
    _validate_mask(mask, reference)
    losses = []
    metrics = {}
    for name, weight in zip(names, weights):
        prediction, target = components[name], targets[name]
        _validate_value(prediction, reference, name)
        _validate_value(target, reference, f"{name}_target")
        if target.requires_grad:
            raise ValueError(f"{name}_target 必须是冻结的无梯度监督")
        mse = (prediction.float()[mask] - target.float()[mask]).square().mean()
        losses.append(float(weight) * mse)
        metrics[f"frozen_stv_{name}_mse"] = mse.detach()
    loss = sum(losses)
    metrics["frozen_stv_component_loss"] = loss.detach()
    return loss, metrics


def anatomy_smooth_weight(
    anatomy: torch.Tensor, confidence: torch.Tensor, mask: torch.Tensor,
    *, edge_scale: float, margin_pixels: int,
) -> torch.Tensor:
    """在完整真实切片上建立无梯度权重，避免 patch 边缘制造假梯度。

    teacher 边缘置信度与 anatomy 梯度共同保护边界；sector 外及其邻域不做
    三维平滑。edge_scale 使用 display 强度/像素，不解释为物理散斑参数。
    """
    dx = F.pad((anatomy[..., 1:] - anatomy[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((anatomy[..., 1:, :] - anatomy[..., :-1, :]).abs(), (0, 0, 0, 1))
    radius = max(2, int(margin_pixels))
    kernel = 2 * radius + 1
    gradient = F.max_pool2d(torch.maximum(dx, dy), kernel, stride=1, padding=radius)
    protected = F.max_pool2d(confidence.clamp(0, 1), kernel, stride=1, padding=radius)
    # 显式以无效值填充，图像四周也包含在侵蚀范围内。
    invalid = F.pad((~mask).float(), (radius,) * 4, value=1)
    interior = F.max_pool2d(invalid, kernel, stride=1) == 0
    return ((1 - protected) * torch.exp(-gradient / edge_scale) * interior).detach()


def frozen_stv_anatomy_loss(
    components: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor],
    mask: torch.Tensor, *, patch_size: int, anatomy_weight: float, edge_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """显式监督 alpha=0；保边项比较可信 teacher anatomy 的两种尺度梯度。

    仅比较 patch 内两端均有效的像素对，不对原始含散斑图的所有高频做匹配。
    """
    anatomy = components["anatomy"].float()
    target = targets["structure"] + targets["boundary"]
    fit = (anatomy[mask] - target[mask]).square().mean()
    edge = anatomy.sum() * 0
    if edge_weight:
        prediction_patches = anatomy.reshape(-1, patch_size, patch_size)
        target_patches = target.reshape_as(prediction_patches)
        valid = mask.reshape_as(prediction_patches)
        confidence = targets["edge_confidence"].reshape_as(prediction_patches).clamp(0, 1)
        terms = []
        for dim in (1, 2):
            for stride in (1, 2):
                left, right = [slice(None)] * 3, [slice(None)] * 3
                left[dim], right[dim] = slice(None, -stride), slice(stride, None)
                left, right = tuple(left), tuple(right)
                pair_mask = valid[left] & valid[right]
                weights = torch.maximum(confidence[left], confidence[right]) * pair_mask
                difference = ((prediction_patches[right] - prediction_patches[left])
                              - (target_patches[right] - target_patches[left])) / stride
                terms.append((weights * difference.square()).sum() / weights.sum().clamp_min(1e-8))
        edge = torch.stack(terms).mean()
    return anatomy_weight * fit + edge_weight * edge, {
        "anatomy_mse": fit.detach(), "anatomy_edge_mse": edge.detach(),
    }


def anatomy_spatial_loss(
    model, points: torch.Tensor, anatomy: torch.Tensor, mask: torch.Tensor,
    smooth_weight: torch.Tensor, bounding_box, *, step_mm: float, max_points: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """毫米坐标中三个方向的中心二阶差分，约束切片之间的局部 anatomy 曲率。

    这是边界加权的三维邻域先验，不是独立三维真值监督。只对 anatomy 求导，
    不限制 R；共享主干仍会同时接收完整图像重建梯度。随机源独立于 patch 采样。
    """
    xyz = points.reshape(-1, 3)
    box = torch.stack([torch.as_tensor(value, device=xyz.device, dtype=xyz.dtype) for value in bounding_box])
    valid = mask.reshape(-1) & (smooth_weight.reshape(-1) > 0)
    valid &= ((xyz - step_mm >= box[0]) & (xyz + step_mm <= box[1])).all(-1)
    indices = torch.nonzero(valid).flatten()
    indices = indices[torch.randperm(len(indices), device=xyz.device, generator=generator)[:max_points]]
    if not len(indices):
        zero = anatomy.sum() * 0
        return zero, {"anatomy_spatial_mse": zero.detach(), "anatomy_spatial_points": zero.detach()}
    offsets = torch.eye(3, device=xyz.device, dtype=xyz.dtype) * step_mm
    neighbors = xyz[indices, None, :] + torch.cat((offsets, -offsets))[None, :, :]
    values = model.query_components(neighbors.reshape(-1, 3), alpha=0)["anatomy"].reshape(-1, 6)
    center = anatomy.reshape(-1)[indices, None]
    curvature = (values[:, :3] + values[:, 3:] - 2 * center) / step_mm**2
    weights = smooth_weight.reshape(-1)[indices, None]
    loss = (weights * curvature.square()).mean()
    return loss, {
        "anatomy_spatial_mse": loss.detach(),
        "anatomy_spatial_points": anatomy.new_tensor(len(indices)),
    }
