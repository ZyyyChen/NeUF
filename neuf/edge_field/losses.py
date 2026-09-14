from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_masked(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def blur(image, sigma):
    radius = int(3 * sigma)
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = (-axis.square() / (2 * sigma * sigma)).exp()
    kernel = kernel / kernel.sum()
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), kernel[None, None, None])
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), kernel[None, None, :, None])


def local_correlation(a, b, mask, window=9):
    """只评价完整有效窗口；常量目标窗口不提供位置匹配信号。"""
    pool = lambda x: F.avg_pool2d(x, window, stride=1)
    ma, mb = pool(a), pool(b)
    va, vb = (pool(a * a) - ma * ma).clamp_min(0), (pool(b * b) - mb * mb).clamp_min(0)
    covariance = pool(a * b) - ma * mb
    valid = (pool(mask.float()) > .999) & (vb > 1e-6)
    score = covariance / (va * vb + 1e-10).sqrt()
    return mean_masked(1 - score.clamp(-1, 1), valid)


def gradients(image):
    dy = (image[..., 2:, 1:-1] - image[..., :-2, 1:-1]) / 2
    dx = (image[..., 1:-1, 2:] - image[..., 1:-1, :-2]) / 2
    return dx, dy


def response_losses(prediction, patch, progress, *, pose_step=False):
    """仅依赖 teacher response；灰度既不提供目标，也不参与置信度筛选。"""
    target, mask = patch["edge"], patch["mask"]
    zero = prediction.sum() * 0
    # 分别归一化有边缘和低响应区域，避免大面积背景淹没稀疏边缘。
    foreground = mask & (target > .15)
    background = mask & ~foreground
    balanced = lambda error: .5 * (mean_masked(error, foreground) + mean_masked(error, background))
    if pose_step:
        # 位姿利用逐渐收紧的平滑场，保留对坐标的梯度；末段冻结位姿。
        sigma = 2. if progress < .5 else 1.
        p, t = blur(prediction, sigma), blur(target, sigma)
        valid = mask.clone()
        radius = int(3 * sigma)
        valid[..., :radius, :] = valid[..., -radius:, :] = False
        valid[..., :, :radius] = valid[..., :, -radius:] = False
        shape = local_correlation(p, t, valid)
        return dict(response=mean_masked((p-t).abs(), valid), shape=shape, detail=zero)
    # 原生 response 保真项始终保留，最后阶段不再模糊监督目标。
    fidelity = balanced(((prediction-target).square() + 1e-6).sqrt() - .001)
    sigma = 2. if progress < .35 else (1. if progress < .7 else 0.)
    p, t = (blur(prediction, sigma), blur(target, sigma)) if sigma else (prediction, target)
    valid = mask.clone()
    border = max(1, int(3*sigma))
    valid[..., :border, :] = valid[..., -border:, :] = False
    valid[..., :, :border] = valid[..., :, -border:] = False
    shape = local_correlation(p, t, valid)
    px, py = gradients(prediction)
    tx, ty = gradients(target)
    interior = mask[..., 1:-1, 1:-1] & mask[..., :-2, 1:-1] & mask[..., 2:, 1:-1]
    interior = interior & mask[..., 1:-1, :-2] & mask[..., 1:-1, 2:]
    detail = mean_masked((px-tx).abs() + (py-ty).abs(), interior)
    return dict(response=fidelity, shape=shape, detail=detail)


def losses(prediction, patch, progress, *, use_edges, pose_step=False):
    image, edge, mask = patch["image"], patch["edge"], patch["mask"]
    gray, response = prediction[:, :1], prediction[:, 1:]
    error = (gray - image).square()
    # pose 阶段使用低频灰度；网络阶段保留原图细节监督。
    if pose_step:
        error = (blur(gray, 2) - blur(image, 2)).square()
    gray_loss = mean_masked((error + 1e-6).sqrt() - .001, mask)
    zero = gray_loss * 0
    if not use_edges:
        return dict(gray=gray_loss, edge=zero, couple=zero)
    sigmas = (4., 2.) if progress < .35 else ((2., 1.) if progress < .7 else (1.,))
    edge_loss = zero
    for sigma in sigmas:
        p, t = blur(response, sigma), blur(edge, sigma)
        # 排除平滑核越过 patch 边缘的像素，不让复制填充制造匹配。
        valid = mask.clone()
        radius = int(3 * sigma)
        valid[..., :radius, :] = valid[..., -radius:, :] = False
        valid[..., :, :radius] = valid[..., :, -radius:] = False
        edge_loss = edge_loss + mean_masked((p - t).abs(), valid) + .1 * local_correlation(p, t, valid)
    edge_loss = edge_loss / len(sigmas)
    px, py = gradients(blur(gray, 1))
    tx, ty = gradients(blur(image, 1))
    pm, tm = (px.square() + py.square() + 1e-10).sqrt(), (tx.square() + ty.square() + 1e-10).sqrt()
    target_response = edge[..., 1:-1, 1:-1]
    # teacher 必须同时获得原始灰度梯度支持；标量 response 不冒充梯度方向。
    confidence = mask[..., 1:-1, 1:-1] * (target_response > .15) * (tm.detach() > .005)
    direction = 1 - (px * tx + py * ty) / (pm * tm + 1e-8)
    couple = mean_masked(direction, confidence)
    couple = couple + .25 * local_correlation(pm, target_response, mask[..., 1:-1, 1:-1])
    return dict(gray=gray_loss, edge=edge_loss, couple=couple)
