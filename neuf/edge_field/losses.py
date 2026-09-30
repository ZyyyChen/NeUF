from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_masked(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def sagittal_profile_losses(source_gray, reference_gray, source_edge, reference_edge, *,
                            windows, window_weight, support, valid, distance_mm,
                            minimum_variance, ssim_weight=0., coverage_weight=.2):
    """沿固定交线窗口比较结构；输入 [帧,深度]，不计算跨切面像素 MSE。"""
    a, b = source_edge[:, windows], reference_edge[:, windows]
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    va, vb = ac.square().mean(-1), bc.square().mean(-1)
    ncc = ((ac * bc).mean(-1) / ((va + 1e-12) * (vb + 1e-12)).sqrt()).clamp(-1, 1)
    a, b = source_gray[:, windows], reference_gray[:, windows]
    ma, mb = a.mean(-1), b.mean(-1)
    ac, bc = a - ma[..., None], b - mb[..., None]
    ga, gb = ac.square().mean(-1), bc.square().mean(-1)
    ssim = ((2 * ma * mb + .01 ** 2) * (2 * (ac * bc).mean(-1) + .03 ** 2)
            / ((ma.square() + mb.square() + .01 ** 2) * (ga + gb + .03 ** 2)))
    # 低结构或离开初始有效域时，原固定窗口仍在分母内，不能靠丢弃难点降低损失。
    invalid = 1 - valid[:, windows].mean(-1)
    low_structure = torch.relu(1 - vb / minimum_variance.clamp_min(1e-12))
    residual = 1 - ncc + ssim_weight * (1 - ssim) + 2 * invalid + 2 * low_structure
    count = window_weight.sum(-1).clamp_min(1)
    sample_count = support.sum(-1).clamp_min(1)
    outside = (distance_mm.square() * support).sum(-1) / sample_count
    return dict(loss=(residual * window_weight).sum(-1) / count + coverage_weight * outside,
                edge_ncc=(ncc * window_weight).sum(-1) / count,
                ssim=(ssim * window_weight).sum(-1) / count,
                coverage=(valid * support).sum(-1) / sample_count,
                window_edge_ncc=ncc)


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


def laplacian(image):
    """原生像素网格的五点 Laplacian，用二阶变化约束边缘过渡宽度。"""
    return (image[..., 2:, 1:-1] + image[..., :-2, 1:-1]
            + image[..., 1:-1, 2:] + image[..., 1:-1, :-2]
            - 4 * image[..., 1:-1, 1:-1])


def normal_profile_loss(gray, target, response, mask, *, edge_weighted=False):
    """在 observed 灰度确定的法线两侧比较阶跃，直接约束原生像素的边缘过渡。"""
    if edge_weighted:
        target, response = target.detach(), response.detach()
    batch, _, height, width = gray.shape
    tx, ty = gradients(blur(target, 1))
    magnitude = (tx.square() + ty.square()).sqrt()
    nx, ny = tx / magnitude.clamp_min(1e-6), ty / magnitude.clamp_min(1e-6)
    # 整个采样剖面须位于有效扇区内；方向和候选像素不对预测图求导。
    interior = F.avg_pool2d(mask.float(), 9, stride=1, padding=4)[..., 1:-1, 1:-1] > .999
    selected = interior & (response[..., 1:-1, 1:-1] > .15) & (magnitude > .01)
    rows = torch.arange(1, height-1, device=gray.device, dtype=gray.dtype)[None, None, :, None]
    cols = torch.arange(1, width-1, device=gray.device, dtype=gray.dtype)[None, None, None, :]
    offsets = gray.new_tensor((-3., -2., -1., 1., 2., 3.))[None, :, None, None]
    x = (cols + offsets * nx) * (2 / (width-1)) - 1
    y = (rows + offsets * ny) * (2 / (height-1)) - 1
    grid = torch.stack((x, y), -1).reshape(batch, 6 * (height-2), width-2, 2)

    def across(image):
        values = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
        values = values.reshape(batch, 1, 6, height-2, width-2)
        return values[:, :, 3:] - values[:, :, :3].flip(2)

    # 每个半径单独对应 observed 两侧灰度差；不把无方向 response 当灰度目标。
    difference = across(gray) - across(target)
    if edge_weighted:
        # 新分支按固定软 response 独立归一化；旧分支仍保留原有 L1 和二值筛选。
        error = ((difference.square() + 1e-6).sqrt() - .001).mean(2)
        weight = response[..., 1:-1, 1:-1] * selected
        return (error * weight).sum() / weight.sum().clamp_min(1e-8)
    error = difference.abs().mean(2)
    return mean_masked(error, selected)


def edge_preserve_weights(target, response, mask):
    """训练和诊断共用的固定保边权重，输出对应中心差分的 [B,1,H-2,W-2]。"""
    target, response = target.detach(), response.detach()
    sx, sy = gradients(blur(target, 1))
    magnitude = (sx.square() + sy.square()).sqrt()
    valid = F.avg_pool2d(mask.float(), 9, stride=1, padding=4)[..., 1:-1, 1:-1] > .999
    edge = response[..., 1:-1, 1:-1]
    return edge * (edge > .15) * (magnitude > .01) * valid


def edge_preserving_losses(prediction, patch):
    """仅在固定结构边缘匹配原生梯度和剖面；背景只保留灰度保真。"""
    gray, predicted_edge = prediction[:, :1], prediction[:, 1:]
    target, response = patch["image"].detach(), patch["edge"].detach()
    mask = patch["mask"]
    photo = mean_masked(((gray - target).square() + 1e-6).sqrt() - .001, mask)
    # 半径 4 的有效邻域同时覆盖结构平滑、中心差分和最远法向采样。
    weight = edge_preserve_weights(target, response, mask)
    gx, gy = gradients(gray)
    tx, ty = gradients(target)
    error = ((gx - tx).square() + 1e-6).sqrt() - .001
    error = error + ((gy - ty).square() + 1e-6).sqrt() - .001
    # 空权重区域的分子仍连接预测图，因此返回可反传的零。
    gradient = (error * weight).sum() / weight.sum().clamp_min(1e-8)
    profile = normal_profile_loss(gray, target, response, mask, edge_weighted=True)
    foreground = mask & (response > .15)
    background = mask & ~foreground
    edge_error = (predicted_edge - response).abs()
    edge_loss = .5 * (mean_masked(edge_error, foreground) + mean_masked(edge_error, background))
    return dict(photo=photo, gradient=gradient, profile=profile, edge=edge_loss)


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


def edge_guided_losses(prediction, patch, progress, *, use_guidance, use_sharpness=False,
                       focus_observed_edges=False, use_profile=False, use_l1=False):
    """Observed B-mode 重建；NLSTV response 只作为边缘权重和辅助目标。

    前 10% 仅拟合灰度，10%–20% 线性打开边缘项，避免网络尚未形成低频结构时
    被稀疏高响应主导。response 是无方向标量，因此梯度方向来自 observed 灰度目标。
    """
    gray, predicted_edge = prediction[:, :1], prediction[:, 1:]
    target, mask = patch["image"], patch["mask"]
    # 严格 L1 基线不读取 teacher；既有方法继续使用原 Charbonnier 保真项。
    error = (gray - target).abs() if use_l1 else ((gray - target).square() + 1e-6).sqrt() - .001
    photo = mean_masked(error, mask)
    zero = photo * 0
    if not use_guidance:
        return dict(photo=photo, gradient=zero, edge=zero, sharpness=zero, ramp=zero)

    response = patch["edge"].detach()
    ramp = min(1.0, max(0.0, (float(progress) - .1) / .1))
    gx, gy = gradients(gray)
    tx, ty = gradients(target)
    valid = mask[..., 1:-1, 1:-1]
    valid = valid & mask[..., :-2, 1:-1] & mask[..., 2:, 1:-1]
    valid = valid & mask[..., 1:-1, :-2] & mask[..., 1:-1, 2:]
    weight = 1 + 2 * response[..., 1:-1, 1:-1]
    gradient_error = (gx - tx).abs() + (gy - ty).abs()
    gradient = mean_masked(weight * gradient_error, valid)
    if focus_observed_edges:
        # response 只定位候选区域；灰度边缘方向及幅度始终来自 observed 图。
        # 单独归一化可避免稀疏的陡峭边界被整块 patch 的普通像素稀释。
        target_gradient = (tx.square() + ty.square()).sqrt().detach()
        focused = valid & (response[..., 1:-1, 1:-1] > .15) & (target_gradient > .01)
        gradient = gradient + .5 * mean_masked(gradient_error, focused)

    # 只在 teacher 指示的结构区域匹配二阶变化；目标仍来自 observed 灰度，
    # 因而不会把无方向的 response 本身当作灰度边缘。
    sharpness = zero
    if use_sharpness:
        sharp_weight = response[..., 1:-1, 1:-1]
        sharpness = mean_masked(
            (laplacian(gray) - laplacian(target)).abs(),
            valid.to(sharp_weight.dtype) * sharp_weight,
        )

    foreground = mask & (response > .15)
    background = mask & ~foreground
    edge_error = (predicted_edge - response).abs()
    edge = .5 * (mean_masked(edge_error, foreground) + mean_masked(edge_error, background))
    ramp_tensor = photo.new_tensor(ramp)
    result = dict(photo=photo, gradient=gradient, edge=edge, sharpness=sharpness, ramp=ramp_tensor)
    if use_profile:
        result["profile"] = normal_profile_loss(gray, target, response, mask)
    return result


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
