"""仅用其他训练帧构造局部三维 response 插值；不假设不同截面的边缘点重合。"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.nn import functional as F

from .losses import blur, local_correlation, mean_masked


class NeighborEdgeVolume:
    def __init__(self, data, *, max_gap_mm=3., max_angle_deg=15., frame_radius=8, erosion_px=32):
        self.data = data
        self.initial = data.initial
        self.origin = data.local[0, 0, :2]
        self.span = data.local[-1, -1, :2] - self.origin
        if torch.any(self.span.abs() < 1e-6):
            raise ValueError("面内坐标跨度无效")
        # 插值要求笛卡尔像素网格，禁止将极坐标网格当作线性像素坐标。
        rows = torch.linspace(0, 1, data.height, device=self.initial.device)
        cols = torch.linspace(0, 1, data.width, device=self.initial.device)
        torch.testing.assert_close(data.local[:, 0, 0], self.origin[0] + rows*self.span[0], atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(data.local[0, :, 1], self.origin[1] + cols*self.span[1], atol=1e-4, rtol=1e-5)
        mask = binary_erosion(data.mask.cpu().numpy(), iterations=erosion_px) if erosion_px else data.mask.cpu().numpy()
        self.mask = torch.as_tensor(mask, device=self.initial.device)
        self.center = data.local[self.mask].mean(0)
        centers = self.initial[:, :3, :3] @ self.center + self.initial[:, :3, 3]
        normals = self.initial[:, :3, 2]
        train = data.splits["training"]
        # distance[i,j] 是第 i 帧中心到候选 j 平面的有符号毫米距离。
        distance = ((centers[:, None] - self.initial[None, train, :3, 3]) * normals[None, train]).sum(-1)
        angle_ok = normals @ normals[train].T > math.cos(math.radians(max_angle_deg))
        id_gap = (data.frame_ids[:, None] - data.frame_ids[train][None]).abs()
        eligible = angle_ok & (id_gap > 0) & (id_gap <= frame_radius) & (distance.abs() <= max_gap_mm)
        below = torch.where(eligible & (distance > 1e-4), distance, torch.inf)
        above = torch.where(eligible & (distance < -1e-4), -distance, torch.inf)
        left, li = below.min(-1)
        right, ri = above.min(-1)
        self.neighbors = torch.stack((train[li], train[ri]), -1)
        self.valid_frames = left.isfinite() & right.isfinite()
        self.max_gap_mm = max_gap_mm
        self.maps = data.edges
        self.sigma = 0.
        active = torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
        active[train] = self.valid_frames[train]
        # 每个相连的训练子图至少固定一个参考帧，孤立或无双侧支持的帧保持初始位姿。
        parent = list(range(len(centers)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in train.cpu().tolist():
            if self.valid_frames[i]:
                for j in self.neighbors[i].cpu().tolist():
                    parent[root(i)] = root(j)
        anchors = {}
        for i in train.cpu().tolist():
            anchors.setdefault(root(i), i)
        active[list(anchors.values())] = False
        self.active = active
        gaps = torch.stack((left, right), -1)[self.valid_frames]
        self.metadata = dict(
            method="leave-one-frame-out, two-sided local response interpolation",
            assumption="nearby sections vary continuously; no same-3D-edge correspondence assumption",
            max_gap_mm=max_gap_mm, max_angle_deg=max_angle_deg, frame_radius=frame_radius,
            erosion_px=erosion_px, anchors=[int(data.frame_ids[i]) for i in anchors.values()],
            valid_frames=int(self.valid_frames.sum()), optimized_frames=int(active.sum()),
            side_gap_mm=dict(min=float(gaps.min()), median=float(gaps.median()), max=float(gaps.max())) if gaps.numel() else None,
            donor_frame_ids=data.frame_ids[self.neighbors].cpu().tolist(),
            frame_ids=data.frame_ids.cpu().tolist(), valid=self.valid_frames.cpu().tolist(),
            nearest_side_distance_mm=[[float(a) if np.isfinite(a) else None for a in pair]
                                      for pair in torch.stack((left, right), -1).cpu().numpy()],
            support="fixed initial mask and bracketing; losing source coverage is penalized",
            no_gray=True)

    def set_scale(self, sigma):
        self.sigma = sigma
        self.maps = torch.cat([blur(batch[:, None], sigma)[:, 0] for batch in self.data.edges.split(8)]) if sigma else self.data.edges

    def _grid(self, frames, local, matrices, source_matrices):
        world = torch.einsum("bij,bpj->bpi", matrices[frames, :3, :3], local) + matrices[frames, None, :3, 3]
        donors = source_matrices[self.neighbors[frames]]
        points = torch.einsum("bkij,bkpi->bkpj", donors[..., :3, :3], world[:, None]-donors[:, :, None, :3, 3])
        grid = (2*(points[..., :2]-self.origin)/self.span-1).flip(-1)
        return grid, points[..., 2]

    def sample(self, frames, local, matrices, *, source_matrices=None):
        """返回 [B,P] 的预测、固定有效域和当前覆盖率；源端为其他训练帧。"""
        batch, count = local.shape[:2]
        donors = matrices.detach() if source_matrices is None else source_matrices.detach()
        grid, _ = self._grid(frames, local, matrices, donors)
        flat = grid.reshape(batch*2, count, 1, 2)
        source = self.maps[self.neighbors[frames]].reshape(batch*2, 1, self.data.height, self.data.width)
        response = F.grid_sample(source, flat, align_corners=True, padding_mode="zeros")[:, 0, :, 0].reshape(batch, 2, count)
        mask = self.mask.float()[None, None].expand(batch*2, 1, -1, -1)
        coverage = F.grid_sample(mask, flat, align_corners=True)[:, 0, :, 0].reshape(batch, 2, count).amin(1)
        with torch.no_grad():
            initial_grid, distance = self._grid(frames, local, self.initial, self.initial)
            initial_coverage = F.grid_sample(mask, initial_grid.reshape(batch*2, count, 1, 2), align_corners=True)
            initial_coverage = initial_coverage[:, 0, :, 0].reshape(batch, 2, count).amin(1)
            target_grid = (2*(local[..., :2]-self.origin)/self.span-1).flip(-1)
            target_valid = F.grid_sample(self.mask.float()[None, None].expand(batch, 1, -1, -1), target_grid[:, :, None], align_corners=True)[:, 0, :, 0] > .999
            valid = self.valid_frames[frames, None] & (initial_coverage > .999) & target_valid
            valid &= (distance[:, 0] > 1e-4) & (distance[:, 1] < -1e-4) & (distance.abs().amax(1) <= self.max_gap_mm)
            # 双侧距离反向加权，平面位置固定使权重不参与位姿优化。
            weights = distance.abs().flip(1)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        return (response*weights).sum(1), valid, coverage


def alignment_loss(prediction, target, mask, coverage):
    """固定有效域的局部相关 + response 保真；移出支持域不能使像素消失。"""
    shape = local_correlation(prediction, target, mask, window=9)
    error = ((prediction-target).square()+1e-6).sqrt()-.001
    foreground = mask & (target > .15)
    background = mask & ~foreground
    fidelity = .5*(mean_masked(error, foreground)+mean_masked(error, background))
    support = mean_masked(1-coverage, mask)
    return dict(shape=shape, response=fidelity, coverage=support)


def alignment_checks(data):
    """用相同 edge 截面的已知面内扰动验证坐标恢复；不作为真实配准准确率。"""
    from types import SimpleNamespace
    from .model import InPlanePoseRefiner

    device = data.edges.device
    ids = torch.arange(5, device=device)
    truth = torch.eye(4, device=device).repeat(5, 1, 1)
    truth[:, 2, 3] = ids.float()-2
    center = data.local[data.interior].mean(0)
    active = ids == 2
    injected = InPlanePoseRefiner(truth, active, center)
    with torch.no_grad():
        injected.raw[2] = torch.tensor([.3, -.2, .3], device=device)
    initial = injected.matrices().detach()
    example = data.edges[int(data.splits["training"][len(data.splits["training"])//2])]
    toy = SimpleNamespace(initial=initial, local=data.local, edges=example[None].expand(5, -1, -1),
                          mask=data.mask, height=data.height, width=data.width, frame_ids=ids,
                          splits={"training": ids})
    reference = NeighborEdgeVolume(toy)
    pose = InPlanePoseRefiner(initial, active, center)
    indices = torch.tensor([2], device=device)
    local = data.local[::4, ::4].reshape(1, -1, 3)
    target = example[::4, ::4][None, None]
    optimizer = torch.optim.Adam([pose.raw], lr=.03)
    before = None
    for step in range(160):
        optimizer.zero_grad(set_to_none=True)
        predicted, valid, coverage = reference.sample(indices, local, pose.matrices())
        terms = alignment_loss(predicted.reshape_as(target), target, valid.reshape_as(target), coverage.reshape_as(target))
        loss = terms["shape"]+terms["response"]+terms["coverage"]
        before = float(loss.detach()) if before is None else before
        loss.backward()
        optimizer.step()
    corrected = pose.matrices().detach()
    toy_final_loss = float(loss.detach())
    probes = data.local[::32, ::32].reshape(-1, 3)
    apply = lambda m: probes @ m[:3, :3].T + m[:3, 3]
    before_mm = float((apply(initial[2])-apply(truth[2])).norm(dim=-1).mean())
    after_mm = float((apply(corrected[2])-apply(truth[2])).norm(dim=-1).mean())
    torch.testing.assert_close(corrected[:, :3, 2], initial[:, :3, 2], atol=1e-6, rtol=0)
    assert after_mm < .03 and after_mm < before_mm*.15, (before_mm, after_mm)
    assert not (reference.neighbors[reference.valid_frames] == ids[reference.valid_frames, None]).any()
    far = SimpleNamespace(**vars(toy))
    far.initial = truth.clone()
    far.initial[:, 2, 3] *= 10
    assert not NeighborEdgeVolume(far).valid_frames.any()
    # 使用真实、不同截面的 edge 检验增量恢复；源切片保持原始坐标且不含被扰动帧自身。
    real_reference = NeighborEdgeVolume(data)
    eligible = torch.where(real_reference.active)[0]
    selected = eligible[torch.linspace(0, len(eligible)-1, 6, device=device).long()]
    real_active = torch.zeros(len(data.initial), dtype=torch.bool, device=device)
    real_active[selected] = True
    perturbation = InPlanePoseRefiner(data.initial, real_active, real_reference.center)
    with torch.no_grad():
        signs = torch.tensor([1., -1., 1., -1., 1., -1.], device=device)
        perturbation.raw[selected] = signs[:, None]*torch.tensor([.3, -.2, .3], device=device)
    noisy = perturbation.matrices().detach()
    estimate = InPlanePoseRefiner(noisy, real_active, real_reference.center)
    optimizer = torch.optim.Adam([estimate.raw], lr=.02)
    local = data.local[::4, ::4].reshape(1, -1, 3).expand(len(selected), -1, -1)
    for step in range(200):
        sigma = 1. if step < 100 else 0.
        if real_reference.sigma != sigma:
            real_reference.set_scale(sigma)
        target = real_reference.maps[selected, ::4, ::4][:, None]
        optimizer.zero_grad(set_to_none=True)
        prediction, valid, coverage = real_reference.sample(selected, local, estimate.matrices(), source_matrices=data.initial)
        terms = alignment_loss(prediction.reshape_as(target), target, valid.reshape_as(target), coverage.reshape_as(target))
        loss = terms["shape"]+.5*terms["response"]+2*terms["coverage"]
        loss.backward()
        optimizer.step()
    estimated = estimate.matrices().detach()
    real_rows = []
    for i in selected.cpu().tolist():
        error_before = float((apply(noisy[i])-apply(data.initial[i])).norm(dim=-1).mean())
        error_after = float((apply(estimated[i])-apply(data.initial[i])).norm(dim=-1).mean())
        real_rows.append(dict(frame_id=int(data.frame_ids[i]), before_point_error_mm=error_before,
                              after_point_error_mm=error_after))
    real_before = float(np.mean([r["before_point_error_mm"] for r in real_rows]))
    real_after = float(np.mean([r["after_point_error_mm"] for r in real_rows]))
    assert real_after < real_before*.25, real_rows
    return dict(status="PASS", experiment="same edge extruded into five parallel slices; only central pose perturbed",
                before_point_error_mm=before_mm, after_point_error_mm=after_mm,
                initial_loss=before, final_loss=toy_final_loss, self_excluded=True,
                unsupported_frames_frozen=True, plane_normal_fixed=True,
                real_sequence_injected_perturbations=dict(frames=real_rows, mean_before_mm=real_before, mean_after_mm=real_after,
                    final_loss=float(loss.detach()),
                    reference="original poses used only as a controlled baseline, not anatomical ground truth"),
                interpretation="controlled implementation check, not real ultrasound pose accuracy")
