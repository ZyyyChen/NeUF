"""训练帧的局部三维邻接、response 插值和边缘保护空间正则。"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.nn import functional as F

from .losses import blur, gradients, local_correlation, mean_masked


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


class EdgeAwareSpatialRegularizer:
    """固定训练观测保护边界，在有双侧支持的毫米空间约束局部灰度变化。"""

    @torch.no_grad()
    def __init__(self, data, *, max_gap_mm=3., band_radius_px=3):
        if max_gap_mm <= 0 or band_radius_px < 0 or int(band_radius_px) != band_radius_px:
            raise ValueError("空间正则要求 max_gap_mm>0、band_radius_px 为非负整数")
        train = data.splits["training"]
        if not len(train):
            raise ValueError("空间正则至少需要一张训练帧")
        device = data.initial.device
        # 几何邻接也限制在训练子集；此视图不持有验证/测试图像或 response。
        source = SimpleNamespace(
            initial=data.initial[train].detach().clone(),
            frame_ids=data.frame_ids[train].detach().clone(),
            local=data.local.detach(), height=data.height, width=data.width,
            mask=data.mask.detach().clone(), edges=None,
            splits={"training": torch.arange(len(train), device=device)},
        )
        self.reference = NeighborEdgeVolume(source, max_gap_mm=max_gap_mm, erosion_px=4)
        self.frame_lookup = torch.full((len(data.initial),), -1, dtype=torch.long, device=device)
        self.frame_lookup[train] = torch.arange(len(train), device=device)
        self.coverage = self.reference.mask.float()[None, None]
        # 世界探针沿用训练 patch 的源扇区；旧合成数据仍可仅调用原训练接口。
        self.source_interior = getattr(data, "interior", None)
        if self.source_interior is not None:
            self.source_interior = self.source_interior.detach().clone()
        self.protected = torch.empty((len(train), data.height, data.width), dtype=torch.bool, device=device)
        # 仅缓存可靠切向；半精度缓存减小占用，采样前转回坐标精度。
        self.tangents = torch.empty((len(train), 2, data.height, data.width), dtype=torch.float16, device=device)
        spacing = self.reference.span / self.reference.span.new_tensor((data.height-1, data.width-1))
        radius = int(band_radius_px)
        for start in range(0, len(train), 8):
            indices = train[start:start+8]
            response = data.edges[indices].detach()[:, None]
            edges = (response > .15) & source.mask[None, None]
            band = F.max_pool2d(edges.float(), 2*radius+1, stride=1, padding=radius)
            self.protected[start:start+len(indices)] = band[:, 0].bool()
            gx, gy = gradients(blur(data.images[indices].detach()[:, None], 1.))
            reliable = ((response[..., 1:-1, 1:-1] > .15)
                        & ((gx.square()+gy.square()).sqrt() > .01)
                        & self.reference.mask[None, None, 1:-1, 1:-1])
            # pixel=(col,row)，local=(row_mm,col_mm,0)：切向为 (gx*sh,-gy*sw,0)。
            tangent = torch.cat((gx*spacing[0], -gy*spacing[1]), 1)
            tangent = tangent / tangent.square().sum(1, keepdim=True).sqrt().clamp_min(1e-8)
            self.tangents[start:start+len(indices)] = F.pad(tangent*reliable, (1, 1, 1, 1)).half()
        self.metadata = dict(
            method="fixed training-only edge protection and local 3D gray differences",
            max_gap_mm=max_gap_mm, max_angle_deg=15., frame_radius=8,
            band_radius_px=radius, edge_threshold=.15, observed_gradient_threshold=.01,
            training_frames=int(len(train)), supported_training_frames=int(self.reference.valid_frames.sum()),
            background="off-plane points; random signed world axis; five segment support checks",
            tangent="observed in-plane edge tangent only; no extrapolated 3D normal",
            protection="maximum fixed teacher band over source and both training donors; five segment points",
        )

    @torch.no_grad()
    def _segment_support(self, frames, start, delta):
        """[B,K,3] 本地坐标线段；五点均须被两训练平面夹持并落在有效扇区。"""
        batch, count = start.shape[:2]
        fraction = torch.linspace(0, 1, 5, device=start.device, dtype=start.dtype)
        segment = start[:, :, None] + fraction[None, None, :, None]*delta[:, :, None]
        reference = self.reference
        grid, distance = reference._grid(frames, segment.reshape(batch, count*5, 3),
                                         reference.initial, reference.initial)
        flat = grid.reshape(batch*2, count*5, 1, 2)
        coverage = F.grid_sample(self.coverage.expand(batch*2, -1, -1, -1), flat,
                                 align_corners=True)[:, 0, :, 0].reshape(batch, 2, count, 5)
        band = self.protected[reference.neighbors[frames]].reshape(batch*2, 1, *self.protected.shape[-2:])
        protected = F.grid_sample(band.to(start.dtype), flat, align_corners=True)
        protected = protected[:, 0, :, 0].reshape(batch, 2, count, 5).amax(dim=(1, 3)) > 0
        # 源帧独有的边缘也保守保护；空间有效性仍完全由两侧 donor 决定。
        source_grid = (2*(segment[..., :2]-reference.origin)/reference.span-1).flip(-1)
        source_band = F.grid_sample(self.protected[frames, None].to(start.dtype),
                                   source_grid.reshape(batch, count*5, 1, 2), align_corners=True)
        protected |= source_band[:, 0, :, 0].reshape(batch, count, 5).amax(-1) > 0
        distance = distance.reshape(batch, 2, count, 5)
        valid = reference.valid_frames[frames, None] & (coverage.amin(dim=(1, 3)) > .999)
        valid &= (distance[:, 0].amin(-1) > 1e-4) & (distance[:, 1].amax(-1) < -1e-4)
        valid &= distance.abs().amax(dim=(1, 3)) <= reference.max_gap_mm
        return valid, protected

    @torch.no_grad()
    def classify_world_segments(self, world, delta, *, chunk=4096):
        """分类 [N,3] 世界毫米线段，delta 可为 [3]；这是支持探针，不是训练采样频率。

        最近源帧只要求投影位于训练 interior，不优先选择有双侧支持的帧。
        source_index 为训练子集索引；无源帧时两个编号为 -1、距离为 inf。
        valid 包含离源平面大于 1e-4 mm 的条件，protected 独立报告固定保护带。
        """
        if self.source_interior is None:
            raise ValueError("世界支持探针需要与训练 patch 一致的 data.interior")
        reference = self.reference
        matrices = reference.initial
        world = torch.as_tensor(world, device=matrices.device, dtype=matrices.dtype)
        delta = torch.as_tensor(delta, device=world.device, dtype=world.dtype)
        if (world.ndim != 2 or world.shape[1] != 3 or delta.shape not in ((3,), world.shape)
                or not isinstance(chunk, int) or chunk < 1):
            raise ValueError("要求 world=[N,3]、delta=[3] 或 [N,3]、chunk 为正整数")
        if not torch.isfinite(world).all() or not torch.isfinite(delta).all():
            raise ValueError("世界线段坐标必须为有限毫米值")
        delta = delta.expand_as(world)
        source_index = torch.full((len(world),), -1, dtype=torch.long, device=world.device)
        distance_mm = world.new_full((len(world),), torch.inf)
        source_mask = self.source_interior.to(world)[None, None]
        # 按点分块投影到全部训练平面，避免建立完整 N×帧数 坐标矩阵。
        for start in range(0, len(world), chunk):
            points = world[start:start+chunk]
            local = torch.einsum("fij,fni->fnj", matrices[:, :3, :3],
                                 points[None]-matrices[:, None, :3, 3])
            grid = (2*(local[..., :2]-reference.origin)/reference.span-1).flip(-1)
            covered = F.grid_sample(source_mask, grid.reshape(1, -1, 1, 2), align_corners=True)
            covered = covered.reshape(len(matrices), len(points)) > .999
            distances = torch.where(covered, local[..., 2].abs(), torch.inf)
            nearest, frames = distances.min(0)
            source_index[start:start+len(points)] = torch.where(nearest.isfinite(), frames, -1)
            distance_mm[start:start+len(points)] = nearest

        valid = torch.zeros(len(world), dtype=torch.bool, device=world.device)
        protected = torch.zeros_like(valid)
        # 按源帧分组复用原五点判定，避免为每个世界点复制整张 donor 掩膜。
        for frame in source_index.unique():
            if frame < 0:
                continue
            indices = torch.where(source_index == frame)[0]
            matrix = matrices[frame]
            for group in indices.split(chunk):
                local = (world[group]-matrix[:3, 3]) @ matrix[:3, :3]
                local_delta = delta[group] @ matrix[:3, :3]
                supported, band = self._segment_support(frame[None], local[None], local_delta[None])
                valid[group] = supported[0] & (distance_mm[group] > 1e-4)
                protected[group] = band[0]
        source_frame_id = torch.full_like(source_index, -1)
        present = source_index >= 0
        source_frame_id[present] = reference.data.frame_ids[source_index[present]]
        return dict(valid=valid, protected=protected, source_frame_id=source_frame_id,
                    distance_mm=distance_mm, source_index=source_index)

    def loss(self, model, patch, generator, *, points=512, step_mm=.24, tangent_weight=.1):
        """每批共 points 个候选；独立 generator 不消耗主 patch 采样随机流。"""
        if points <= 0 or int(points) != points or step_mm <= 0 or tangent_weight < 0:
            raise ValueError("空间正则要求 points 为正整数、step_mm>0、tangent_weight>=0")
        reference = self.reference
        with torch.no_grad():
            frames = self.frame_lookup[patch["frames"]]
            if torch.any(frames < 0):
                raise ValueError("空间正则只接受 training 帧的 patch")
            local = patch["local"].detach()
            batch, total = local.shape[:2]
            count = math.ceil(points/batch)
            picks = torch.randint(total, (batch, count), generator=generator,
                                  device=generator.device).to(local.device)
            anchor = local.gather(1, picks[..., None].expand(-1, -1, 3))
            selected = patch["mask"].reshape(batch, -1).gather(1, picks).bool()
            selected &= torch.arange(batch*count, device=local.device).reshape(batch, count) < points
            edge_weight = patch["edge"].detach().reshape(batch, -1).gather(1, picks).clamp(0, 1)
            matrices = reference.initial[frames]
            _, distances = reference._grid(frames, anchor, reference.initial, reference.initial)
            donors = reference.initial[reference.neighbors[frames]]
            cosines = (matrices[:, None, :3, 2]*donors[..., :3, 2]).sum(-1)
            # 沿源平面法向采样夹持区间；不把其他帧的灰度插值作为监督目标。
            limits = -distances / cosines[:, :, None].clamp_min(1e-6)
            fraction = torch.rand((batch, count), generator=generator, device=generator.device).to(local)
            offset = limits[:, 0] + (.1+.8*fraction)*(limits[:, 1]-limits[:, 0])
            offset = torch.where(reference.valid_frames[frames, None], offset, 0.)
            spatial = anchor.clone()
            spatial[..., 2] += offset
            axis = torch.randint(6, (batch, count), generator=generator,
                                 device=generator.device).to(local.device)
            world_delta = F.one_hot(axis % 3, 3).to(local) * (1-2*(axis//3))[..., None] * step_mm
            delta = torch.einsum("bij,bki->bkj", matrices[:, :3, :3], world_delta)
            valid, protected = self._segment_support(frames, spatial, delta)
            valid &= selected & (offset.abs() > 1e-4)
            active = valid & ~protected

            grid = (2*(anchor[..., :2]-reference.origin)/reference.span-1).flip(-1)
            tangent = F.grid_sample(self.tangents[frames].to(local), grid[:, :, None],
                                    align_corners=True)[:, :, :, 0].transpose(1, 2)
            norm = tangent.square().sum(-1, keepdim=True).sqrt()
            tangent = F.pad(tangent/norm.clamp_min(1e-8), (0, 1))*step_mm
            # 面内切向只在原观测面上使用；向离面点复制会假定未知的三维法向。
            tangent_valid, _ = self._segment_support(frames, anchor, tangent)
            tangent_valid &= selected & (norm[..., 0] > .5)
            samples = torch.cat((spatial, spatial+delta, anchor, anchor+tangent), 1)
            world = torch.einsum("bij,bkj->bki", matrices[:, :3, :3], samples) + matrices[:, None, :3, 3]

        gray = model(world)[..., 0].reshape(batch, 4, count)
        rho = lambda value: (value.square()+1e-6).sqrt()-.001
        background = mean_masked(rho((gray[:, 1]-gray[:, 0])/step_mm), active)
        weight = tangent_valid.to(gray.dtype)*edge_weight
        tangent = (rho((gray[:, 3]-gray[:, 2])/step_mm)*weight).sum()/weight.sum().clamp_min(1e-8)
        zero = gray.sum()*0
        return dict(spatial=background+tangent_weight*tangent,
                    spatial_background=background, spatial_tangent=tangent,
                    spatial_valid_fraction=zero+valid.sum()/points,
                    spatial_active_fraction=zero+active.sum()/points)


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
