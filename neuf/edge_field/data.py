from __future__ import annotations

import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, minimum_filter

from neuf.dataset import Dataset
from neuf.phase1_data import build_phase1_split, hash_array, hash_file

from .losses import mean_masked, sagittal_profile_losses


class EdgeData:
    """沿用 NeUF 的毫米坐标；缓存的 response 仅在训练 split 被采样。"""

    def __init__(self, dataset_path, teacher_path, source_images, *, device="cuda", cache_images=True):
        dataset = Dataset.open_from_save(dataset_path, map_location="cpu")
        splits = build_phase1_split(dataset)
        refs = sorted((ref for group in splits.values() for ref in group),
                      key=lambda ref: ref.original_frame_index)
        ids = [ref.original_frame_index for ref in refs]
        if len(ids) != len(set(ids)):
            raise ValueError("原始帧编号重复，无法唯一映射 teacher")
        self.height, self.width = int(dataset.px_height), int(dataset.px_width)
        shape = self.height, self.width
        source = np.load(source_images, mmap_mode="r")
        if source.dtype != np.uint8 or source.shape[1:] != shape:
            raise ValueError("当前 teacher 原图接口要求 uint8 [N,H,W] 且方向一致")
        # 转换记录明确说明 images.npy 将扇区外显示内容清零，因此只在共享有效域核验。
        source_mask = np.load(Path(source_images).with_name("sector_mask.npy")).astype(bool)
        audit_mask = dataset.get_sector_mask().cpu().numpy().astype(bool) & source_mask
        # 纯 edge 模式使用已保存的固定扇区，灰度重建出的 legacy mask 只参与身份核验。
        mask = audit_mask if cache_images else source_mask
        if mask.shape != shape or mask.mean() < .1:
            raise ValueError("共享扇区 mask 无效")
        images, edges, matrices, alignment = [], [], [], []
        with h5py.File(teacher_path, "r") as teacher:
            expected_shape = (self.width, self.height, len(source))
            if teacher["edge_response"].shape != expected_shape:
                raise ValueError("teacher 必须为 MATLAB v7.3 [W,H,N]，禁止猜测转置")
            completed = np.asarray(teacher["completed"]).reshape(-1).astype(bool)
            # MATLAB 压缩 chunk 跨帧；一次读取避免每帧重复解压同一批数据。
            responses = np.asarray(teacher["edge_response"]).transpose(2, 1, 0)
            for ref in refs:
                idx = ref.original_frame_index
                if idx >= len(source) or not completed[idx]:
                    raise ValueError(f"teacher 帧 {idx} 不存在或未完成")
                pixels = dataset.pixels if ref.source == "training_pool" else dataset.pixels_valid
                observed = pixels[ref.slice_info.start:ref.slice_info.end].reshape(shape).numpy()
                original = np.asarray(source[idx], dtype=np.float32) / 255
                error = float(np.max(np.abs(observed - original)[audit_mask]))
                if error > 1e-6:
                    raise ValueError(f"teacher 原图和物理数据帧 {idx} 在共享扇区不一致: max_error={error}")
                edge = responses[idx].copy()
                if not np.isfinite(edge).all() or edge.min() < 0 or edge.max() > 1.00001:
                    raise ValueError(f"teacher response 数值无效: frame={idx}")
                matrix = np.eye(4, dtype=np.float32)
                matrix[:3, :3] = ref.slice_info.rotation.as_rotmat()
                matrix[:3, 3] = ref.slice_info.position
                if cache_images:
                    images.append(observed.astype(np.float32, copy=False))
                edges.append(edge)
                matrices.append(matrix)
                alignment.append(dict(
                    frame_id=idx, pixel_max_error=error,
                    response_hash=hash_array(edge), observed_hash=hash_array(observed),
                ))
        self.local = torch.tensor(np.stack((dataset.Y, dataset.X, np.zeros_like(dataset.X)), -1),
                                  dtype=torch.float32, device=device).reshape(*shape, 3)
        # 核对现有缓存点与新可微路径的方向、轴顺序和物理单位。
        probes = np.linspace(0, self.height * self.width - 1, 17).astype(int)
        for ref, matrix in zip(refs, matrices):
            points = dataset.points if ref.source == "training_pool" else dataset.points_valid
            actual = self.local.reshape(-1, 3)[probes].cpu().numpy() @ matrix[:3, :3].T + matrix[:3, 3]
            np.testing.assert_allclose(actual, points[ref.slice_info.start + probes].numpy(), atol=2e-4, rtol=1e-5)
        if cache_images:
            self.images = torch.tensor(np.stack(images), device=device)
        self.edges = torch.tensor(np.stack(edges), device=device)
        self.initial = torch.tensor(np.stack(matrices), device=device)
        self.frame_ids = torch.tensor(ids, dtype=torch.long, device=device)
        lookup = {idx: i for i, idx in enumerate(ids)}
        self.splits = {name: torch.tensor(sorted(lookup[r.original_frame_index] for r in group),
                                         dtype=torch.long, device=device) for name, group in splits.items()}
        interior = binary_erosion(mask, iterations=10)
        self.mask = torch.tensor(mask, device=device)
        self.interior = torch.tensor(interior, device=device)
        self.valid_centers = torch.nonzero(self.interior).cpu()
        self.edge_centers = [
            torch.tensor(np.argwhere(interior & (edge > .15)), dtype=torch.long)
            for edge in edges
        ]
        if any(len(self.edge_centers[index]) == 0 for index in self.splits["training"].cpu().tolist()):
            raise ValueError("至少一张训练帧没有 response>0.15 的可用边缘中心")
        self.bounds = torch.tensor(np.stack((dataset.point_min, dataset.point_max)), device=device)
        self.spacing = (dataset.roi_px_size_height_mm, dataset.roi_px_size_width_mm)
        # 延续工作区已有四帧和 ROI 对照；若划分变化，停止而不悄悄换图。
        comparison_ids = [236, 226, 216, 206]
        self.comparison = [lookup[idx] for idx in comparison_ids]
        if not set(self.comparison) <= set(self.splits["validation"].cpu().tolist()):
            raise ValueError("固定比较帧不全在 validation split 中")
        self.metadata = dict(dataset=str(Path(dataset_path).resolve()), teacher=str(Path(teacher_path).resolve()),
                             dataset_sha256=hash_file(dataset_path), teacher_sha256=hash_file(teacher_path),
                             source_images_sha256=hash_file(source_images), alignment=alignment,
                             mask_sha256=hash_array(mask), alignment_domain="intersection of baked sector and teacher-source sector; source has zeroed exterior",
                             training_mask="shared sector" if cache_images else "frozen source sector_mask.npy; independent of loaded gray",
                             frame_ids=ids, shape_hw=shape, spacing_hw_mm=self.spacing,
                             splits={k: self.frame_ids[v].cpu().tolist() for k, v in self.splits.items()},
                             comparison_indices=self.frame_ids[self.comparison].cpu().tolist(),
                             comparison_roi_xywh=[272, 407, 128, 128],
                             normalization=("baked observed B-mode [0,1]; source uint8 / 255 used for identity audit; "
                                            "saved per-frame p99.5 response [0,1]"),
                             coordinate_convention="local=(Y axial,X lateral,0) mm; world=R local+t",
                             teacher_kind="traditional NLSTV lambda=0.009; no neural teacher")
        del dataset, images, edges
        gc.collect()

    def patches(self, batch_size, size, generator, *, edge_fraction=0.0):
        if not 0 <= edge_fraction <= 1:
            raise ValueError("edge_fraction 必须位于 [0,1]")
        train = self.splits["training"]
        frames = train[torch.randint(len(train), (batch_size,), generator=generator).to(train.device)]
        picks = torch.randint(len(self.valid_centers), (batch_size,), generator=generator)
        centers = self.valid_centers[picks]
        edge_count = int(round(batch_size * edge_fraction))
        for patch_index in range(edge_count):
            candidates = self.edge_centers[int(frames[patch_index])]
            pick = torch.randint(len(candidates), (1,), generator=generator)
            centers[patch_index] = candidates[int(pick)]
        centers = centers.to(train.device)
        rows = (centers[:, 0] - size // 2).clamp(0, self.height - size)
        cols = (centers[:, 1] - size // 2).clamp(0, self.width - size)
        rr = rows[:, None, None] + torch.arange(size, device=train.device)[None, :, None]
        cc = cols[:, None, None] + torch.arange(size, device=train.device)[None, None, :]
        indices = frames[:, None, None]
        patch = dict(frames=frames, local=self.local[rr, cc].reshape(batch_size, -1, 3),
                     edge=self.edges[indices, rr, cc][:, None], mask=self.interior[rr, cc][:, None])
        if hasattr(self, "images"):
            patch["image"] = self.images[indices, rr, cc][:, None]
        patch["edge_centered_count"] = edge_count
        return patch


class SagittalIntersection:
    """冻结两侧实测图和标定，仅让当前位姿改变 sagittal 交线的采样位置。"""

    def __init__(self, data, reference_path, calibration_path, *, column=477,
                 world_x_mm=-.659604, sigmas_mm=(.5, .25), window_mm=6., stride_mm=3.,
                 minimum_edge_std=.005, ssim_weight=0., coverage_weight=.2):
        if not hasattr(data, "images"):
            raise ValueError("sagittal 交线监督需要缓存实测灰度图")
        if len(sigmas_mm) != 2 or min(sigmas_mm) <= 0 or not 0 <= column < data.width:
            raise ValueError("要求两个正的物理平滑尺度和有效源图列号")
        device, dtype = data.initial.device, data.initial.dtype
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        with h5py.File(reference_path, "r") as handle:
            reference = np.asarray(handle["data_sag"]).T.astype(np.float32) / 255.
        with np.load(calibration_path, allow_pickle=False) as calibration:
            linear, offset = calibration["linear"].copy(), calibration["offset"].copy()
            structure = calibration["sagittal_structure"].astype(bool)
        if reference.shape != structure.shape or linear.shape != (2, 2) or offset.shape != (2,):
            raise ValueError("sagittal 图像和冻结标定形状不一致")
        if not np.isfinite(reference).all() or reference.min() < 0 or reference.max() > 1:
            raise ValueError("sagittal 必须是 data_sag.T/255 的有限 [0,1] 图像")
        self.pitch_mm = float(1 / np.linalg.norm(linear[0]))
        if not np.allclose(linear @ linear.T, np.eye(2) / self.pitch_mm ** 2, rtol=1e-5):
            raise ValueError("当前物理尺度平滑要求已冻结的 similarity 标定")
        self.linear, self.offset = tensor(linear), tensor(offset)
        self.reference = tensor(reference)
        self.reference_mask = tensor(structure).bool()
        self.maximum_xy = tensor([reference.shape[1] - 1, reference.shape[0] - 1])
        self.frame_ids = data.frame_ids.detach()
        self.training = torch.zeros(len(data.frame_ids), dtype=torch.bool, device=device)
        self.training[data.splits["training"]] = True
        self.splits = {name: indices.detach() for name, indices in data.splits.items()}
        self.ssim_weight, self.coverage_weight = float(ssim_weight), float(coverage_weight)
        self.sigmas_mm = tuple(float(sigma) for sigma in sigmas_mm)
        rows = np.arange(0, data.height, 2)
        self.local = data.local[rows, column].detach()
        self.depth_mm = self.local[:, 0]
        self.sample_spacing_mm = float(2 * data.spacing[0])
        self.world_x_mm = float(world_x_mm)
        initial_world = self.local[None] @ data.initial[:, :3, :3].transpose(-1, -2)
        initial_world = initial_world + data.initial[:, None, :3, 3]
        if float((initial_world[..., 0] - world_x_mm).abs().max()) > 1e-3:
            raise ValueError("源图列不是固定 world-x sagittal 平面上的交线")

        # 两幅图都按相同毫米尺度作二维 Gaussian，再沿同一深度采样计算一维响应。
        # 只传回源图列周围的 Gaussian 支持条带，避免复制整套 GPU 灰度图。
        radius_y, radius_x = [int(np.ceil(4 * max(sigmas_mm) / pitch)) for pitch in data.spacing]
        left, right = max(0, column - radius_x), min(data.width, column + radius_x + 1)
        source_strip = data.images[:, :, left:right].detach().cpu().numpy()
        source = [gaussian_filter(source_strip, sigma=(0, sigma / data.spacing[0], sigma / data.spacing[1]),
                                  mode="nearest")[:, rows, column - left]
                  for sigma in sigmas_mm]
        self.source_gray = tensor(np.stack(source))
        self.source_edge = self.depth_response(self.source_gray)
        source_mask = data.mask.detach().cpu().numpy()
        source_safe = minimum_filter(source_mask, size=(2 * (radius_y + 2) + 1, 2 * radius_x + 1),
                                     mode="constant", cval=0)[rows, column]
        self.source_support = tensor(source_safe).bool()[None].expand(len(self.frame_ids), -1).clone().detach()
        radius = int(np.ceil(4 * max(sigmas_mm) / self.pitch_mm))
        reference_safe = minimum_filter(structure, size=2 * radius + 1, mode="constant", cval=0)
        distance = distance_transform_edt(~reference_safe) * self.pitch_mm
        features = [gaussian_filter(reference, sigma / self.pitch_mm, mode="nearest") for sigma in sigmas_mm]
        self.features = tensor(np.stack(features + [reference_safe.astype(float), distance]))[None]
        length = max(3, int(round(window_mm / self.sample_spacing_mm)))
        stride = max(1, int(round(stride_mm / self.sample_spacing_mm)))
        starts = torch.arange(0, len(rows) - length + 1, stride, device=device)
        self.windows = starts[:, None] + torch.arange(length, device=device)[None]
        if not len(starts):
            raise ValueError("交线短于预定物理窗口")
        with torch.no_grad():
            initial = self.sample(data.initial)
            valid = (initial["valid"] > .999) & tensor(source_safe).bool()[None]
            valid = valid & torch.roll(valid, 1, -1) & torch.roll(valid, -1, -1)
            valid[:, 0] = valid[:, -1] = False
            weight = valid[:, self.windows].all(-1)
            a = self.source_edge[:, :, self.windows]
            b = initial["edge"][:, :, self.windows]
            av, bv = a.var(-1, unbiased=False), b.var(-1, unbiased=False)
            # 支持在初始化时冻结，两个尺度使用共同窗口，不依据后续匹配表现重新挑选。
            weight &= ((av > minimum_edge_std ** 2) & (bv > minimum_edge_std ** 2)).all(0)
            self.window_weight = weight.to(dtype)
            self.window_count = weight.sum(-1)
            self.minimum_variance = (.25 ** 2 * bv).clamp_min((minimum_edge_std / 2) ** 2)
            self.support = torch.zeros_like(valid)
            for index, window in enumerate(self.windows):
                self.support[:, window] |= weight[:, index, None]
            self.support = self.support.to(dtype)
        if not bool((self.window_count[self.training] > 0).any()):
            raise ValueError("没有训练帧具有固定的 sagittal 结构窗口")
        self.training_indices = torch.nonzero(self.training & (self.window_count > 0)).flatten()
        self.metadata = dict(
            reference=str(Path(reference_path).resolve()), calibration=str(Path(calibration_path).resolve()),
            reference_sha256=hash_file(reference_path), calibration_sha256=hash_file(calibration_path),
            reference_normalization="data_sag.T / 255", column_zero_based=column, world_x_mm=world_x_mm,
            sigmas_mm=list(sigmas_mm), sample_spacing_mm=self.sample_spacing_mm,
            window_mm=window_mm, stride_mm=stride_mm, minimum_edge_std=minimum_edge_std,
            window_count=self.window_count.cpu().tolist(), fixed_support_sha256=hash_array(self.support.cpu().numpy()),
            frame_ids=self.frame_ids.cpu().tolist(), estimated_reference_pitch_mm=self.pitch_mm,
            ssim_weight=self.ssim_weight, coverage_weight=self.coverage_weight,
            edge_response="absolute central depth derivative after equal-mm 2D Gaussian; measured images only",
            matching="fixed-window 1-NCC(edge) + optional 1-SSIM(gray) + fixed-support and low-variance penalties",
            calibration_status="frozen train13 initial similarity; sagittal pose and pitch not independently verified")

    def depth_response(self, profile):
        """深度导数单位为 1/mm；首尾没有中心差分，固定支持始终将其排除。"""
        derivative = (profile[..., 2:] - profile[..., :-2]) / (2 * self.sample_spacing_mm)
        return F.pad(derivative.abs(), (1, 1), mode="replicate")

    def sample(self, matrices, frame_indices=None):
        """matrices 始终为全部帧的 [N,4,4] 可微位姿；像素坐标顺序为 (列,行)。"""
        indices = torch.arange(len(self.frame_ids), device=matrices.device) if frame_indices is None else frame_indices
        poses = matrices[indices]
        world = self.local[None] @ poses[:, :3, :3].transpose(-1, -2) + poses[:, None, :3, 3]
        xy = world[..., 1:] @ self.linear + self.offset
        grid = (2 * xy / self.maximum_xy - 1)[None]
        sampled = F.grid_sample(self.features, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        overflow = (torch.relu(-xy) + torch.relu(xy - self.maximum_xy)).sum(-1) * self.pitch_mm
        valid = sampled[2] * (overflow == 0)
        # 深度响应使用相邻采样点，覆盖率也必须覆盖这两个邻点。
        valid = torch.minimum(valid, torch.minimum(torch.roll(valid, 1, -1), torch.roll(valid, -1, -1)))
        return dict(gray=sampled[:2], edge=self.depth_response(sampled[:2]), valid=valid,
                    distance_mm=sampled[3] + overflow, xy=xy)

    def _components(self, matrices, indices, progress):
        stage = 0 if progress < .5 else 1
        sample = self.sample(matrices, indices)
        result = sagittal_profile_losses(
            self.source_gray[stage, indices], sample["gray"][stage],
            self.source_edge[stage, indices], sample["edge"][stage], windows=self.windows,
            window_weight=self.window_weight[indices], support=self.support[indices],
            valid=sample["valid"], distance_mm=sample["distance_mm"],
            minimum_variance=self.minimum_variance[stage, indices],
            ssim_weight=self.ssim_weight, coverage_weight=self.coverage_weight)
        return result, sample, stage

    def loss(self, matrices, frame_indices, progress):
        indices = frame_indices.unique()
        if not bool(self.training[indices].all()):
            raise ValueError("sagittal 训练损失禁止读取 validation 帧")
        result, _, _ = self._components(matrices, indices, progress)
        active = self.window_count[indices] > 0
        return {key: mean_masked(value, active) for key, value in result.items()
                if key != "window_edge_ncc"}

    @torch.no_grad()
    def evaluate(self, matrices, progress=1.):
        indices = torch.arange(len(self.frame_ids), device=matrices.device)
        result, sample, stage = self._components(matrices, indices, progress)
        result.update(frame_ids=self.frame_ids, depth_mm=self.depth_mm, xy=sample["xy"],
                      source_profile=self.source_gray[stage], reference_profile=sample["gray"][stage],
                      source_edge=self.source_edge[stage], reference_edge=sample["edge"][stage],
                      fixed_support=self.support, window_count=self.window_count)
        for key in ("loss", "edge_ncc", "ssim", "coverage"):
            result[key] = result[key].masked_fill(self.window_count == 0, float("nan"))
        arrays = {key: value.detach().cpu().numpy() for key, value in result.items()}
        split = np.full(len(self.frame_ids), "unassigned", dtype="<U16")
        for name, frame_indices in self.splits.items():
            split[frame_indices.cpu().numpy()] = name
        return dict(arrays, split=split, sigma_mm=np.asarray(self.sigmas_mm[stage]))


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
