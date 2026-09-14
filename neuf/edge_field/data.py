from __future__ import annotations

import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_erosion

from neuf.dataset import Dataset
from neuf.phase1_data import build_phase1_split, hash_array, hash_file


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
            if teacher["edge_response"].shape != (self.width, self.height, len(source)):
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
                    images.append(original)
                edges.append(edge)
                matrices.append(matrix)
                alignment.append(dict(frame_id=idx, pixel_max_error=error, response_hash=hash_array(edge)))
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
        self.valid_centers = torch.nonzero(self.interior)
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
                             normalization="original uint8 / 255; saved per-frame p99.5 response [0,1]",
                             coordinate_convention="local=(Y axial,X lateral,0) mm; world=R local+t",
                             teacher_kind="traditional NLSTV lambda=0.009; no neural teacher")
        del dataset, images, edges
        gc.collect()

    def patches(self, batch_size, size, generator):
        train = self.splits["training"]
        frames = train[torch.randint(len(train), (batch_size,), generator=generator).to(train.device)]
        picks = torch.randint(len(self.valid_centers), (batch_size,), generator=generator).to(train.device)
        centers = self.valid_centers[picks]
        rows = (centers[:, 0] - size // 2).clamp(0, self.height - size)
        cols = (centers[:, 1] - size // 2).clamp(0, self.width - size)
        rr = rows[:, None, None] + torch.arange(size, device=train.device)[None, :, None]
        cc = cols[:, None, None] + torch.arange(size, device=train.device)[None, None, :]
        indices = frames[:, None, None]
        patch = dict(frames=frames, local=self.local[rr, cc].reshape(batch_size, -1, 3),
                     edge=self.edges[indices, rr, cc][:, None], mask=self.interior[rr, cc][:, None])
        if hasattr(self, "images"):
            patch["image"] = self.images[indices, rr, cc][:, None]
        return patch


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
