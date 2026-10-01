"""全部标注点的固定重点区域、sagittal 场监督和无标注交线约束。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from scipy.spatial import ConvexHull

from neuf.phase1_data import hash_array, hash_file
from neuf.ultrasound_mask import detect_ultrasound_sector_mask

from .data import SagittalIntersection


class SagittalGuidance(SagittalIntersection):
    """目标、区域、特征与交线窗口仅由初始化数据确定，不随拟合结果筛选。

    reference 为 [H,W] 灰度，roi/core_roi/extension_roi/reference_mask 为
    同设备的 bool 张量；xyz_grid 为 [H,W,3] 毫米坐标，field_bounds 为
    [2,3] 毫米包围盒。标注文件采用 MATLAB 一基坐标，在读取时减一。
    """

    def __init__(self, data, reference_path, calibration_path, landmarks_path, margin_mm=10.,
                 landmark_tolerance_mm=0.):
        if not np.isfinite(margin_mm) or margin_mm <= 0:
            raise ValueError("ROI 外扩距离必须为正的有限毫米数")
        if not np.isfinite(landmark_tolerance_mm) or landmark_tolerance_mm < 0:
            raise ValueError("标注容差必须为非负有限毫米数")
        self.landmark_tolerance_mm = float(landmark_tolerance_mm)
        with h5py.File(reference_path, "r") as handle:
            reference = np.asarray(handle["data_sag"]).T.astype(np.float32) / 255.
        if reference.shape != (708, 944):
            raise ValueError("此标定要求原生 708×944 sagittal 图，禁止缩放后复用像素标注")
        annotations = loadmat(landmarks_path)
        source = np.asarray(annotations["coord_pts_img_seqdyn"], dtype=np.float64) - 1
        target = np.asarray(annotations["coord_pts_img_ref"], dtype=np.float64) - 1
        if source.shape != (11, 2) or target.shape != (11, 2) or not np.isfinite(np.r_[source, target]).all():
            raise ValueError("要求 11 对有限的源帧/深度与 sagittal 列/行坐标")
        physical_mask = detect_ultrasound_sector_mask(reference, safety_margin_px=2).mask
        pixel = np.rint(target).astype(int)
        height, width = reference.shape
        if ((pixel < 0).any() or (pixel[:, 0] >= width).any() or (pixel[:, 1] >= height).any()
                or not physical_mask[pixel[:, 1], pixel[:, 0]].all()):
            raise ValueError("至少一个标注点不在真实 sagittal 扇区，不能静默丢弃标注")
        # 不读取旧 sagittal_structure，保留过去被排除的第 8、9 点。
        super().__init__(data, reference_path, calibration_path, reference_mask=physical_mask)
        self.data, self.column = data, 477
        tensor = lambda value: torch.as_tensor(value, device=self.reference.device, dtype=self.reference.dtype)
        self.source_landmarks, self.target_landmarks = tensor(source), tensor(target)
        self.landmark_frames, self.landmark_target = self.source_landmarks[:, 0], self.target_landmarks
        self.landmark_local = tensor(np.zeros((len(source), 3)))
        self.landmark_local[:, 0] = (self.source_landmarks[:, 1] + .5) * data.spacing[0]
        self.landmark_local[:, 1] = data.local[0, self.column, 1]
        if (bool(((self.source_landmarks[:, 0] < data.frame_ids[0])
                  | (self.source_landmarks[:, 0] > data.frame_ids[-1])).any())
                or bool(((self.source_landmarks[:, 1] < -.5)
                         | (self.source_landmarks[:, 1] > data.height - .5)).any())):
            raise ValueError("源标注超出原始帧或深度范围")

        yy, xx = np.mgrid[:height, :width]
        coordinates = np.stack((xx, yy), -1)
        equations = ConvexHull(target).equations
        core = (coordinates @ equations[:, :2].T + equations[:, 2] <= 1e-7).all(-1)
        core[pixel[:, 1], pixel[:, 0]] = True
        distance = distance_transform_edt(~core) * self.pitch_mm
        roi = (distance <= margin_mm) & physical_mask
        core &= physical_mask
        self.roi, self.core_roi = tensor(roi).bool(), tensor(core).bool()
        self.extension_roi = self.roi & ~self.core_roi
        xy = tensor(coordinates)
        yz = (xy - self.offset) @ torch.linalg.inv(self.linear)
        self.xyz_grid = torch.cat((torch.full_like(yz[..., :1], self.world_x_mm), yz), -1)
        # 额外保留 3 mm 滤波上下文与 2 mm 包围盒余量，避免 HashGrid 在 ROI 附近截断坐标。
        context = tensor((distance <= margin_mm + 3.) & physical_mask).bool()
        context_world = self.xyz_grid[context]
        self.field_bounds = torch.stack((torch.minimum(data.bounds[0], context_world.amin(0) - 2.),
                                         torch.maximum(data.bounds[1], context_world.amax(0) + 2.)))

        with torch.no_grad():
            initial = self.sample(data.initial)
            grid = (2 * initial["xy"] / self.maximum_xy - 1)[None]
            within_roi = F.grid_sample(self.roi[None, None].float(), grid, align_corners=True)[0, 0] > .999
            self.window_weight *= within_roi[:, self.windows].all(-1)
            self.window_count = self.window_weight.sum(-1)
            self.support.zero_()
            for index, window in enumerate(self.windows):
                self.support[:, window] = torch.maximum(self.support[:, window], self.window_weight[:, index, None])
            self.training_indices = torch.nonzero(self.training & (self.window_count > 0)).flatten()
        if not len(self.training_indices):
            raise ValueError("全点 ROI 内没有初始化固定的有效训练交线窗口")

        # Shi–Tomasi 角点与梯度仅来自实采参考图；强边缘提高邻域权重。
        smooth = gaussian_filter(reference, .25 / self.pitch_mm)
        gy, gx = np.gradient(smooth)
        a, b, c = [gaussian_filter(value, .5 / self.pitch_mm) for value in (gx * gx, gy * gy, gx * gy)]
        corner = .5 * (a + b - np.sqrt((a - b) ** 2 + 4 * c ** 2))
        magnitude = np.hypot(gx, gy)
        feature_domain = roi & (distance_transform_edt(physical_mask) * self.pitch_mm >= 1.)
        separation = max(3, 2 * int(round(1.5 / self.pitch_mm)) + 1)
        candidates = (corner == maximum_filter(corner, separation)) & feature_domain
        candidates &= corner > max(float(np.quantile(corner[feature_domain], .90)), 1e-10)
        locations = np.argwhere(candidates)
        if not len(locations):
            raise ValueError("固定 ROI 内未提取到可用 sagittal 角点")
        locations = locations[np.argsort(corner[locations[:, 0], locations[:, 1]])[::-1][:256]].copy()
        self.corner_centers = torch.as_tensor(locations, dtype=torch.long)
        self.uniform_centers = torch.as_tensor(np.argwhere(roi), dtype=torch.long)
        corner_scale = max(float(np.quantile(corner[feature_domain], .99)), 1e-10)
        edge_scale = max(float(np.quantile(magnitude[feature_domain], .95)), 1e-10)
        weights = 1 + 2 * np.clip(magnitude / edge_scale, 0, 1) + 2 * np.clip(corner / corner_scale, 0, 1)
        self.feature_weights = tensor(weights)
        self.source_raw = data.images[:, ::2, self.column].detach()
        self.metadata.update(
            kind="all-point sagittal field and trajectory guidance", landmarks=str(Path(landmarks_path).resolve()),
            landmarks_sha256=hash_file(landmarks_path), fit_landmark_ids=list(range(1, 12)), diagnostic_landmark_ids=[],
            landmark_loss="mean vector Huber outside reference-location tolerance, delta=1 estimated mm; all 11 points",
            landmark_tolerance_mm=self.landmark_tolerance_mm,
            landmark_interpretation="approximate localization references, not exact anatomical ground truth",
            landmark_source_convention="zero-based fractional original frame and row; axial=(row+0.5)*pitch",
            margin_mm=float(margin_mm), roi_definition="all 11 target-point convex hull dilated in mm, intersect true fan",
            reference_mask_definition="detect_ultrasound_sector_mask(data_sag.T / 255), safety_margin_px=2",
            roi_sha256=hash_array(roi), core_roi_sha256=hash_array(core),
            roi_pixels=int(roi.sum()), core_pixels=int(core.sum()), extension_pixels=int((roi & ~core).sum()),
            all_landmarks_in_roi=bool(roi[pixel[:, 1], pixel[:, 0]].all()),
            field_bounds_mm=self.field_bounds.cpu().tolist(), field_bounds_context_mm=3., field_bounds_padding_mm=2.,
            corner_locations_xy=locations[:, ::-1].tolist(), corner_count=len(locations),
            field_sampling="half uniform ROI centers, half frozen Shi-Tomasi corners; mean over valid window centers",
            feature_weights="1 + 2*clipped gradient/p95 + 2*clipped corner/p99; fixed measured reference",
            window_count=self.window_count.cpu().tolist(), fixed_support_sha256=hash_array(self.support.cpu().numpy()),
            dense_pose_loss="equal-scale 1D measured/model profiles: .4 gray NCC + .4 edge NCC + .2 L1; fixed 6 mm windows",
            main_metric_scope="sagittal is used for training; reconstruction metrics are not held-out generalization")

    def landmark_projection(self, poses):
        world = poses.fractional_world(self.source_landmarks[:, 0], self.landmark_local)
        return world[:, 1:] @ self.linear + self.offset

    def landmark_loss(self, poses):
        distance = torch.linalg.vector_norm(self.landmark_projection(poses) - self.target_landmarks, dim=-1) * self.pitch_mm
        # 容差内不追逐手工点；容差外保留 Huber 的有界影响，不把参考点当精确真值。
        distance = (distance - self.landmark_tolerance_mm).clamp_min(0)
        return torch.where(distance <= 1, .5 * distance.square(), distance - .5).mean()

    @torch.no_grad()
    def landmark_rows(self, poses):
        source, target = self.source_landmarks.cpu().numpy(), self.target_landmarks.cpu().numpy()
        projected = self.landmark_projection(poses).cpu().numpy()
        error = np.linalg.norm(projected - target, axis=-1)
        return [dict(id=index + 1, source_frame=float(source[index, 0]), source_row=float(source[index, 1]),
                     target_x=float(target[index, 0]), target_y=float(target[index, 1]),
                     projected_x=float(projected[index, 0]), projected_y=float(projected[index, 1]),
                     error_px=float(error[index]), error_mm=float(error[index] * self.pitch_mm), is_fit=True)
                for index in range(len(source))]

    def field_patches(self, model, progress, generator, batch_size=4, size=40):
        """返回 prediction/reference/support/feature_weights，均为 [B,1,h,w]。"""
        height, width = self.reference.shape
        if batch_size < 1 or size < 3 or size > min(height, width):
            raise ValueError("sagittal 图块尺寸或数量无效")
        corners = batch_size // 2
        centers = self.uniform_centers[torch.randint(len(self.uniform_centers), (batch_size,), generator=generator)].clone()
        centers[:corners] = self.corner_centers[torch.randint(len(self.corner_centers), (corners,), generator=generator)]
        centers = centers.to(self.reference.device)
        rows = (centers[:, 0] - size // 2).clamp(0, height - size)
        cols = (centers[:, 1] - size // 2).clamp(0, width - size)
        rr = rows[:, None, None] + torch.arange(size, device=rows.device)[None, :, None]
        cc = cols[:, None, None] + torch.arange(size, device=rows.device)[None, None, :]
        prediction = model(self.xyz_grid[rr, cc], progress=progress)[..., 0][:, None]
        return prediction, self.reference[rr, cc][:, None], self.roi[rr, cc][:, None], self.feature_weights[rr, cc][:, None]

    def dense_pose_loss(self, model, poses, progress, frame_indices=None):
        """冻结网络参数时保留坐标导数；无标注帧同样受实测交线结构约束。"""
        indices = self.training_indices if frame_indices is None else frame_indices.unique()
        if not bool(self.training[indices].all()):
            raise ValueError("NeRF 交线监督禁止读取 validation 源帧")
        local = self.local[None].expand(len(indices), -1, -1)
        predicted = model(poses(indices, local), progress=progress)[..., 0]
        sigma = self.sigmas_mm[0 if progress < .5 else 1] / self.sample_spacing_mm
        radius = max(1, int(np.ceil(4 * sigma)))
        axis = torch.arange(-radius, radius + 1, dtype=predicted.dtype, device=predicted.device)
        kernel = torch.exp(-.5 * (axis / sigma).square())
        kernel = (kernel / kernel.sum())[None, None]
        smooth = lambda values: F.conv1d(F.pad(values[:, None], (radius, radius), mode="replicate"), kernel)[:, 0]
        predicted, measured = smooth(predicted), smooth(self.source_raw[indices])
        weights = self.window_weight[indices]
        active = (weights.sum(-1) > 0).to(predicted.dtype)
        if not bool(active.any()):
            raise ValueError("所选源帧没有固定有效交线窗口")

        def ncc(a, b):
            a, b = a[:, self.windows], b[:, self.windows]
            a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
            score = ((a * b).mean(-1) / ((a.square().mean(-1) + 1e-10)
                                                * (b.square().mean(-1) + 1e-10)).sqrt()).clamp(-1, 1)
            per_frame = (score * weights).sum(-1) / weights.sum(-1).clamp_min(1)
            return (per_frame * active).sum() / active.sum().clamp_min(1)

        gray_ncc = ncc(predicted, measured)
        edge_ncc = ncc(self.depth_response(predicted), self.depth_response(measured))
        support = self.support[indices]
        photo = ((predicted - measured).abs() * support).sum() / support.sum().clamp_min(1)
        present = active.sum().clamp_max(1)
        loss = .4 * (present - gray_ncc) + .4 * (present - edge_ncc) + .2 * photo
        return dict(loss=loss, gray_ncc=gray_ncc, edge_ncc=edge_ncc, photo=photo)
