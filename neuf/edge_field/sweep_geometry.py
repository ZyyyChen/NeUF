"""联合重建时学习同一旋转平面内的角度或帧间角速度。"""

from __future__ import annotations

import math

import torch
from torch import nn


def _rotation_x(angle):
    c, s, z = angle.cos(), angle.sin(), torch.zeros_like(angle)
    return torch.stack((torch.ones_like(angle), z, z, z, c, -s, z, s, c), -1).reshape(-1, 3, 3)


class SweepPoseRefiner(nn.Module):
    """仅修改绕世界 x 轴的旋转；原探头旋转中心及其漂移保持固定。

    initial 为 [N,4,4]，局部坐标为 (axial,lateral,0) mm，轴向位于
    世界 YZ 平面。angle 参数是训练帧角度修正，velocity 参数是每原始帧
    的角增量修正；没有真实时间戳，不能将后者解释为 rad/s。两者均有
    n_training-1 个自由度，锚定首训练帧，留出帧仅按帧号插值角度修正。
    """

    def __init__(self, initial, train_indices, frame_ids, radial_offset_mm, mode="angle", *,
                 angle_scale_deg=1., velocity_scale_deg=None, prior_angle_deg=3.,
                 prior_velocity_deg=None):
        super().__init__()
        initial = torch.as_tensor(initial).detach().clone()
        ids = torch.as_tensor(frame_ids, dtype=torch.long, device=initial.device)
        train = torch.as_tensor(train_indices, dtype=torch.long, device=initial.device)
        if initial.ndim != 3 or initial.shape[1:] != (4, 4) or not initial.is_floating_point():
            raise ValueError("initial 必须为浮点 [N,4,4] 位姿")
        if ids.shape != (len(initial),) or len(ids) < 2 or not bool((ids.diff() > 0).all()):
            raise ValueError("frame_ids 必须与 initial 一一对应并严格递增")
        if train.ndim != 1 or len(train) < 2 or bool(((train < 0) | (train >= len(ids))).any()):
            raise ValueError("至少需要两个有效训练帧")
        train = train.sort().values
        if not bool((train.diff() > 0).all()) or mode not in ("angle", "velocity"):
            raise ValueError("训练索引不得重复，mode 必须为 angle 或 velocity")
        if not bool(torch.isfinite(initial).all()) or not math.isfinite(float(radial_offset_mm)):
            raise ValueError("位姿及 radial_offset_mm 必须有限")
        if float(radial_offset_mm) < 0:
            raise ValueError("radial_offset_mm 不得为负")
        # 明确拒绝不符合 cerebral 原坐标约定的输入，避免在另一平面静默优化。
        if bool((initial[:, 0, 0].abs() > 1e-5).any()) or bool((initial[:, 1:3, 1].abs() > 1e-5).any()):
            raise ValueError("要求局部 axial 位于世界 YZ 平面，lateral 平行世界 x 轴")
        centers = initial[:, :3, 3] - float(radial_offset_mm) * initial[:, :3, 0]
        if bool(((centers[:, 0] - centers[0, 0]).abs() > 1e-4).any()):
            raise ValueError("原旋转中心须处于同一个世界 x 平面")
        theta = torch.atan2(initial[:, 1, 0], -initial[:, 2, 0])
        differences = theta.diff()
        theta = torch.cat((theta[:1], theta[:1] + torch.atan2(differences.sin(), differences.cos()).cumsum(0)))
        gaps = ids.diff().to(initial.dtype)
        omega = theta.diff() / gaps
        baseline_speed = float((theta[-1] - theta[0]) / (ids[-1] - ids[0]))
        if abs(baseline_speed) < 1e-8 or bool((omega * baseline_speed <= 0).any()):
            raise ValueError("初始轨迹须为同向旋转且角速度非零")
        velocity_scale_deg = math.degrees(abs(baseline_speed)) if velocity_scale_deg is None else velocity_scale_deg
        prior_velocity_deg = math.degrees(abs(baseline_speed)) if prior_velocity_deg is None else prior_velocity_deg
        scales = (angle_scale_deg, velocity_scale_deg, prior_angle_deg, prior_velocity_deg)
        if any(not math.isfinite(float(value)) or value <= 0 for value in scales):
            raise ValueError("参数及先验的角度尺度必须为正的有限值")
        self.mode = mode
        for name, value in dict(initial=initial, frame_ids=ids, train_indices=train,
                                initial_angles=theta, initial_increments=omega, frame_gaps=gaps,
                                centers=centers, radial_offset=initial.new_tensor(radial_offset_mm),
                                direction=initial.new_tensor(math.copysign(1., baseline_speed)),
                                angle_scale=initial.new_tensor(math.radians(angle_scale_deg)),
                                velocity_scale=initial.new_tensor(math.radians(velocity_scale_deg)),
                                prior_angle_scale=initial.new_tensor(math.radians(prior_angle_deg)),
                                prior_velocity_scale=initial.new_tensor(math.radians(prior_velocity_deg)),
                                training_gaps=ids[train].diff().to(initial.dtype)).items():
            self.register_buffer(name, value)
        # 插值权重仅使用帧号；端点外保持最近训练帧的修正，不读取留出图像。
        right = torch.searchsorted(ids[train], ids).clamp(1, len(train) - 1)
        left = right - 1
        alpha = ((ids - ids[train[left]]).to(initial.dtype)
                 / (ids[train[right]] - ids[train[left]]).to(initial.dtype)).clamp(0, 1)
        self.register_buffer("interpolation_left", left)
        self.register_buffer("interpolation_right", right)
        self.register_buffer("interpolation_weight", alpha)
        # 每段使用最小原始角速度，使训练角度投影也保证插值后的留出帧不回摆。
        minimum_speed = torch.stack([(omega[a:b] * self.direction).amin()
                                     for a, b in zip(train[:-1].tolist(), train[1:].tolist())])
        allowance = minimum_speed * self.training_gaps * (1 - 1e-4)
        self.register_buffer("monotonic_base", torch.cat((allowance.new_zeros(1), allowance.cumsum(0))))
        self.raw = nn.Parameter(initial.new_zeros(len(train) - 1))
        self.metadata = dict(
            mode=mode, parameters=len(train) - 1, angle_units="radian", velocity_units="radian/original_frame",
            rotation_axis_world=[1., 0., 0.], radial_offset_mm=float(radial_offset_mm),
            fixed_geometry="initial rotation centers including original drift; world x rotation plane",
            anchor_frame_id=int(ids[train[0]]), endpoint_locked=False,
            heldout_policy="linear interpolation of training angle corrections by original frame id; constant outside training span",
            angle_scale_deg=float(angle_scale_deg), velocity_scale_deg_per_frame=float(velocity_scale_deg),
            prior_angle_deg=float(prior_angle_deg), prior_velocity_deg_per_frame=float(prior_velocity_deg),
            parameter_mapping="linear correction; shared bounded monotonic projection in training-angle space after optimizer step",
            projection="intersect initial-offset and previous-angle step bounds; monotone bound envelopes then forward cumulative maximum; re-encode raw",
        )

    def _offsets(self, raw):
        training = raw * self.angle_scale if self.mode == "angle" else (raw * self.velocity_scale * self.training_gaps).cumsum(0)
        training = torch.cat((raw.new_zeros(1), training))
        weight = self.interpolation_weight
        return training[self.interpolation_left] * (1 - weight) + training[self.interpolation_right] * weight

    def angle_offsets(self):
        """全部帧的角度修正 [N]，单位 rad。"""
        return self._offsets(self.raw)

    def angles(self):
        return self.initial_angles + self.angle_offsets()

    def increments(self):
        """相邻已加载帧区间的平均角增量 [N-1]，单位 rad/原始帧。"""
        return self.initial_increments + self.angle_offsets().diff() / self.frame_gaps

    def matrices(self):
        rotation = _rotation_x(self.angle_offsets()) @ self.initial[:, :3, :3]
        result = self.initial.clone()
        result[:, :3, :3] = rotation
        # 等价于 centers + radius * 新轴向，差量形式使零修正精确复现原平移。
        result[:, :3, 3] = self.initial[:, :3, 3] + self.radial_offset * (rotation[:, :, 0] - self.initial[:, :3, 0])
        return result

    def forward(self, frame_indices, local_points):
        matrices = self.matrices()[frame_indices]
        return torch.einsum("bij,bpj->bpi", matrices[:, :3, :3], local_points) + matrices[:, None, :3, 3]

    def fractional_world(self, frame_ids, local_points):
        """小数原始帧号上的 [M,3] 或 [M,P,3] 局部点，毫米单位。

        角度与旋转中心分别插值，再构造正交旋转；不把弧上的两点或旋转
        矩阵线性混合，避免改变标注点到探头旋转中心的物理半径。
        """
        ids = torch.as_tensor(frame_ids, device=self.initial.device, dtype=self.initial.dtype)
        local = torch.as_tensor(local_points, device=self.initial.device, dtype=self.initial.dtype)
        if ids.ndim != 1 or local.ndim not in (2, 3) or local.shape[0] != len(ids) or local.shape[-1] != 3:
            raise ValueError("要求 frame_ids=[M]、local_points=[M,3] 或 [M,P,3]")
        if not bool(torch.isfinite(ids).all()) or bool(((ids < self.frame_ids[0]) | (ids > self.frame_ids[-1])).any()):
            raise ValueError("小数帧号必须位于已加载帧号范围内")
        right = torch.searchsorted(self.frame_ids.to(ids.dtype), ids).clamp(1, len(self.frame_ids) - 1)
        left = right - 1
        alpha = (ids - self.frame_ids[left]) / (self.frame_ids[right] - self.frame_ids[left])
        angles = self.angles()
        angle = angles[left] * (1 - alpha) + angles[right] * alpha
        rotation = _rotation_x(angle - self.initial_angles[left]) @ self.initial[left, :3, :3]
        center = self.centers[left] * (1 - alpha[:, None]) + self.centers[right] * alpha[:, None]
        translation = center + self.radial_offset * rotation[:, :, 0]
        if local.ndim == 2:
            return torch.einsum("bij,bj->bi", rotation, local) + translation
        return torch.einsum("bij,bpj->bpi", rotation, local) + translation[:, None]

    def prior_components(self):
        """两种参数化都在同一物理角度/速度尺度下正则化，避免 raw 尺度偏置。"""
        offset = self.angle_offsets()
        speed = self.increments()
        speed_delta = (speed - self.initial_increments) / self.prior_velocity_scale
        acceleration = speed_delta.diff()
        return dict(angle_offset=(offset / self.prior_angle_scale).square().mean(),
                    velocity_offset=speed_delta.square().mean(),
                    velocity_smooth=acceleration.square().mean() if len(acceleration) else speed.sum() * 0,
                    reversal=torch.relu(-self.direction * speed / self.prior_velocity_scale).square().mean())

    def prior(self, train_indices=None, frame_ids=None):
        # 索引已在构造时固定；保留既有训练入口的调用签名。
        return sum(self.prior_components().values())

    @torch.no_grad()
    def limit_update_(self, previous_raw, max_angle_step_deg=.01, max_angle_offset_deg=3.):
        """在训练角度域共同限制实际步幅、总修正和回摆，再编码为原参数。

        采用有界单调投影，单个区间碰到边界不会冻结其余帧。调用方须在
        optimizer.step 前 clone raw。accepted_fraction 是实际/提议最大角度
        步幅之比，投影可能改变方向，并非整次 raw 更新的缩放比例。
        """
        if max_angle_step_deg <= 0 or max_angle_offset_deg <= 0:
            raise ValueError("角度步幅及总修正上限必须为正")
        before = self._offsets(previous_raw)
        proposed = self.angle_offsets()
        old = self.direction * before[self.train_indices]
        candidate = self.direction * proposed[self.train_indices]
        limit = math.radians(max_angle_offset_deg)
        step = math.radians(max_angle_step_deg)
        lower = (old - step).clamp_min(-limit) + self.monotonic_base
        upper = (old + step).clamp_max(limit) + self.monotonic_base
        lower[0] = upper[0] = 0
        # 单调上下包络保证后面的累积最大值不会越过未来节点的上界。
        lower = lower.cummax(0).values
        upper = upper.flip(0).cummin(0).values.flip(0)
        candidate = torch.maximum(torch.minimum(candidate + self.monotonic_base, upper), lower)
        offset = self.direction * (candidate.cummax(0).values - self.monotonic_base)
        if self.mode == "angle":
            self.raw.copy_(offset[1:] / self.angle_scale)
        else:
            self.raw.copy_(offset.diff() / (self.training_gaps * self.velocity_scale))
        actual = torch.rad2deg(self.angle_offsets() - before)
        proposed_max = torch.rad2deg(proposed - before).abs().amax().clamp_min(torch.finfo(before.dtype).tiny)
        return dict(accepted_fraction=float((actual.abs().amax() / proposed_max).clamp_max(1)),
                    actual_angle_step_max_deg=float(actual.abs().amax()),
                    actual_angle_step_rms_deg=float(actual.square().mean().sqrt()))


def load_sweep(checkpoint, device="cpu"):
    """仅依赖已保存的 checkpoint 恢复角度轨迹，不重新读取原始数据。"""
    metadata, state = checkpoint["sweep_metadata"], checkpoint["poses"]
    model = SweepPoseRefiner(
        state["initial"].to(device), state["train_indices"].to(device), state["frame_ids"].to(device),
        metadata["radial_offset_mm"], metadata["mode"], angle_scale_deg=metadata["angle_scale_deg"],
        velocity_scale_deg=metadata["velocity_scale_deg_per_frame"], prior_angle_deg=metadata["prior_angle_deg"],
        prior_velocity_deg=metadata["prior_velocity_deg_per_frame"],
    )
    model.load_state_dict(state, strict=True)
    return model


def smoke_check(device="cpu"):
    """供已有 qsub smoke 调用：零修正、两参数化等价、共面、留出插值及梯度。"""
    ids = torch.tensor([0, 1, 2, 4, 5, 8], device=device)
    train = torch.tensor([0, 2, 4, 5], device=device)
    theta = -.4 + .005 * ids.to(torch.float64)
    zero = torch.zeros_like(theta)
    rotation = torch.stack((zero, -torch.ones_like(theta), zero,
                            theta.sin(), zero, theta.cos(), -theta.cos(), zero, theta.sin()), -1).reshape(-1, 3, 3)
    initial = torch.eye(4, dtype=theta.dtype, device=device).repeat(len(ids), 1, 1)
    initial[:, :3, :3] = rotation
    centers = torch.stack((zero, 50 + .1 * ids, 4 - .02 * ids), -1)
    initial[:, :3, 3] = centers + 3.4 * rotation[:, :, 0]
    models = [SweepPoseRefiner(initial, train, ids, 3.4, mode) for mode in ("angle", "velocity")]
    for model in models:
        torch.testing.assert_close(model.matrices(), initial, atol=0, rtol=0)
    with torch.no_grad():
        models[0].raw.copy_(theta.new_tensor([.2, .5, .7]))
        training_offset = models[0].angle_offsets()[train]
        models[1].raw.copy_(training_offset.diff() / models[1].training_gaps / models[1].velocity_scale)
    torch.testing.assert_close(models[0].angle_offsets(), models[1].angle_offsets())
    torch.testing.assert_close(models[0].prior(), models[1].prior())
    torch.testing.assert_close(models[0].matrices(), models[1].matrices())
    for model in models:
        offsets = model.angle_offsets()
        torch.testing.assert_close(offsets[1], offsets[2] / 2)
        torch.testing.assert_close(offsets[3], offsets[2] / 3 + offsets[4] * 2 / 3)
        torch.testing.assert_close(model.matrices()[:, 0], initial[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(model.matrices()[:, :3, 3] - 3.4 * model.matrices()[:, :3, 0], centers)
        local = theta.new_tensor([[[10., 2., 0.]]]).expand(len(train), -1, -1)
        world = model(train, local)
        world[..., 1].sum().backward()
        assert bool(torch.isfinite(model.raw.grad).all()) and bool((model.raw.grad.abs() > 0).all())
        previous = model.raw.detach().clone()
        with torch.no_grad():
            model.raw.add_(100)
        statistics = model.limit_update_(previous)
        assert statistics["actual_angle_step_max_deg"] <= .01000001
        assert float(torch.rad2deg(model.angle_offsets()).abs().max()) <= 3.000001
        assert float((model.direction * model.increments()).min()) >= -1e-10
        torch.testing.assert_close(model.matrices()[train[0]], initial[train[0]], atol=0, rtol=0)
    return dict(initial_exact=True, shared_plane=True, gradients=True, heldout_interpolation=True,
                parameterizations_equivalent=True, physical_angle_step_limit_deg=.01)
