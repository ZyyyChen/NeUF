from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class PoseRefiner(nn.Module):
    """Learn a right-composed SE(3) correction for every training slice.

    NeUF stores probe poses as local-to-world transforms.  Following BARF, a
    six-dimensional Lie-algebra increment is initialized to zero and composed
    with each tracked pose.  Translation values use NeUF's native millimetre
    coordinate system; rotation values are in radians.
    """

    def __init__(
        self,
        base_rotations: torch.Tensor,
        base_translations: torch.Tensor,
        *,
        anchor_first: bool = True,
    ) -> None:
        super().__init__()
        if base_rotations.ndim != 3 or base_rotations.shape[-2:] != (3, 3):
            raise ValueError(
                "base_rotations must have shape [num_poses, 3, 3], "
                f"got {tuple(base_rotations.shape)}"
            )
        if base_translations.shape != (base_rotations.shape[0], 3):
            raise ValueError(
                "base_translations must have shape [num_poses, 3], "
                f"got {tuple(base_translations.shape)}"
            )
        if base_rotations.shape[0] == 0:
            raise ValueError("PoseRefiner requires at least one training pose")

        self.anchor_first = bool(anchor_first)
        self.register_buffer("base_rotations", base_rotations.detach().float().clone())
        self.register_buffer("base_translations", base_translations.detach().float().clone())
        self.se3_refine = nn.Embedding(base_rotations.shape[0], 6)
        nn.init.zeros_(self.se3_refine.weight)

    @classmethod
    def from_slices(
        cls,
        slices: Sequence,
        *,
        device: torch.device | str,
        anchor_first: bool = True,
    ) -> "PoseRefiner":
        if not slices:
            raise ValueError("Cannot optimize poses without training slices")

        rotations = np.stack(
            [np.asarray(slice_info.rotation.as_rotmat(), dtype=np.float32) for slice_info in slices]
        )
        translations = np.stack(
            [np.asarray(slice_info.position, dtype=np.float32) for slice_info in slices]
        )
        return cls(
            torch.from_numpy(rotations).to(device),
            torch.from_numpy(translations).to(device),
            anchor_first=anchor_first,
        ).to(device)

    @property
    def num_poses(self) -> int:
        return int(self.base_rotations.shape[0])

    @staticmethod
    def _skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
        x, y, z = vector.unbind(dim=-1)
        zero = torch.zeros_like(x)
        return torch.stack(
            (
                torch.stack((zero, -z, y), dim=-1),
                torch.stack((z, zero, -x), dim=-1),
                torch.stack((-y, x, zero), dim=-1),
            ),
            dim=-2,
        )

    @staticmethod
    def _taylor_coefficient(
        squared_angle: torch.Tensor,
        coefficient: str,
        terms: int = 10,
    ) -> torch.Tensor:
        """Stable Taylor series used by the SO(3)/SE(3) exponential map."""
        result = torch.zeros_like(squared_angle)
        power = torch.ones_like(squared_angle)
        denominator = 1.0

        for index in range(terms + 1):
            if coefficient == "A":
                if index > 0:
                    denominator *= (2 * index) * (2 * index + 1)
            elif coefficient == "B":
                denominator *= (2 * index + 1) * (2 * index + 2)
            elif coefficient == "C":
                denominator *= (2 * index + 2) * (2 * index + 3)
            else:
                raise ValueError(f"Unknown Taylor coefficient: {coefficient}")

            result = result + ((-1.0) ** index) * power / denominator
            power = power * squared_angle

        return result

    @classmethod
    def se3_to_transform(cls, se3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [..., 6] Lie-algebra vectors to rotation and translation."""
        if se3.shape[-1] != 6:
            raise ValueError(f"se3 must end in dimension 6, got {tuple(se3.shape)}")

        rotation_vector, translation_vector = se3.split((3, 3), dim=-1)
        skew = cls._skew_symmetric(rotation_vector)
        squared_angle = torch.sum(rotation_vector * rotation_vector, dim=-1)[..., None, None]
        identity = torch.eye(3, dtype=se3.dtype, device=se3.device)
        identity = identity.expand(se3.shape[:-1] + (3, 3))

        coefficient_a = cls._taylor_coefficient(squared_angle, "A")
        coefficient_b = cls._taylor_coefficient(squared_angle, "B")
        coefficient_c = cls._taylor_coefficient(squared_angle, "C")
        skew_squared = skew @ skew

        rotation = identity + coefficient_a * skew + coefficient_b * skew_squared
        translation_jacobian = identity + coefficient_b * skew + coefficient_c * skew_squared
        translation = (translation_jacobian @ translation_vector[..., None]).squeeze(-1)
        return rotation, translation

    def corrections(self, frame_indices: torch.Tensor | int | None = None) -> torch.Tensor:
        if frame_indices is None:
            corrections = self.se3_refine.weight
            if not self.anchor_first:
                return corrections
            anchor_mask = torch.ones(
                (self.num_poses, 1),
                dtype=corrections.dtype,
                device=corrections.device,
            )
            anchor_mask[0] = 0
            return corrections * anchor_mask

        frame_indices = torch.as_tensor(
            frame_indices,
            dtype=torch.long,
            device=self.se3_refine.weight.device,
        )
        corrections = self.se3_refine(frame_indices)
        if self.anchor_first:
            corrections = corrections * (frame_indices != 0).unsqueeze(-1).to(corrections.dtype)
        return corrections

    def forward(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor,
        frame_indices: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply refined poses to points/view directions generated by base poses."""
        if points.shape != viewdirs.shape or points.shape[-1] != 3:
            raise ValueError(
                "points and viewdirs must have the same shape ending in 3, "
                f"got {tuple(points.shape)} and {tuple(viewdirs.shape)}"
            )

        original_shape = points.shape
        flat_points = points.reshape(-1, 3)
        flat_viewdirs = viewdirs.reshape(-1, 3)
        flat_indices = torch.as_tensor(
            frame_indices,
            dtype=torch.long,
            device=flat_points.device,
        ).reshape(-1)
        if flat_indices.numel() == 1 and flat_points.shape[0] != 1:
            flat_indices = flat_indices.expand(flat_points.shape[0])
        if flat_indices.numel() != flat_points.shape[0]:
            raise ValueError(
                "frame_indices must contain one index per point (or a scalar), "
                f"got {flat_indices.numel()} for {flat_points.shape[0]} points"
            )

        unique_indices, inverse_indices = torch.unique(
            flat_indices,
            sorted=True,
            return_inverse=True,
        )
        unique_base_rotation = self.base_rotations[unique_indices].to(flat_points.dtype)
        unique_base_translation = self.base_translations[unique_indices].to(flat_points.dtype)
        correction = self.corrections(unique_indices).to(flat_points.dtype)
        unique_delta_rotation, unique_delta_translation = self.se3_to_transform(correction)

        base_rotation = unique_base_rotation[inverse_indices]
        base_translation = unique_base_translation[inverse_indices]
        delta_rotation = unique_delta_rotation[inverse_indices]
        delta_translation = unique_delta_translation[inverse_indices]

        # Recover probe-local coordinates, apply the BARF increment there, then
        # map back with the tracked local-to-world pose: T_refined = T_base @ dT.
        local_points = torch.bmm(
            base_rotation.transpose(1, 2),
            (flat_points - base_translation).unsqueeze(-1),
        ).squeeze(-1)
        local_viewdirs = torch.bmm(
            base_rotation.transpose(1, 2),
            flat_viewdirs.unsqueeze(-1),
        ).squeeze(-1)
        refined_local_points = torch.bmm(delta_rotation, local_points.unsqueeze(-1)).squeeze(-1)
        refined_local_points = refined_local_points + delta_translation
        refined_local_viewdirs = torch.bmm(
            delta_rotation,
            local_viewdirs.unsqueeze(-1),
        ).squeeze(-1)

        refined_points = torch.bmm(
            base_rotation,
            refined_local_points.unsqueeze(-1),
        ).squeeze(-1) + base_translation
        refined_viewdirs = torch.bmm(
            base_rotation,
            refined_local_viewdirs.unsqueeze(-1),
        ).squeeze(-1)
        return refined_points.reshape(original_shape), refined_viewdirs.reshape(original_shape)

    def refined_poses(self) -> torch.Tensor:
        """Return all corrected local-to-world poses with shape [N, 3, 4]."""
        delta_rotation, delta_translation = self.se3_to_transform(self.corrections())
        rotation = self.base_rotations @ delta_rotation
        translation = self.base_translations + (
            self.base_rotations @ delta_translation.unsqueeze(-1)
        ).squeeze(-1)
        return torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)

    def regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        correction = self.corrections()
        rotation = torch.mean(torch.sum(correction[:, :3] ** 2, dim=-1))
        _, translation = self.se3_to_transform(correction)
        translation = torch.mean(torch.sum(translation ** 2, dim=-1))
        return rotation, translation

    def trajectory_regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return first- and second-order smoothness of ordered SE(3) increments.

        The training slices are stored in trajectory order.  Penalizing differences
        between their six-dimensional corrections prevents independently optimized
        poses from explaining image noise with frame-to-frame pose oscillations.
        Rotation components are in radians and translation components are in
        millimetres, so callers control their joint scale through the loss weights.
        """
        correction = self.corrections()
        zero = correction.sum() * 0.0
        if self.num_poses < 2:
            return zero, zero

        velocity = correction[1:] - correction[:-1]
        velocity_loss = torch.mean(torch.sum(velocity ** 2, dim=-1))
        if self.num_poses < 3:
            return velocity_loss, zero

        acceleration = velocity[1:] - velocity[:-1]
        acceleration_loss = torch.mean(torch.sum(acceleration ** 2, dim=-1))
        return velocity_loss, acceleration_loss

    @torch.no_grad()
    def statistics(self) -> dict[str, float]:
        correction = self.corrections()
        rotation_degrees = torch.rad2deg(
            torch.linalg.vector_norm(correction[:, :3], dim=-1)
        )
        _, translation = self.se3_to_transform(correction)
        translation_mm = torch.linalg.vector_norm(translation, dim=-1)
        return {
            "rotation_mean_deg": float(rotation_degrees.mean().cpu()),
            "rotation_max_deg": float(rotation_degrees.max().cpu()),
            "translation_mean_mm": float(translation_mm.mean().cpu()),
            "translation_max_mm": float(translation_mm.max().cpu()),
        }
