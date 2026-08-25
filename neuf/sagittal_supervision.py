from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

from neuf.pose_refinement import PoseRefiner
from neuf.ultrasound_mask import SECTOR_MASK_VERSION, detect_ultrasound_sector_mask


def load_matlab_image(
    mat_path: str | Path,
    variable_name: str,
    *,
    expected_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load one 2-D image from either a classic or v7.3 MATLAB file.

    MATLAB v7.3 stores array dimensions in the reverse order exposed by h5py,
    so those arrays are transposed once while loading.  ``expected_shape`` is
    then used as a final orientation check against the NeUF training images.
    """
    path = Path(mat_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Sagittal MATLAB file not found: {path}")

    try:
        with h5py.File(path, "r") as mat_file:
            if variable_name not in mat_file:
                available = sorted(key for key in mat_file.keys() if not key.startswith("#"))
                raise KeyError(
                    f"Variable '{variable_name}' is missing from {path}; "
                    f"available variables: {available}"
                )
            image = np.asarray(mat_file[variable_name]).squeeze().T
    except OSError:
        mat_file = loadmat(path)
        if variable_name not in mat_file:
            available = sorted(key for key in mat_file if not key.startswith("__"))
            raise KeyError(
                f"Variable '{variable_name}' is missing from {path}; "
                f"available variables: {available}"
            )
        image = np.asarray(mat_file[variable_name]).squeeze()

    if image.ndim != 2:
        raise ValueError(
            f"Sagittal variable '{variable_name}' must be 2-D after squeezing, "
            f"got shape {tuple(image.shape)}"
        )

    if expected_shape is not None and image.shape != expected_shape:
        if image.T.shape == expected_shape:
            image = image.T
        else:
            expected_height, expected_width = expected_shape
            crop_candidates = [
                candidate
                for candidate in (image, image.T)
                if candidate.shape[0] >= expected_height
                and candidate.shape[1] >= expected_width
            ]
            if crop_candidates:
                # NeUF trims only trailing black image rows, so keep the same
                # shallow/top and lateral/left origin when matching sagittal data.
                image = min(
                    crop_candidates,
                    key=lambda candidate: candidate.size,
                )[:expected_height, :expected_width]
            else:
                raise ValueError(
                    "Sagittal image shape does not match the NeUF slice grid: "
                    f"image={tuple(image.shape)}, expected={expected_shape}"
                )

    image = image.astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Sagittal image contains non-finite values: {path}")

    image_min = float(image.min())
    image_max = float(image.max())
    if image_min < 0:
        raise ValueError(
            f"Sagittal image intensities must be non-negative, got min={image_min:g}"
        )
    if image_max > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def find_central_training_slice(dataset) -> int:
    """Find the tracked training pose nearest the middle of the scan."""
    if not getattr(dataset, "slices", None):
        raise ValueError("Cannot initialize sagittal supervision without training slices")

    positions = np.stack(
        [np.asarray(slice_info.position, dtype=np.float32) for slice_info in dataset.slices]
    )
    scan_axis = getattr(dataset, "scan_axis", None)
    front = getattr(dataset, "front_plane_point", None)
    back = getattr(dataset, "back_plane_point", None)
    if scan_axis is not None and front is not None and back is not None:
        axis = np.asarray(scan_axis, dtype=np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 0:
            axis = axis / axis_norm
            center = 0.5 * (
                np.asarray(front, dtype=np.float32)
                + np.asarray(back, dtype=np.float32)
            )
            distances = np.abs((positions - center) @ axis)
            return int(np.argmin(distances))

    # Legacy baked datasets may predate scan-axis metadata.  Their slices are
    # still stored in trajectory order (historically reversed), so the middle
    # element remains the appropriate centre initialization.
    return len(dataset.slices) // 2


class SagittalSliceSupervisor(nn.Module):
    """Auxiliary sagittal image and its independently refinable centre pose."""

    def __init__(
        self,
        *,
        target: torch.Tensor,
        base_points: torch.Tensor,
        base_viewdirs: torch.Tensor,
        base_rotation: torch.Tensor,
        base_translation: torch.Tensor,
        height: int,
        width: int,
        source_path: str | Path,
        variable_name: str,
        initial_slice_index: int,
        optimize_pose: bool,
        valid_mask: Optional[torch.Tensor] = None,
        valid_mask_source: str = "full-frame-acquisition",
    ) -> None:
        super().__init__()
        point_count = int(height) * int(width)
        if target.shape != (point_count, 1):
            raise ValueError(
                f"target must have shape ({point_count}, 1), got {tuple(target.shape)}"
            )
        if base_points.shape != (point_count, 3):
            raise ValueError(
                f"base_points must have shape ({point_count}, 3), "
                f"got {tuple(base_points.shape)}"
            )
        if base_viewdirs.shape != base_points.shape:
            raise ValueError(
                "base_viewdirs must match base_points, got "
                f"{tuple(base_viewdirs.shape)} and {tuple(base_points.shape)}"
            )

        self.height = int(height)
        self.width = int(width)
        self.source_path = str(Path(source_path).expanduser())
        self.variable_name = str(variable_name)
        self.initial_slice_index = int(initial_slice_index)
        self.optimize_pose = bool(optimize_pose)
        if valid_mask is None:
            valid_mask = torch.ones((height, width), dtype=torch.bool, device=target.device)
        valid_mask = torch.as_tensor(
            valid_mask,
            dtype=torch.bool,
            device=target.device,
        ).reshape(height, width)
        if not torch.any(valid_mask):
            raise ValueError("Sagittal ultrasound sector mask contains no valid pixels")

        target = target.detach().float().clone()
        target = torch.where(
            valid_mask.reshape(-1, 1),
            target,
            torch.zeros_like(target),
        )
        self.register_buffer("target", target)
        self.register_buffer("valid_mask", valid_mask)
        self.register_buffer(
            "valid_flat_indices",
            torch.nonzero(valid_mask.reshape(-1), as_tuple=False).reshape(-1),
        )
        self.valid_mask_source = str(valid_mask_source)
        self.valid_mask_version = SECTOR_MASK_VERSION
        self.register_buffer("base_points", base_points.detach().float().clone())
        self.register_buffer("base_viewdirs", base_viewdirs.detach().float().clone())
        self.pose_refiner = PoseRefiner(
            base_rotation.detach().float().reshape(1, 3, 3),
            base_translation.detach().float().reshape(1, 3),
            anchor_first=False,
        )
        self.pose_refiner.requires_grad_(self.optimize_pose)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        mat_path: str | Path,
        *,
        variable_name: str = "data_sag",
        optimize_pose: bool = True,
        device: torch.device | str,
    ) -> "SagittalSliceSupervisor":
        height = int(dataset.px_height)
        width = int(dataset.px_width)
        image = load_matlab_image(
            mat_path,
            variable_name,
            expected_shape=(height, width),
        )
        mask_detection = detect_ultrasound_sector_mask(image)
        valid_mask = torch.from_numpy(mask_detection.mask).to(
            device=device,
            dtype=torch.bool,
        )
        initial_slice_index = find_central_training_slice(dataset)
        slice_info = dataset.slices[initial_slice_index]

        base_points = dataset.get_slice_points(initial_slice_index).reshape(-1, 3)
        base_viewdirs = dataset.get_slice_viewdirs(initial_slice_index).reshape(-1, 3)
        base_rotation = torch.as_tensor(
            np.asarray(slice_info.rotation.as_rotmat(), dtype=np.float32),
            device=device,
        )
        base_translation = torch.as_tensor(
            np.asarray(slice_info.position, dtype=np.float32),
            device=device,
        )
        return cls(
            target=torch.from_numpy(image).to(device).reshape(-1, 1),
            base_points=base_points.to(device),
            base_viewdirs=base_viewdirs.to(device),
            base_rotation=base_rotation,
            base_translation=base_translation,
            height=height,
            width=width,
            source_path=mat_path,
            variable_name=variable_name,
            initial_slice_index=initial_slice_index,
            optimize_pose=optimize_pose,
            valid_mask=valid_mask,
            valid_mask_source=mask_detection.source,
        ).to(device)

    @property
    def point_count(self) -> int:
        return int(self.target.shape[0])

    @property
    def valid_point_count(self) -> int:
        return int(self.valid_flat_indices.numel())

    def valid_mask_signature(self) -> dict:
        mask_bytes = self.valid_mask.detach().cpu().numpy().astype(np.uint8).tobytes()
        return {
            "version": int(self.valid_mask_version),
            "source": self.valid_mask_source,
            "shape_hw": [self.height, self.width],
            "valid_pixels": self.valid_point_count,
            "total_pixels": self.point_count,
            "valid_fraction": self.valid_point_count / self.point_count,
            "sha256": hashlib.sha256(mask_bytes).hexdigest(),
        }

    def pose_parameters(self):
        return self.pose_refiner.parameters()

    def refined_geometry(
        self,
        indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points = self.base_points if indices is None else self.base_points[indices]
        viewdirs = self.base_viewdirs if indices is None else self.base_viewdirs[indices]
        return self.pose_refiner(points, viewdirs, 0)

    def sample(
        self,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_count <= 0:
            raise ValueError(f"sample_count must be >= 1, got {sample_count}")
        valid_ranks = torch.randint(
            self.valid_point_count,
            (min(int(sample_count), self.valid_point_count),),
            dtype=torch.long,
            device=self.target.device,
        )
        indices = self.valid_flat_indices[valid_ranks]
        points, viewdirs = self.refined_geometry(indices)
        return self.target[indices], points, viewdirs

    def target_image(self) -> torch.Tensor:
        return self.target.reshape(self.height, self.width)

    def refined_pose(self) -> torch.Tensor:
        return self.pose_refiner.refined_poses()[0]
