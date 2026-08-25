from __future__ import annotations

import json
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import crop
from tqdm import tqdm

from neuf.ultrasound_mask import (
    SECTOR_MASK_VERSION,
    detect_ultrasound_sector_mask,
    evenly_spaced_sample_indices,
)
from neuf.utils import get_base_points, get_oriented_points_and_views

# Datasets created before the package migration were pickled as
# ``dataset.Dataset``. Keep that module name resolvable while loading them.
sys.modules.setdefault("dataset", sys.modules[__name__])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_REQUIRED_KEYS = {"x", "y", "width", "height"}
DEFAULT_SPLIT_SEED = 17081998
PHYSICAL_METADATA_VERSION = 2
DATASET_METADATA_VERSION = 3
CM_TO_MM = 10.0


@dataclass(frozen=True)
class DatasetLoadConfig:
    folder: Path
    nb_valid: int
    seed: int
    name: str
    img_folder: str
    info_folder: str
    prefix: str
    suffix: str
    reverse_quat: bool
    y2z: bool
    exclude_valid: bool
    image_step: int
    gt_folder: Optional[str]
    gt_prefix: str
    gt_suffix: str


@dataclass(frozen=True)
class PhysicalCalibration:
    width_mm: float
    height_mm: float
    pixel_width_mm: float
    pixel_height_mm: float
    source: str


def resolve_physical_calibration(
    infos: dict,
    *,
    image_width_px: int,
    image_height_px: int,
) -> PhysicalCalibration:
    """Resolve physical image calibration, preferring the current metadata.

    ``px_size_cm`` is authoritative when present. ``scan_dims_mm`` is retained
    only as a compatibility fallback for datasets that have not yet acquired
    per-pixel calibration metadata.
    """
    pixel_size_cm = infos.get("px_size_cm")
    if pixel_size_cm is not None:
        try:
            pixel_width_mm = float(pixel_size_cm["width"]) * CM_TO_MM
            pixel_height_mm = float(pixel_size_cm["height"]) * CM_TO_MM
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "px_size_cm must contain numeric 'width' and 'height' values"
            ) from error
        if pixel_width_mm <= 0 or pixel_height_mm <= 0:
            raise ValueError(
                "px_size_cm values must be positive, got "
                f"{pixel_size_cm!r}"
            )
        return PhysicalCalibration(
            width_mm=pixel_width_mm * int(image_width_px),
            height_mm=pixel_height_mm * int(image_height_px),
            pixel_width_mm=pixel_width_mm,
            pixel_height_mm=pixel_height_mm,
            source="px_size_cm",
        )

    scan_dims = infos.get("scan_dims_mm", {})
    try:
        width_mm = float(scan_dims.get("width", 0.0))
        height_mm = float(scan_dims.get("depth", scan_dims.get("height", 0.0)))
    except (TypeError, ValueError) as error:
        raise ValueError("scan_dims_mm must contain numeric dimensions") from error
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(
            "Physical calibration is missing: provide positive px_size_cm "
            "width/height values (preferred) or legacy scan_dims_mm width/depth values"
        )
    return PhysicalCalibration(
        width_mm=width_mm,
        height_mm=height_mm,
        pixel_width_mm=width_mm / int(image_width_px),
        pixel_height_mm=height_mm / int(image_height_px),
        source="scan_dims_mm_legacy",
    )


@dataclass
class FrameRecord:
    frame_index: int
    position: np.ndarray
    rotation: "Quat"
    image: torch.Tensor
    gt: Optional[torch.Tensor]
    points: torch.Tensor
    points_numpy: np.ndarray
    viewdirs: torch.Tensor


@dataclass
class Slice:
    start: int
    end: int
    position: np.ndarray
    rotation: "Quat"
    frame_index: Optional[int] = None


class Dataset:
    def __init__(self, folder, nb_valid=4, seed=-1, **kwargs):
        config = self._build_config(folder, nb_valid, seed, kwargs)
        self._initialize_state(config)
        self._apply_random_seed(config.seed)

        infos_json = self._load_infos(config)
        frame_keys = self._get_selected_frame_keys(infos_json, config.image_step)
        self._initialize_sector_mask(config, frame_keys)
        frames = self._load_frames(config, infos_json, frame_keys)

        if not frames:
            raise ValueError(f"No frames were loaded from dataset: {config.folder}")

        point_extent = self._update_point_bounds_and_scan_metadata(frames)
        train_records, valid_records = self._split_train_and_validation_frames(frames, config.nb_valid)
        self._materialize_records(train_records, valid_records)
        self._finalize_pixel_geometry()
        self._print_loading_summary(config.folder, len(frames), point_extent)

    def _build_config(self, folder, nb_valid, seed, kwargs) -> DatasetLoadConfig:
        image_prefix = kwargs.get("prefix", "us/img_")
        image_suffix = kwargs.get("suffix", ".jpg")

        return DatasetLoadConfig(
            folder=Path(folder),
            nb_valid=int(nb_valid),
            seed=int(seed),
            name=kwargs.get("name", os.path.basename(folder)),
            img_folder=kwargs.get("img_folder", "us"),
            info_folder=kwargs.get("info_folder", ""),
            prefix=image_prefix,
            suffix=image_suffix,
            reverse_quat=kwargs.get("reverse_quat", False),
            y2z=kwargs.get("y2z", False),
            exclude_valid=kwargs.get("exclude_valid", True),
            image_step=int(kwargs.get("image_step", 1)),
            gt_folder=kwargs.get("gt_folder"),
            gt_prefix=kwargs.get("gt_prefix", image_prefix),
            gt_suffix=kwargs.get("gt_suffix", image_suffix),
        )

    def _initialize_state(self, config: DatasetLoadConfig) -> None:
        self.metadata_version = DATASET_METADATA_VERSION
        self.pixel_calibration_source = "unknown"
        self.metadata_migrated_on_load = False
        self.width = 0.0
        self.height = 0.0
        self.px_width = 0
        self.px_height = 0
        self.orig_px_width = 0
        self.orig_px_height = 0
        self.orig_px_size_width_mm = 0.0
        self.orig_px_size_height_mm = 0.0
        self.roi_px_size_width_mm = 0.0
        self.roi_px_size_height_mm = 0.0
        self.point_min = np.zeros(3, dtype=np.float32)
        self.point_max = np.full(3, 100.0, dtype=np.float32)
        self.point_min_dev = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        self.point_max_dev = torch.zeros(3, dtype=torch.float32, device=DEVICE)

        self.slices: list[Slice] = []
        self.slices_valid: list[Slice] = []
        self.X = np.array([], dtype=np.float32)
        self.Y = np.array([], dtype=np.float32)
        self.name = config.name
        self.has_gt = False
        self.image_value_scale = 1.0
        self.roi_2d = None
        self.roi_offset_x_mm = 0.0
        self.roi_offset_y_mm = 0.0
        self.front_plane_point = None
        self.front_plane_normal = None
        self.back_plane_point = None
        self.back_plane_normal = None
        self.scan_axis = None
        self.scan_length_mm = 0.0
        self.image_step = config.image_step
        self.exclude_valid = config.exclude_valid
        self.reverse_quat = config.reverse_quat
        self.y2z = config.y2z
        self.R_y2z = Rotation.from_matrix(
            np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        self.nb_valid = config.nb_valid
        self.infos_json_path = ""
        self.trimmed_px_height = 0
        self.sector_mask_version = SECTOR_MASK_VERSION
        self.sector_mask_source = "uninitialized"
        self.sector_mask_sampled_frames = 0
        self.sector_mask_threshold = 0.0
        self.sector_mask = torch.empty((0, 0), dtype=torch.bool, device=DEVICE)
        self.sector_valid_flat_indices = torch.empty(
            (0,), dtype=torch.long, device=DEVICE
        )
        self._valid_patch_origins_cache: dict[int, torch.Tensor] = {}

    def _apply_random_seed(self, seed: int) -> None:
        effective_seed = DEFAULT_SPLIT_SEED if seed == -1 else seed
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)

    def _load_infos(self, config: DatasetLoadConfig) -> dict:
        first_image_path = self._build_frame_path(
            config.folder,
            config.img_folder,
            config.prefix,
            0,
            config.suffix,
        )
        first_image_raw = read_image(str(first_image_path), ImageReadMode.GRAY)
        self.orig_px_height, self.orig_px_width = first_image_raw.shape[1], first_image_raw.shape[2]
        print(f"Original image size: {self.orig_px_width} x {self.orig_px_height} px")

        infos_path = config.folder / config.info_folder / "infos.json"
        self.infos_json_path = str(infos_path)
        with infos_path.open("r") as infos_file:
            infos_json = json.load(infos_file)

        infos = infos_json["infos"]
        calibration = resolve_physical_calibration(
            infos,
            image_width_px=self.orig_px_width,
            image_height_px=self.orig_px_height,
        )
        self.pixel_calibration_source = calibration.source
        orig_width_mm = calibration.width_mm
        orig_height_mm = calibration.height_mm
        self.orig_px_size_width_mm = calibration.pixel_width_mm
        self.orig_px_size_height_mm = calibration.pixel_height_mm
        print(
            f"Using {calibration.source} calibration: "
            f"{calibration.pixel_width_mm:.6f} x "
            f"{calibration.pixel_height_mm:.6f} mm/px"
        )
        print(f"Original physical size: {orig_width_mm:.2f} x {orig_height_mm:.2f} mm")

        self._configure_roi(infos, orig_width_mm, orig_height_mm)
        return infos_json

    def _configure_roi(self, infos: dict, orig_width_mm: float, orig_height_mm: float) -> None:
        roi = infos.get("ROI")
        if roi and roi.get("width", 0) > 1 and roi.get("height", 0) > 1:
            if not ROI_REQUIRED_KEYS.issubset(roi.keys()):
                raise ValueError(f"ROI must contain keys {ROI_REQUIRED_KEYS}")

            self.roi_2d = roi
            print(
                f"ROI config: x={self.roi_2d['x']}, y={self.roi_2d['y']}, "
                f"w={self.roi_2d['width']}, h={self.roi_2d['height']}"
            )

        if self.roi_2d and orig_width_mm > 0 and orig_height_mm > 0:
            self.width = self.roi_2d["width"] * self.orig_px_size_width_mm
            self.height = self.roi_2d["height"] * self.orig_px_size_height_mm
            self.roi_offset_x_mm = self.roi_2d["x"] * self.orig_px_size_width_mm
            self.roi_offset_y_mm = self.roi_2d["y"] * self.orig_px_size_height_mm
            self.roi_px_size_width_mm = self.orig_px_size_width_mm
            self.roi_px_size_height_mm = self.orig_px_size_height_mm

            print("ROI cropping applied:")
            print(f"  Cropped physical size: {self.width:.2f} x {self.height:.2f} mm")
            print(f"  Physical offset: x={self.roi_offset_x_mm:.2f}mm, y={self.roi_offset_y_mm:.2f}mm")
            return

        self.width = orig_width_mm
        self.height = orig_height_mm
        self.roi_px_size_width_mm = self.orig_px_size_width_mm
        self.roi_px_size_height_mm = self.orig_px_size_height_mm
        print("ROI cropping not applied")

    def _get_selected_frame_keys(self, infos_json: dict, image_step: int) -> list[str]:
        if image_step <= 0:
            raise ValueError(f"image_step must be >= 1, got {image_step}")

        frame_keys = sorted((key for key in infos_json.keys() if key != "infos"), key=lambda key: int(key))
        return frame_keys[::image_step]

    def _initialize_sector_mask(
        self,
        config: DatasetLoadConfig,
        frame_keys: list[str],
    ) -> None:
        """Detect the acquisition fan before any image becomes a NeUF sample."""
        sampled_images = []
        sample_indices = evenly_spaced_sample_indices(len(frame_keys))
        for frame_list_index in tqdm(
            sample_indices,
            desc="Detecting ultrasound valid sector",
        ):
            key = frame_keys[int(frame_list_index)]
            path = self._build_frame_path(
                config.folder, config.img_folder, config.prefix, key, config.suffix
            )
            img = read_image(str(path), ImageReadMode.GRAY)
            if self.roi_2d:
                img = crop(
                    img,
                    self.roi_2d["y"], self.roi_2d["x"],
                    self.roi_2d["height"], self.roi_2d["width"],
                )
            sampled_images.append(torch.squeeze(img).cpu().numpy())

        detection = detect_ultrasound_sector_mask(np.stack(sampled_images, axis=0))
        self.sector_mask = torch.as_tensor(
            detection.mask,
            dtype=torch.bool,
            device=DEVICE,
        )
        self.sector_mask_source = detection.source
        self.sector_mask_sampled_frames = detection.sampled_frames
        self.sector_mask_threshold = detection.foreground_threshold
        self.trimmed_px_height = int(detection.mask.shape[0])
        self._finalize_sector_mask()
        print(
            "Mandatory ultrasound sector mask: "
            f"source={detection.source}, valid={detection.valid_fraction:.2%}, "
            f"sampled_frames={detection.sampled_frames}"
        )

    def _load_frames(self, config: DatasetLoadConfig, infos_json: dict, frame_keys: list[str]) -> list[FrameRecord]:
        frames: list[FrameRecord] = []

        for frame_index, frame_key in enumerate(
            tqdm(frame_keys, desc="Opening dataset", total=len(frame_keys))
        ):
            frame = infos_json[frame_key]
            position = np.array(
                [float(frame["x"]), float(frame["y"]), float(frame["z"])],
                dtype=np.float32,
            )
            if self.y2z:
                position = self.R_y2z.apply(position).astype(np.float32)
            rotation = self._parse_quaternion(frame, config.reverse_quat)

            image_path = self._build_frame_path(
                config.folder,
                config.img_folder,
                config.prefix,
                frame_key,
                config.suffix,
            )
            image = self.get_torch_image(str(image_path))
            gt = self._load_ground_truth(config, frame_key)

            self._ensure_consistent_image_shape(image)
            self._ensure_base_grid()

            points_numpy, viewdirs_numpy = get_oriented_points_and_views(
                self.X,
                self.Y,
                position,
                rotation,
            )

            frames.append(
                FrameRecord(
                    frame_index=frame_index,
                    position=position,
                    rotation=rotation,
                    image=torch.reshape(image, (-1,)),
                    gt=None if gt is None else torch.reshape(gt, (-1,)),
                    points=torch.from_numpy(points_numpy.astype(np.float32)).to(DEVICE),
                    points_numpy=points_numpy,
                    viewdirs=torch.from_numpy(viewdirs_numpy.astype(np.float32)).to(DEVICE),
                )
            )

        return frames

    def _parse_quaternion(self, frame: dict, reverse_quat: bool) -> "Quat":
        if reverse_quat:
            return Quat(
                float(frame["w3"]),
                float(frame["w0"]),
                float(frame["w1"]),
                float(frame["w2"]),
            )

        return Quat(
            float(frame["w0"]),
            float(frame["w1"]),
            float(frame["w2"]),
            float(frame["w3"]),
        )

    def _load_ground_truth(self, config: DatasetLoadConfig, frame_key: str) -> Optional[torch.Tensor]:
        gt_path = self._build_gt_path(config, frame_key)
        if gt_path is None:
            return None

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground-truth frame is missing: {gt_path}")

        self.has_gt = True
        return self.get_torch_image(str(gt_path))

    def _build_gt_path(self, config: DatasetLoadConfig, frame_key: str) -> Optional[Path]:
        if not config.gt_folder:
            return None

        return self._build_frame_path(
            config.folder,
            config.gt_folder,
            config.gt_prefix,
            frame_key,
            config.gt_suffix,
        )

    def _build_frame_path(
        self,
        folder: Path,
        subfolder: str,
        prefix: str,
        frame_key,
        suffix: str,
    ) -> Path:
        return folder / subfolder / f"{prefix}{frame_key}{suffix}"

    def _ensure_consistent_image_shape(self, image: torch.Tensor) -> None:
        image_height, image_width = int(image.shape[0]), int(image.shape[1])
        if (
            (self.px_width and self.px_width != image_width)
            or (self.px_height and self.px_height != image_height)
        ):
            raise ValueError("Images must have consistent dimensions")

        self.px_width = image_width
        self.px_height = image_height

    def _ensure_base_grid(self) -> None:
        if self.X.size != 0 or self.Y.size != 0:
            return

        self.X, self.Y = get_base_points(
            self.width,
            self.height,
            self.px_width,
            self.px_height,
            offset_x_mm=self.roi_offset_x_mm,
            offset_y_mm=self.roi_offset_y_mm,
        )

    def _update_point_bounds_and_scan_metadata(self, frames: list[FrameRecord]) -> np.ndarray:
        valid_flat = self.sector_mask.detach().cpu().numpy().reshape(-1)
        point_mins = np.stack(
            [frame.points_numpy[valid_flat].min(axis=0) for frame in frames], axis=0
        )
        point_maxs = np.stack(
            [frame.points_numpy[valid_flat].max(axis=0) for frame in frames], axis=0
        )
        self.point_min = point_mins.min(axis=0).astype(np.float32)
        self.point_max = point_maxs.max(axis=0).astype(np.float32)
        self.point_min_dev = torch.as_tensor(self.point_min, dtype=torch.float32, device=DEVICE)
        self.point_max_dev = torch.as_tensor(self.point_max, dtype=torch.float32, device=DEVICE)

        positions = [frame.position for frame in frames]
        rotations = [frame.rotation for frame in frames]
        self._update_scan_metadata(positions, rotations)
        return self.point_max - self.point_min

    def _update_scan_metadata(self, positions: list[np.ndarray], rotations: list["Quat"]) -> None:
        if len(positions) < 2:
            return

        self.front_plane_point = np.array(positions[0], dtype=np.float32)
        self.back_plane_point = np.array(positions[-1], dtype=np.float32)
        scan_vec = self.back_plane_point - self.front_plane_point
        self.scan_length_mm = float(np.linalg.norm(scan_vec))

        if self.scan_length_mm > 0:
            self.scan_axis = scan_vec / self.scan_length_mm
        else:
            self.scan_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        mid_point = np.array(positions[len(positions) // 2], dtype=np.float32)
        front_normal = np.asarray(rotations[0].as_rotmat()[:, 2], dtype=np.float32)
        back_normal = np.asarray(rotations[-1].as_rotmat()[:, 2], dtype=np.float32)

        front_normal = self._normalize_vector(front_normal)
        back_normal = self._normalize_vector(back_normal)

        if np.dot(mid_point - self.front_plane_point, front_normal) < 0:
            front_normal = -front_normal
        if np.dot(mid_point - self.back_plane_point, back_normal) < 0:
            back_normal = -back_normal

        self.front_plane_normal = front_normal
        self.back_plane_normal = back_normal

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector
        return vector / norm

    def _split_train_and_validation_frames(
        self,
        frames: list[FrameRecord],
        nb_valid: int,
    ) -> tuple[list[FrameRecord], list[FrameRecord]]:
        if nb_valid < 0:
            raise ValueError(f"nb_valid must be >= 0, got {nb_valid}")

        nb_valid = min(nb_valid, len(frames))
        if nb_valid == 0:
            valid_indices: set[int] = set()
        else:
            valid_indices = set(np.random.choice(len(frames), nb_valid, replace=False).tolist())

        train_records: list[FrameRecord] = []
        valid_records: list[FrameRecord] = []

        # Keep the historical reverse ordering so old assumptions about slice index
        # direction still hold, but ensure metadata and tensor storage stay aligned.
        for frame in reversed(frames):
            if frame.frame_index in valid_indices:
                valid_records.append(frame)
                if not self.exclude_valid:
                    train_records.append(frame)
            else:
                train_records.append(frame)

        if not train_records:
            raise ValueError("Dataset split produced zero training slices")

        return train_records, valid_records

    def _materialize_records(
        self,
        train_records: list[FrameRecord],
        valid_records: list[FrameRecord],
    ) -> None:
        self.slices = self._build_slices(train_records)
        self.slices_valid = self._build_slices(valid_records)

        self.pixels = self._flatten_record_tensors(train_records, "image")
        self.points = self._concat_record_tensors(train_records, "points")
        self.viewdirs = self._concat_record_tensors(train_records, "viewdirs")

        self.pixels_valid = self._flatten_record_tensors(valid_records, "image")
        self.points_valid = self._concat_record_tensors(valid_records, "points")
        self.viewdirs_valid = self._concat_record_tensors(valid_records, "viewdirs")

        if self.has_gt:
            self.gt = self._flatten_record_tensors(train_records, "gt")
            self.gt_valid = self._flatten_record_tensors(valid_records, "gt")

        self._ensure_unit_intensity_range()

    def _build_slices(self, records: list[FrameRecord]) -> list[Slice]:
        pixels_per_slice = self.px_width * self.px_height
        slices: list[Slice] = []
        for record_index, record in enumerate(records):
            start = record_index * pixels_per_slice
            end = (record_index + 1) * pixels_per_slice
            slices.append(
                Slice(
                    start=int(start),
                    end=int(end),
                    position=record.position,
                    rotation=record.rotation,
                    frame_index=record.frame_index,
                )
            )
        return slices

    def _flatten_record_tensors(self, records: list[FrameRecord], attribute: str) -> torch.Tensor:
        if not records:
            return torch.empty((0,), dtype=torch.float32, device=DEVICE)

        tensors = []
        for record in records:
            value = getattr(record, attribute)
            if value is None:
                raise ValueError(f"Record attribute '{attribute}' is missing")
            tensors.append(value)

        return torch.flatten(torch.stack(tensors))

    def _concat_record_tensors(self, records: list[FrameRecord], attribute: str) -> torch.Tensor:
        if not records:
            return torch.empty((0, 3), dtype=torch.float32, device=DEVICE)

        tensors = [getattr(record, attribute) for record in records]
        return torch.cat(tensors, dim=0)

    def _finalize_pixel_geometry(self) -> None:
        if self.px_width > 0:
            self.roi_px_size_width_mm = self.width / self.px_width
        if self.px_height > 0:
            self.roi_px_size_height_mm = self.height / self.px_height

    def _finalize_sector_mask(self) -> None:
        existing_mask = self.sector_mask
        if isinstance(existing_mask, torch.Tensor):
            mask_device = existing_mask.device
        else:
            mask_device = getattr(getattr(self, "pixels", None), "device", DEVICE)
        mask = torch.as_tensor(existing_mask, dtype=torch.bool, device=mask_device)
        if mask.ndim != 2:
            raise ValueError(f"sector_mask must have shape [H, W], got {tuple(mask.shape)}")
        if self.px_height and self.px_width and tuple(mask.shape) != (
            int(self.px_height),
            int(self.px_width),
        ):
            raise ValueError(
                "sector_mask shape does not match dataset images: "
                f"mask={tuple(mask.shape)}, image={(self.px_height, self.px_width)}"
            )
        valid_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
        if valid_indices.numel() == 0:
            raise ValueError("Mandatory ultrasound sector mask contains no valid pixels")
        self.sector_mask = mask
        self.sector_valid_flat_indices = valid_indices
        self.sector_mask_version = SECTOR_MASK_VERSION
        self._valid_patch_origins_cache = {}

    def _print_loading_summary(self, folder: Path, frame_count: int, point_extent: np.ndarray) -> None:
        print("\n=== Dataset loading summary ===")
        print(f"Dataset path: {folder}")
        print(f"Image count: {frame_count}")
        print(f"Final image size: {self.px_width} x {self.px_height} px")
        print("Image intensity range: [0, 1]")
        print(
            "Valid ultrasound sector: "
            f"{self.valid_pixel_count:,}/{self.pixels_per_slice:,} pixels "
            f"({self.valid_pixel_fraction:.2%}); source={self.sector_mask_source}"
        )
        print(f"Final physical size: {self.width:.2f} x {self.height:.2f} mm")
        print(
            "ROI pixel size: "
            f"{self.roi_px_size_width_mm:.4f} x {self.roi_px_size_height_mm:.4f} mm/px"
        )
        print("Point range:")
        print(f"  X: {self.point_min[0]:.2f} ~ {self.point_max[0]:.2f} mm")
        print(f"  Y: {self.point_min[1]:.2f} ~ {self.point_max[1]:.2f} mm")
        print(f"  Z: {self.point_min[2]:.2f} ~ {self.point_max[2]:.2f} mm")
        print("Bounding box size:")
        print(f"  X: {point_extent[0]:.2f} mm")
        print(f"  Y: {point_extent[1]:.2f} mm")
        print(f"  Z: {point_extent[2]:.2f} mm")

        if self.front_plane_point is None or self.back_plane_point is None:
            return

        print("Scan clipping planes:")
        print(
            f"  Front plane point: ({self.front_plane_point[0]:.2f}, "
            f"{self.front_plane_point[1]:.2f}, {self.front_plane_point[2]:.2f}) mm"
        )
        print(
            f"  Back plane point: ({self.back_plane_point[0]:.2f}, "
            f"{self.back_plane_point[1]:.2f}, {self.back_plane_point[2]:.2f}) mm"
        )
        print(
            f"  Front slice-plane normal (inward): ({self.front_plane_normal[0]:.6f}, "
            f"{self.front_plane_normal[1]:.6f}, {self.front_plane_normal[2]:.6f})"
        )
        print(
            f"  Back slice-plane normal (inward): ({self.back_plane_normal[0]:.6f}, "
            f"{self.back_plane_normal[1]:.6f}, {self.back_plane_normal[2]:.6f})"
        )
        print(
            f"  Scan axis: ({self.scan_axis[0]:.6f}, "
            f"{self.scan_axis[1]:.6f}, {self.scan_axis[2]:.6f})"
        )
        print(f"  Scan length: {self.scan_length_mm:.2f} mm")

    def get_torch_image(self, img_path: str) -> torch.Tensor:
        image = read_image(img_path, ImageReadMode.GRAY)

        if self.roi_2d:
            image = crop(
                image,
                self.roi_2d["y"],
                self.roi_2d["x"],
                self.roi_2d["height"],
                self.roi_2d["width"],
            )

        image = torch.squeeze(image.float() / 255.0).to(DEVICE)

        if self.trimmed_px_height > 0:
            image = image[: self.trimmed_px_height, :]

        if self.sector_mask.numel() == 0:
            raise RuntimeError("Ultrasound sector mask must be created before reading frames")
        if tuple(image.shape) != tuple(self.sector_mask.shape):
            raise ValueError(
                f"Image shape {tuple(image.shape)} does not match sector mask "
                f"{tuple(self.sector_mask.shape)} for {img_path}"
            )
        image = torch.where(self.sector_mask, image, torch.zeros_like(image))

        return image

    def _ensure_unit_intensity_range(self) -> None:
        tensor_names = ["pixels", "pixels_valid"]
        if self.has_gt:
            tensor_names.extend(["gt", "gt_valid"])

        max_values = []
        for name in tensor_names:
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.numel() > 0:
                max_values.append(float(torch.max(tensor).detach().cpu()))

        if not max_values or max(max_values) <= 1.0:
            self.image_value_scale = float(getattr(self, "image_value_scale", 1.0))
            return

        for name in tensor_names:
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.numel() > 0:
                setattr(self, name, tensor / 255.0)

        self.image_value_scale = 1.0
        print("Converted dataset image intensities from [0, 255] to [0, 1].")

    def get_bounding_box(self):
        return self.point_min_dev, self.point_max_dev

    def save(self, file_name):
        save_path = Path(file_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dataset": self}, save_path)

    def physical_calibration_signature(self) -> dict:
        return {
            "metadata_version": int(
                getattr(self, "metadata_version", DATASET_METADATA_VERSION)
            ),
            "source": str(getattr(self, "pixel_calibration_source", "unknown")),
            "image_size_px_wh": [int(self.px_width), int(self.px_height)],
            "pixel_size_mm_wh": [
                float(self.roi_px_size_width_mm),
                float(self.roi_px_size_height_mm),
            ],
            "physical_size_mm_wh": [float(self.width), float(self.height)],
        }

    def sector_mask_signature(self) -> dict:
        mask_bytes = self.sector_mask.detach().cpu().numpy().astype(np.uint8).tobytes()
        return {
            "version": int(getattr(self, "sector_mask_version", 0)),
            "source": str(getattr(self, "sector_mask_source", "unknown")),
            "shape_hw": [int(self.px_height), int(self.px_width)],
            "valid_pixels": int(self.valid_pixel_count),
            "total_pixels": int(self.pixels_per_slice),
            "valid_fraction": float(self.valid_pixel_fraction),
            "sha256": hashlib.sha256(mask_bytes).hexdigest(),
        }

    def _find_infos_json_for_saved_dataset(self, save_file: str | Path) -> Optional[Path]:
        candidates = []
        stored_path = getattr(self, "infos_json_path", "")
        if stored_path:
            candidates.append(Path(stored_path).expanduser())

        save_path = Path(save_file).expanduser()
        candidates.extend(
            [
                save_path.parent / "infos.json",
                save_path.parent / "us_recal_original" / "infos.json",
                save_path.parent.parent / "infos.json",
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _upgrade_saved_sector_mask(self) -> bool:
        existing_mask = getattr(self, "sector_mask", None)
        if existing_mask is not None and torch.as_tensor(existing_mask).numel() > 0:
            self.sector_mask_source = str(
                getattr(self, "sector_mask_source", "stored-baked-dataset")
            )
            self.sector_mask_sampled_frames = int(
                getattr(self, "sector_mask_sampled_frames", 0)
            )
            self.sector_mask_threshold = float(
                getattr(self, "sector_mask_threshold", 0.0)
            )
            self._finalize_sector_mask()
            return False

        slice_count = len(getattr(self, "slices", [])) + len(
            getattr(self, "slices_valid", [])
        )
        if slice_count < 1:
            raise ValueError("Saved dataset has no slices from which to detect a sector mask")

        frames = []
        combined_slices = [
            (self.pixels, item) for item in getattr(self, "slices", [])
        ] + [
            (self.pixels_valid, item) for item in getattr(self, "slices_valid", [])
        ]
        for sample_index in evenly_spaced_sample_indices(len(combined_slices)):
            tensor, slice_info = combined_slices[int(sample_index)]
            frame = tensor[int(slice_info.start) : int(slice_info.end)].reshape(
                int(self.px_height),
                int(self.px_width),
            )
            frames.append(frame.detach().float().cpu().numpy())

        detection = detect_ultrasound_sector_mask(np.stack(frames, axis=0))
        target_device = self.pixels.device
        self.sector_mask = torch.as_tensor(
            detection.mask,
            dtype=torch.bool,
            device=target_device,
        )
        self.sector_mask_source = f"legacy-{detection.source}"
        self.sector_mask_sampled_frames = detection.sampled_frames
        self.sector_mask_threshold = detection.foreground_threshold
        self._finalize_sector_mask()
        print(
            "Added mandatory ultrasound sector mask to legacy baked dataset: "
            f"valid={detection.valid_fraction:.2%}, "
            f"sampled_frames={detection.sampled_frames}"
        )
        return True

    def _recompute_cached_point_bounds_from_sector(self) -> None:
        point_min = torch.full(
            (3,),
            float("inf"),
            dtype=self.points.dtype,
            device=self.points.device,
        )
        point_max = torch.full(
            (3,),
            float("-inf"),
            dtype=self.points.dtype,
            device=self.points.device,
        )
        valid_flat = self.sector_valid_flat_indices.to(self.points.device)
        for slices, points_tensor in (
            (self.slices, self.points),
            (self.slices_valid, self.points_valid),
        ):
            for slice_info in slices:
                start = int(slice_info.start)
                valid_points = points_tensor[start + valid_flat]
                point_min = torch.minimum(point_min, valid_points.amin(dim=0))
                point_max = torch.maximum(point_max, valid_points.amax(dim=0))
        if not torch.all(torch.isfinite(point_min)) or not torch.all(torch.isfinite(point_max)):
            raise ValueError("Could not compute finite point bounds from the valid sector")
        self.point_min = point_min.detach().cpu().numpy().astype(np.float32)
        self.point_max = point_max.detach().cpu().numpy().astype(np.float32)
        self.point_min_dev = point_min.to(dtype=torch.float32)
        self.point_max_dev = point_max.to(dtype=torch.float32)

    def _rebuild_cached_slice_geometry(self) -> None:
        self.X, self.Y = get_base_points(
            self.width,
            self.height,
            self.px_width,
            self.px_height,
            offset_x_mm=self.roi_offset_x_mm,
            offset_y_mm=self.roi_offset_y_mm,
        )

        point_min = np.full(3, np.inf, dtype=np.float32)
        point_max = np.full(3, -np.inf, dtype=np.float32)
        tensor_groups = (
            (self.slices, self.points, self.viewdirs, "training"),
            (self.slices_valid, self.points_valid, self.viewdirs_valid, "validation"),
        )
        with torch.no_grad():
            for slices, points_tensor, viewdirs_tensor, split_name in tensor_groups:
                for slice_info in tqdm(slices, desc=f"Recalibrating {split_name} geometry"):
                    points_numpy, viewdirs_numpy = get_oriented_points_and_views(
                        self.X,
                        self.Y,
                        np.asarray(slice_info.position, dtype=np.float32),
                        slice_info.rotation,
                    )
                    expected_count = int(slice_info.end) - int(slice_info.start)
                    if points_numpy.shape[0] != expected_count:
                        raise ValueError(
                            f"Saved {split_name} slice contains {expected_count} pixels, "
                            f"but current metadata produces {points_numpy.shape[0]}"
                        )
                    start, end = int(slice_info.start), int(slice_info.end)
                    points_tensor[start:end].copy_(
                        torch.as_tensor(
                            points_numpy,
                            dtype=points_tensor.dtype,
                            device=points_tensor.device,
                        )
                    )
                    viewdirs_tensor[start:end].copy_(
                        torch.as_tensor(
                            viewdirs_numpy,
                            dtype=viewdirs_tensor.dtype,
                            device=viewdirs_tensor.device,
                        )
                    )
                    valid_flat = self.sector_mask.detach().cpu().numpy().reshape(-1)
                    point_min = np.minimum(
                        point_min,
                        points_numpy[valid_flat].min(axis=0),
                    )
                    point_max = np.maximum(
                        point_max,
                        points_numpy[valid_flat].max(axis=0),
                    )

        if not np.all(np.isfinite(point_min)) or not np.all(np.isfinite(point_max)):
            raise ValueError("Could not rebuild finite point bounds from saved slices")
        self.point_min = point_min.astype(np.float32)
        self.point_max = point_max.astype(np.float32)
        point_device = self.points.device
        self.point_min_dev = torch.as_tensor(
            self.point_min,
            dtype=torch.float32,
            device=point_device,
        )
        self.point_max_dev = torch.as_tensor(
            self.point_max,
            dtype=torch.float32,
            device=point_device,
        )

    def _upgrade_saved_metadata(self, save_file: str | Path) -> None:
        self.metadata_migrated_on_load = False
        saved_version = int(getattr(self, "metadata_version", 0))
        geometry_needs_rebuild = False
        if saved_version < PHYSICAL_METADATA_VERSION:
            infos_path = self._find_infos_json_for_saved_dataset(save_file)
            if infos_path is None:
                raise FileNotFoundError(
                    "Legacy baked dataset requires its infos.json to upgrade physical "
                    f"calibration: {save_file}"
                )
            with infos_path.open("r", encoding="utf-8") as infos_file:
                infos_json = json.load(infos_file)
            infos = infos_json["infos"]

            orig_px_width = int(getattr(self, "orig_px_width", self.px_width))
            orig_px_height = int(getattr(self, "orig_px_height", self.px_height))
            calibration = resolve_physical_calibration(
                infos,
                image_width_px=orig_px_width,
                image_height_px=orig_px_height,
            )
            roi = infos.get("ROI") or getattr(self, "roi_2d", None)
            roi_x = int(roi.get("x", 0)) if roi else 0
            roi_y = int(roi.get("y", 0)) if roi else 0
            new_values = np.array(
                [
                    int(self.px_width) * calibration.pixel_width_mm,
                    int(self.px_height) * calibration.pixel_height_mm,
                    calibration.pixel_width_mm,
                    calibration.pixel_height_mm,
                    roi_x * calibration.pixel_width_mm,
                    roi_y * calibration.pixel_height_mm,
                ],
                dtype=np.float64,
            )
            old_values = np.array(
                [
                    float(getattr(self, "width", 0.0)),
                    float(getattr(self, "height", 0.0)),
                    float(getattr(self, "roi_px_size_width_mm", 0.0)),
                    float(getattr(self, "roi_px_size_height_mm", 0.0)),
                    float(getattr(self, "roi_offset_x_mm", 0.0)),
                    float(getattr(self, "roi_offset_y_mm", 0.0)),
                ],
                dtype=np.float64,
            )

            self.pixel_calibration_source = calibration.source
            self.infos_json_path = str(infos_path)
            self.orig_px_width = orig_px_width
            self.orig_px_height = orig_px_height
            self.orig_px_size_width_mm = calibration.pixel_width_mm
            self.orig_px_size_height_mm = calibration.pixel_height_mm
            self.roi_2d = roi
            (
                self.width,
                self.height,
                self.roi_px_size_width_mm,
                self.roi_px_size_height_mm,
                self.roi_offset_x_mm,
                self.roi_offset_y_mm,
            ) = (float(value) for value in new_values)

            geometry_needs_rebuild = not np.allclose(
                old_values,
                new_values,
                rtol=1e-6,
                atol=1e-6,
            )
            if geometry_needs_rebuild:
                print(
                    "Upgrading baked dataset geometry from legacy metadata: "
                    f"size {old_values[0]:.6f}x{old_values[1]:.6f} mm -> "
                    f"{self.width:.6f}x{self.height:.6f} mm; "
                    f"spacing {old_values[2]:.6f}x{old_values[3]:.6f} mm/px -> "
                    f"{self.roi_px_size_width_mm:.6f}x"
                    f"{self.roi_px_size_height_mm:.6f} mm/px"
                )

        mask_added = self._upgrade_saved_sector_mask()
        if geometry_needs_rebuild:
            self._rebuild_cached_slice_geometry()
        elif mask_added:
            self._recompute_cached_point_bounds_from_sector()

        self.metadata_version = DATASET_METADATA_VERSION
        self.metadata_migrated_on_load = geometry_needs_rebuild or mask_added

    def _slice_tensor(self, tensor: torch.Tensor, slice_info: Slice) -> torch.Tensor:
        return torch.unsqueeze(tensor[slice_info.start:slice_info.end], 1)

    def _masked_slice_tensor(
        self,
        tensor: torch.Tensor,
        slice_info: Slice,
    ) -> torch.Tensor:
        values = self._slice_tensor(tensor, slice_info)
        mask = self.get_sector_mask(flatten=True, device=values.device)
        return torch.where(mask, values, torch.zeros_like(values))

    def get_slice_pixels(self, number):
        return self._masked_slice_tensor(self.pixels, self.slices[number])

    def get_slice_valid_pixels(self, number):
        return self._masked_slice_tensor(self.pixels_valid, self.slices_valid[number])

    def get_slice_gt(self, number):
        if not self.has_gt:
            return None
        return self._masked_slice_tensor(self.gt, self.slices[number])

    def get_slice_valid_gt(self, number):
        if not self.has_gt:
            return None
        return self._masked_slice_tensor(self.gt_valid, self.slices_valid[number])

    def get_slice_points(self, number):
        return self._slice_tensor(self.points, self.slices[number])

    def get_slice_valid_points(self, number):
        return self._slice_tensor(self.points_valid, self.slices_valid[number])

    def get_slice_viewdirs(self, number):
        return self._slice_tensor(self.viewdirs, self.slices[number])

    def get_slice_valid_viewdirs(self, number):
        return self._slice_tensor(self.viewdirs_valid, self.slices_valid[number])

    def _require_valid_sector_indices(self, indexes) -> torch.Tensor:
        indexes = torch.as_tensor(indexes, dtype=torch.long, device=self.pixels.device)
        local_indices = torch.remainder(indexes, self.pixels_per_slice)
        mask_flat = self.sector_mask.reshape(-1).to(indexes.device)
        if torch.any(~mask_flat[local_indices]):
            raise ValueError(
                "Requested flat indices include pixels outside the mandatory "
                "ultrasound sector mask"
            )
        return indexes

    def get_indices_pixels(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.pixels[indexes], 1)

    def get_indices_pixels_valid(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.pixels_valid[indexes], 1)

    def get_indices_points(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.points[indexes], 1)

    def get_indices_points_values(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.points_valid[indexes], 1)

    def get_indices_viewdirs(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.viewdirs[indexes], 1)

    def get_indices_viewdirs_valid(self, indexes):
        indexes = self._require_valid_sector_indices(indexes)
        return torch.unsqueeze(self.viewdirs_valid[indexes], 1)

    def get_sector_mask(
        self,
        *,
        flatten: bool = False,
        device=None,
    ) -> torch.Tensor:
        mask = self.sector_mask
        if device is not None:
            mask = mask.to(device)
        if flatten:
            return mask.reshape(-1, 1)
        return mask

    def map_valid_training_ranks(self, ranks: torch.Tensor) -> torch.Tensor:
        """Map compact valid-pixel ranks to the legacy flat tensor layout."""
        ranks = torch.as_tensor(ranks, dtype=torch.long, device=self.pixels.device)
        valid_count = self.valid_pixel_count
        if torch.any(ranks < 0) or torch.any(ranks >= valid_count * len(self.slices)):
            raise IndexError("Valid training rank is outside the dataset")
        slice_indices = torch.div(ranks, valid_count, rounding_mode="floor")
        within_slice = torch.remainder(ranks, valid_count)
        valid_flat = self.sector_valid_flat_indices.to(ranks.device)
        slice_starts = torch.as_tensor(
            [item.start for item in self.slices],
            dtype=torch.long,
            device=ranks.device,
        )
        return slice_starts[slice_indices] + valid_flat[within_slice]

    def get_valid_patch_origins(self, patch_size: int) -> torch.Tensor:
        """Return top-left [row, col] locations whose full patch is valid."""
        patch_size = int(patch_size)
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        cached = self._valid_patch_origins_cache.get(patch_size)
        if cached is not None:
            return cached
        if patch_size > self.px_height or patch_size > self.px_width:
            origins = torch.empty((0, 2), dtype=torch.long, device=self.pixels.device)
        else:
            mask = self.get_sector_mask(device=self.pixels.device).float()[None, None]
            coverage = F.avg_pool2d(mask, kernel_size=patch_size, stride=1)
            origins = torch.nonzero(coverage[0, 0] >= 1.0 - 1e-6, as_tuple=False)
        self._valid_patch_origins_cache[patch_size] = origins
        return origins

    @property
    def pixels_per_slice(self) -> int:
        return int(self.px_width) * int(self.px_height)

    @property
    def valid_pixel_count(self) -> int:
        return int(self.sector_valid_flat_indices.numel())

    @property
    def valid_pixel_fraction(self) -> float:
        if self.pixels_per_slice == 0:
            return 0.0
        return self.valid_pixel_count / self.pixels_per_slice

    def get_indices_frame_indices(self, indexes) -> torch.Tensor:
        """Return the training-slice index associated with each flat pixel index."""
        indexes = torch.as_tensor(indexes, dtype=torch.long, device=self.pixels.device)
        slice_ends = torch.as_tensor(
            [slice_info.end for slice_info in self.slices],
            dtype=torch.long,
            device=indexes.device,
        )
        return torch.searchsorted(slice_ends, indexes, right=True)

    def get_slice_frame_indices(self, number: int, *, device=None) -> torch.Tensor:
        """Return one frame index per pixel in a training slice."""
        if number < 0 or number >= len(self.slices):
            raise IndexError(f"Training slice index out of range: {number}")
        target_device = self.points.device if device is None else device
        return torch.full(
            (self.pixels_per_slice,),
            int(number),
            dtype=torch.long,
            device=target_device,
        )

    @staticmethod
    def open_from_save(save_file, *, map_location=None):
        load_device = DEVICE if map_location is None else map_location
        save = torch.load(save_file, weights_only=False, map_location=load_device)
        dataset = save["dataset"]
        dataset._upgrade_saved_metadata(save_file)
        dataset._ensure_unit_intensity_range()
        return dataset


def validate_checkpoint_dataset_geometry(
    checkpoint: dict,
    dataset: Dataset,
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Reject checkpoints whose learned coordinate system no longer matches data."""
    checkpoint_mask = checkpoint.get("ultrasound_sector_mask")
    if checkpoint_mask is not None:
        current_mask = dataset.sector_mask_signature()
        if (
            list(checkpoint_mask.get("shape_hw", [])) != current_mask["shape_hw"]
            or checkpoint_mask.get("sha256") != current_mask["sha256"]
        ):
            location = "" if checkpoint_path is None else f" {checkpoint_path}"
            raise ValueError(
                f"Checkpoint{location} ultrasound sector mask does not match the "
                "current dataset. Use the exact masked baked dataset used for training."
            )

    checkpoint_bbox = checkpoint.get("bounding_box")
    if checkpoint_bbox is None:
        return
    checkpoint_min = torch.as_tensor(checkpoint_bbox[0]).detach().cpu().numpy()
    checkpoint_max = torch.as_tensor(checkpoint_bbox[1]).detach().cpu().numpy()
    if np.allclose(checkpoint_min, dataset.point_min, rtol=1e-5, atol=1e-4) and np.allclose(
        checkpoint_max,
        dataset.point_max,
        rtol=1e-5,
        atol=1e-4,
    ):
        return

    location = "" if checkpoint_path is None else f" {checkpoint_path}"
    raise ValueError(
        f"Checkpoint{location} was trained with geometry that does not match the "
        "current px_size_cm-calibrated dataset. Legacy checkpoints cannot be "
        "rescaled after training; bake/load the corrected dataset and retrain. "
        f"checkpoint bbox={checkpoint_min.tolist()}..{checkpoint_max.tolist()}, "
        f"dataset bbox={dataset.point_min.tolist()}..{dataset.point_max.tolist()}"
    )


class Quat:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.compute_quat_params()

    def normalize(self):
        norm = np.sqrt(self.qw2 + self.qx2 + self.qy2 + self.qz2)
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.compute_quat_params()

    def compute_quat_params(self):
        self.qw2 = self.w ** 2
        self.qx2 = self.x ** 2
        self.qy2 = self.y ** 2
        self.qz2 = self.z ** 2

        self.dqxqy = self.x * self.y * 2
        self.dqwqz = self.w * self.z * 2
        self.dqxqz = self.x * self.z * 2
        self.dqwqy = self.w * self.y * 2
        self.dqyqz = self.y * self.z * 2
        self.dqwqx = self.w * self.x * 2

    def apply_quat(self, point):
        return np.array(
            [
                point[0] * (self.qw2 + self.qx2 - self.qy2 - self.qz2)
                + point[1] * (self.dqxqy + self.dqwqz)
                + point[2] * (self.dqxqz - self.dqwqy),
                point[0] * (self.dqxqy - self.dqwqz)
                + point[1] * (self.qw2 - self.qx2 + self.qy2 - self.qz2)
                + point[2] * (self.dqyqz + self.dqwqx),
                point[0] * (self.dqxqz + self.dqwqy)
                + point[1] * (self.dqyqz - self.dqwqx)
                + point[2] * (self.qw2 - self.qx2 - self.qy2 + self.qz2),
            ]
        )

    def as_rotmat(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
            ]
        )

    def __repr__(self):
        return (
            f"{self.qw2 ** 0.5}_{self.qx2 ** 0.5}_"
            f"{self.qy2 ** 0.5}_{self.qz2 ** 0.5}"
        )

    def __mul__(self, other):
        return Quat(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    @staticmethod
    def identity():
        return Quat(1, 0, 0, 0)
