from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import crop
from tqdm import tqdm

from utils import get_base_points, get_oriented_points_and_views

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_REQUIRED_KEYS = {"x", "y", "width", "height"}
DEFAULT_SPLIT_SEED = 17081998


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
    exclude_valid: bool
    image_step: int
    gt_folder: Optional[str]
    gt_prefix: str
    gt_suffix: str


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


class Dataset:
    def __init__(self, folder, nb_valid=4, seed=-1, **kwargs):
        config = self._build_config(folder, nb_valid, seed, kwargs)
        self._initialize_state(config)
        self._apply_random_seed(config.seed)

        infos_json = self._load_infos(config)
        frame_keys = self._get_selected_frame_keys(infos_json, config.image_step)
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
            exclude_valid=kwargs.get("exclude_valid", True),
            image_step=int(kwargs.get("image_step", 1)),
            gt_folder=kwargs.get("gt_folder"),
            gt_prefix=kwargs.get("gt_prefix", image_prefix),
            gt_suffix=kwargs.get("gt_suffix", image_suffix),
        )

    def _initialize_state(self, config: DatasetLoadConfig) -> None:
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
        self.nb_valid = config.nb_valid
        self.infos_json_path = ""

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
        scan_dims = infos.get("scan_dims_mm", {})
        orig_width_mm = float(scan_dims.get("width", 0.0))
        orig_height_mm = float(scan_dims.get("depth", 0.0))

        if orig_width_mm > 0 and orig_height_mm > 0:
            print(f"Original physical size: {orig_width_mm:.2f} x {orig_height_mm:.2f} mm")
            self.orig_px_size_width_mm = orig_width_mm / self.orig_px_width
            self.orig_px_size_height_mm = orig_height_mm / self.orig_px_height
            print(
                "Original pixel size: "
                f"{self.orig_px_size_width_mm:.4f} x {self.orig_px_size_height_mm:.4f} mm/px"
            )

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
        point_mins = np.stack([frame.points_numpy.min(axis=0) for frame in frames], axis=0)
        point_maxs = np.stack([frame.points_numpy.max(axis=0) for frame in frames], axis=0)
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

    def _print_loading_summary(self, folder: Path, frame_count: int, point_extent: np.ndarray) -> None:
        print("\n=== Dataset loading summary ===")
        print(f"Dataset path: {folder}")
        print(f"Image count: {frame_count}")
        print(f"Final image size: {self.px_width} x {self.px_height} px")
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

        return torch.squeeze(image.float()).to(DEVICE)

    def get_bounding_box(self):
        return self.point_min_dev, self.point_max_dev

    def save(self, file_name):
        save_path = Path(file_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dataset": self}, save_path)

    def _slice_tensor(self, tensor: torch.Tensor, slice_info: Slice) -> torch.Tensor:
        return torch.unsqueeze(tensor[slice_info.start:slice_info.end], 1)

    def get_slice_pixels(self, number):
        return self._slice_tensor(self.pixels, self.slices[number])

    def get_slice_valid_pixels(self, number):
        return self._slice_tensor(self.pixels_valid, self.slices_valid[number])

    def get_slice_gt(self, number):
        if not self.has_gt:
            return None
        return self._slice_tensor(self.gt, self.slices[number])

    def get_slice_valid_gt(self, number):
        if not self.has_gt:
            return None
        return self._slice_tensor(self.gt_valid, self.slices_valid[number])

    def get_slice_points(self, number):
        return self._slice_tensor(self.points, self.slices[number])

    def get_slice_valid_points(self, number):
        return self._slice_tensor(self.points_valid, self.slices_valid[number])

    def get_slice_viewdirs(self, number):
        return self._slice_tensor(self.viewdirs, self.slices[number])

    def get_slice_valid_viewdirs(self, number):
        return self._slice_tensor(self.viewdirs_valid, self.slices_valid[number])

    def get_indices_pixels(self, indexes):
        return torch.unsqueeze(self.pixels[indexes], 1)

    def get_indices_pixels_valid(self, indexes):
        return torch.unsqueeze(self.pixels_valid[indexes], 1)

    def get_indices_points(self, indexes):
        return torch.unsqueeze(self.points[indexes], 1)

    def get_indices_points_values(self, indexes):
        return torch.unsqueeze(self.points_valid[indexes], 1)

    def get_indices_viewdirs(self, indexes):
        return torch.unsqueeze(self.viewdirs[indexes], 1)

    def get_indices_viewdirs_valid(self, indexes):
        return torch.unsqueeze(self.viewdirs_valid[indexes], 1)

    @staticmethod
    def open_from_save(save_file):
        save = torch.load(save_file, weights_only=False, map_location=DEVICE)
        return save["dataset"]


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
