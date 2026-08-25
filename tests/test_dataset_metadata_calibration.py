from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from neuf.dataset import (
    Dataset,
    Quat,
    Slice,
    resolve_physical_calibration,
    validate_checkpoint_dataset_geometry,
)
from neuf.ultrasound_mask import detect_ultrasound_sector_mask
from neuf.utils import get_base_points, get_oriented_points_and_views


def test_px_size_cm_overrides_legacy_scan_dimensions() -> None:
    calibration = resolve_physical_calibration(
        {
            "scan_dims_mm": {"width": 944, "depth": 708},
            "px_size_cm": {"width": 0.0119928, "height": 0.0119928},
        },
        image_width_px=944,
        image_height_px=708,
    )

    assert calibration.source == "px_size_cm"
    assert calibration.pixel_width_mm == pytest.approx(0.119928)
    assert calibration.pixel_height_mm == pytest.approx(0.119928)
    assert calibration.width_mm == pytest.approx(113.212032)
    assert calibration.height_mm == pytest.approx(84.909024)


def test_sector_mask_excludes_screen_annotations_and_keeps_dark_anatomy() -> None:
    frames = np.zeros((4, 64, 80), dtype=np.uint8)
    for frame_index in range(len(frames)):
        for row in range(6, 56):
            half_width = 12 + row // 3
            frames[frame_index, row, 40 - half_width : 40 + half_width] = 60 + frame_index
    frames[:, 25:32, 37:43] = 0  # Dark anatomy inside the fan must remain valid.
    frames[:, 2:8, 70:78] = 220  # Date/device text.
    frames[:, 58:63, 8:28] = 220  # DROIT/GCT-style bottom label.

    detection = detect_ultrasound_sector_mask(frames, safety_margin_px=1)

    assert detection.mask[40, 40]
    assert detection.mask[28, 40]
    assert not detection.mask[4, 74]
    assert not detection.mask[60, 16]


def _write_legacy_dataset(path: Path, infos_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = Dataset.__new__(Dataset)
    dataset.width = 4.0
    dataset.height = 2.0
    dataset.px_width = 4
    dataset.px_height = 2
    dataset.orig_px_width = 4
    dataset.orig_px_height = 2
    dataset.orig_px_size_width_mm = 1.0
    dataset.orig_px_size_height_mm = 1.0
    dataset.roi_px_size_width_mm = 1.0
    dataset.roi_px_size_height_mm = 1.0
    dataset.roi_offset_x_mm = 0.0
    dataset.roi_offset_y_mm = 0.0
    dataset.roi_2d = None
    dataset.infos_json_path = str(infos_path)
    dataset.has_gt = False
    dataset.image_value_scale = 1.0

    position = np.zeros(3, dtype=np.float32)
    rotation = Quat.identity()
    old_x, old_y = get_base_points(4.0, 2.0, 4, 2)
    old_points, old_viewdirs = get_oriented_points_and_views(
        old_x,
        old_y,
        position,
        rotation,
    )
    dataset.X = old_x
    dataset.Y = old_y
    dataset.slices = [Slice(0, 8, position, rotation, frame_index=0)]
    dataset.slices_valid = []
    dataset.points = torch.from_numpy(old_points.astype(np.float32))
    dataset.viewdirs = torch.from_numpy(old_viewdirs.astype(np.float32))
    dataset.points_valid = torch.empty((0, 3), dtype=torch.float32)
    dataset.viewdirs_valid = torch.empty((0, 3), dtype=torch.float32)
    # A legacy acquisition still needs detectable image support when the new
    # mandatory sector mask is added during load.
    dataset.pixels = torch.ones(8, dtype=torch.float32)
    dataset.pixels_valid = torch.empty(0, dtype=torch.float32)
    dataset.point_min = old_points.min(axis=0).astype(np.float32)
    dataset.point_max = old_points.max(axis=0).astype(np.float32)
    dataset.point_min_dev = torch.from_numpy(dataset.point_min)
    dataset.point_max_dev = torch.from_numpy(dataset.point_max)
    torch.save({"dataset": dataset}, path)
    return dataset.point_min.copy(), dataset.point_max.copy()


def test_legacy_pickle_rebuilds_cached_world_geometry() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        infos_path = root / "infos.json"
        infos_path.write_text(
            json.dumps(
                {
                    "infos": {
                        "scan_dims_mm": {"width": 4, "depth": 2},
                        "px_size_cm": {"width": 0.05, "height": 0.05},
                    }
                }
            ),
            encoding="utf-8",
        )
        dataset_path = root / "dataset.pkl"
        old_min, old_max = _write_legacy_dataset(dataset_path, infos_path)

        loaded = Dataset.open_from_save(dataset_path, map_location="cpu")

    expected_x, expected_y = get_base_points(2.0, 1.0, 4, 2)
    expected_points, _ = get_oriented_points_and_views(
        expected_x,
        expected_y,
        np.zeros(3, dtype=np.float32),
        Quat.identity(),
    )
    assert loaded.metadata_migrated_on_load
    assert loaded.pixel_calibration_source == "px_size_cm"
    assert loaded.width == pytest.approx(2.0)
    assert loaded.height == pytest.approx(1.0)
    assert loaded.roi_px_size_width_mm == pytest.approx(0.5)
    assert loaded.roi_px_size_height_mm == pytest.approx(0.5)
    np.testing.assert_allclose(loaded.points.numpy(), expected_points, atol=1e-6)
    np.testing.assert_allclose(loaded.point_min, expected_points.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(loaded.point_max, expected_points.max(axis=0), atol=1e-6)

    with pytest.raises(ValueError, match="Legacy checkpoints cannot be rescaled"):
        validate_checkpoint_dataset_geometry(
            {"bounding_box": (torch.from_numpy(old_min), torch.from_numpy(old_max))},
            loaded,
        )
