from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from neuf.dataset import Quat
from neuf.sagittal_supervision import (
    SagittalSliceSupervisor,
    find_central_training_slice,
    load_matlab_image,
)


class _TinyDataset:
    px_height = 2
    px_width = 3
    scan_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    front_plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    back_plane_point = np.array([0.0, 2.0, 0.0], dtype=np.float32)

    def __init__(self) -> None:
        self.slices = [
            SimpleNamespace(
                start=index * 6,
                end=(index + 1) * 6,
                position=np.array([0.0, float(index), 0.0], dtype=np.float32),
                rotation=Quat.identity(),
            )
            for index in range(3)
        ]
        local_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.points = torch.cat(
            [local_points + torch.tensor([0.0, float(index), 0.0]) for index in range(3)]
        )
        self.viewdirs = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(18, 1)

    def get_slice_points(self, number: int) -> torch.Tensor:
        slice_info = self.slices[number]
        return self.points[slice_info.start:slice_info.end].unsqueeze(1)

    def get_slice_viewdirs(self, number: int) -> torch.Tensor:
        slice_info = self.slices[number]
        return self.viewdirs[slice_info.start:slice_info.end].unsqueeze(1)


class MatlabSagittalLoaderTests(unittest.TestCase):
    def test_v73_image_is_transposed_and_normalized(self) -> None:
        expected = np.array(
            [[0, 64, 128], [192, 224, 255]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slice.mat"
            with h5py.File(path, "w") as mat_file:
                mat_file.create_dataset("data_sag", data=expected.T)

            loaded = load_matlab_image(
                path,
                "data_sag",
                expected_shape=expected.shape,
            )

        np.testing.assert_allclose(loaded, expected.astype(np.float32) / 255.0)

    def test_missing_variable_reports_available_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slice.mat"
            with h5py.File(path, "w") as mat_file:
                mat_file.create_dataset("another_image", data=np.zeros((2, 3)))

            with self.assertRaisesRegex(KeyError, "another_image"):
                load_matlab_image(path, "data_sag")

    def test_trailing_rows_are_cropped_to_match_trimmed_dataset(self) -> None:
        source = np.arange(18, dtype=np.uint8).reshape(3, 6)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slice.mat"
            with h5py.File(path, "w") as mat_file:
                mat_file.create_dataset("data_sag", data=source.T)

            loaded = load_matlab_image(
                path,
                "data_sag",
                expected_shape=(2, 6),
            )

        np.testing.assert_allclose(
            loaded,
            source[:2].astype(np.float32) / 255.0,
        )


class SagittalSupervisorTests(unittest.TestCase):
    def test_central_pose_uses_scan_axis_geometry(self) -> None:
        dataset = _TinyDataset()
        self.assertEqual(find_central_training_slice(dataset), 1)

    def test_zero_initialized_pose_preserves_central_slice_and_receives_gradients(self) -> None:
        dataset = _TinyDataset()
        image = np.arange(6, dtype=np.uint8).reshape(2, 3) * 40 + 10
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slice.mat"
            with h5py.File(path, "w") as mat_file:
                mat_file.create_dataset("data_sag", data=image.T)

            supervisor = SagittalSliceSupervisor.from_dataset(
                dataset,
                path,
                device="cpu",
                optimize_pose=True,
            )

        expected_points = dataset.get_slice_points(1).reshape(-1, 3)
        expected_viewdirs = dataset.get_slice_viewdirs(1).reshape(-1, 3)
        points, viewdirs = supervisor.refined_geometry()
        torch.testing.assert_close(points, expected_points)
        torch.testing.assert_close(viewdirs, expected_viewdirs)
        self.assertEqual(supervisor.initial_slice_index, 1)
        self.assertEqual(tuple(supervisor.target_image().shape), (2, 3))

        (points.square().sum() + viewdirs.sum()).backward()
        gradient = supervisor.pose_refiner.se3_refine.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)


if __name__ == "__main__":
    unittest.main()
