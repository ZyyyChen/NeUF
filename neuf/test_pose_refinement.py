from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from neuf.dataset import Dataset
from neuf.main import NeUF
from neuf.nerf_network import NeRF
from neuf.pose_refinement import PoseRefiner


class PoseRefinerTests(unittest.TestCase):
    def setUp(self) -> None:
        rotations = torch.eye(3).repeat(2, 1, 1)
        translations = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        self.refiner = PoseRefiner(rotations, translations, anchor_first=True)

    def test_zero_initialized_corrections_preserve_geometry(self) -> None:
        points = torch.tensor([[1.0, 2.0, 3.0], [11.0, 22.0, 33.0]])
        viewdirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        refined_points, refined_viewdirs = self.refiner(
            points,
            viewdirs,
            torch.tensor([0, 1]),
        )

        torch.testing.assert_close(refined_points, points)
        torch.testing.assert_close(refined_viewdirs, viewdirs)

    def test_local_se3_increment_corrects_position_and_rotation(self) -> None:
        with torch.no_grad():
            self.refiner.se3_refine.weight[1] = torch.tensor(
                [0.0, 0.0, math.pi / 2.0, 1.0, 0.0, 0.0]
            )

        point = torch.tensor([[11.0, 20.0, 30.0]])
        viewdir = torch.tensor([[1.0, 0.0, 0.0]])
        refined_point, refined_viewdir = self.refiner(point, viewdir, 1)

        torch.testing.assert_close(
            refined_point,
            torch.tensor(
                [[10.0 + 2.0 / math.pi, 21.0 + 2.0 / math.pi, 30.0]]
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            refined_viewdir,
            torch.tensor([[0.0, 1.0, 0.0]]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradients_reach_selected_pose_but_not_anchor(self) -> None:
        points = torch.tensor([[1.0, 0.0, 0.0], [11.0, 20.0, 30.0]])
        viewdirs = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        refined_points, refined_viewdirs = self.refiner(
            points,
            viewdirs,
            torch.tensor([0, 1]),
        )
        (refined_points.sum() + refined_viewdirs.sum()).backward()

        gradient = self.refiner.se3_refine.weight.grad
        self.assertIsNotNone(gradient)
        torch.testing.assert_close(gradient[0], torch.zeros(6))
        self.assertGreater(float(torch.linalg.vector_norm(gradient[1])), 0.0)

    def test_refined_pose_matrix_matches_forward_transform(self) -> None:
        with torch.no_grad():
            self.refiner.se3_refine.weight[1, 3:] = torch.tensor([1.0, 2.0, 3.0])

        pose = self.refiner.refined_poses()[1]
        point = torch.tensor([[10.0, 20.0, 30.0]])
        viewdir = torch.tensor([[1.0, 0.0, 0.0]])
        refined_point, _ = self.refiner(point, viewdir, 1)

        expected = pose[:, :3] @ torch.zeros(3) + pose[:, 3]
        torch.testing.assert_close(refined_point.squeeze(0), expected)

    def test_trajectory_regularization_detects_pose_oscillation(self) -> None:
        refiner = PoseRefiner(
            torch.eye(3).repeat(4, 1, 1),
            torch.zeros(4, 3),
            anchor_first=False,
        )
        with torch.no_grad():
            refiner.se3_refine.weight[:, 3] = torch.tensor([0.0, 1.0, 3.0, 6.0])

        velocity, acceleration = refiner.trajectory_regularization_terms()

        torch.testing.assert_close(velocity, torch.tensor(14.0 / 3.0))
        torch.testing.assert_close(acceleration, torch.tensor(1.0))
        (velocity + acceleration).backward()
        self.assertGreater(
            float(torch.linalg.vector_norm(refiner.se3_refine.weight.grad)),
            0.0,
        )


class TrainingCurriculumTests(unittest.TestCase):
    def test_intensity_activation_is_explicit_and_validated(self) -> None:
        self.assertEqual(NeRF().intensity_activation, "identity")
        self.assertEqual(
            NeRF(intensity_activation="sigmoid").intensity_activation,
            "sigmoid",
        )
        with self.assertRaisesRegex(ValueError, "intensity_activation"):
            NeRF(intensity_activation="relu")

    def test_random_patch_slice_curriculum_uses_configured_durations(self) -> None:
        trainer = NeUF.__new__(NeUF)
        trainer.training_mode = "CurriculumRPS"
        trainer.curriculum_random_ratio = 0.2
        trainer.curriculum_patch_ratio = 0.5

        self.assertEqual(trainer._current_phase(0.0), "Random")
        self.assertEqual(trainer._current_phase(0.19), "Random")
        self.assertEqual(trainer._current_phase(0.2), "Patch")
        self.assertEqual(trainer._current_phase(0.69), "Patch")
        self.assertEqual(trainer._current_phase(0.7), "Slice")

    def test_gradient_loss_supports_structured_patch_batches(self) -> None:
        trainer = NeUF.__new__(NeUF)
        trainer.patch_size = 2
        trainer.dataset = SimpleNamespace(px_height=2, px_width=2)
        trainer.criterion = torch.nn.L1Loss()
        trainer.scharr_x, trainer.scharr_y = trainer._build_scharr_kernels()
        target = torch.tensor(
            [[0.0], [0.0], [1.0], [1.0]],
            device=trainer.scharr_x.device,
        )

        matching = trainer._compute_gradient_loss(target, target.clone(), "Patch")
        different = trainer._compute_gradient_loss(
            target,
            torch.zeros_like(target),
            "Patch",
        )

        torch.testing.assert_close(matching, matching.new_tensor(0.0))
        self.assertGreater(float(different), 0.0)

    def test_sagittal_weight_can_start_late_and_ramp(self) -> None:
        trainer = NeUF.__new__(NeUF)
        trainer.sagittal_weight = 0.2
        trainer.sagittal_start_iter = 100
        trainer.sagittal_ramp_iters = 50

        self.assertEqual(trainer._sagittal_weight_at(99), 0.0)
        self.assertEqual(trainer._sagittal_weight_at(100), 0.0)
        self.assertAlmostEqual(trainer._sagittal_weight_at(125), 0.1)
        self.assertAlmostEqual(trainer._sagittal_weight_at(150), 0.2)
        self.assertAlmostEqual(trainer._sagittal_weight_at(200), 0.2)


class LegacyDatasetPoseMappingTests(unittest.TestCase):
    def test_frame_indices_follow_saved_slice_boundaries(self) -> None:
        dataset = Dataset.__new__(Dataset)
        dataset.pixels = torch.empty(10)
        dataset.slices = [SimpleNamespace(end=3), SimpleNamespace(end=8)]

        frame_indices = dataset.get_indices_frame_indices(torch.tensor([0, 2, 3, 7]))

        torch.testing.assert_close(frame_indices, torch.tensor([0, 0, 1, 1]))

    def test_random_sampling_excludes_unmapped_legacy_tail(self) -> None:
        trainer = NeUF.__new__(NeUF)
        trainer.dataset = SimpleNamespace(
            slices=[
                SimpleNamespace(start=0, end=3),
                SimpleNamespace(start=3, end=8),
            ],
            pixels=torch.empty(10),
            points=torch.empty(10, 3),
            viewdirs=torch.empty(10, 3),
            valid_pixel_count=4,
            map_valid_training_ranks=lambda ranks: ranks,
        )
        trainer.training_mode = "Random"
        trainer.optimize_poses = True
        trainer.training_point_count = 0
        trainer.random_permutation = None
        trainer.random_start_index = 0
        trainer.points_per_iter = 8

        trainer._validate_dataset_configuration()
        sampled_indices = trainer._next_random_indices()

        self.assertEqual(trainer.training_point_count, 8)
        self.assertEqual(len(sampled_indices), 8)
        self.assertLess(int(sampled_indices.max()), 8)


if __name__ == "__main__":
    unittest.main()
