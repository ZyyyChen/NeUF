"""Matched Patient0 ablation of trajectory constraints and COR observation models."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


ARMS = ("fixed", "online", "constrained", "joint_point", "joint_cell")


def adjacent_motion(matrices):
    rotation = matrices[:, :3, :3]
    origin = matrices[:, :3, 3]
    normal = F.normalize(rotation[:-1, :, 2], dim=-1)
    distance = ((origin[1:] - origin[:-1]) * normal).sum(-1).abs()
    relative = rotation[:-1].transpose(1, 2) @ rotation[1:]
    skew = torch.stack((relative[:, 2, 1] - relative[:, 1, 2],
                        relative[:, 0, 2] - relative[:, 2, 0],
                        relative[:, 1, 0] - relative[:, 0, 1]), dim=-1)
    angle = torch.atan2(skew.norm(dim=-1), relative.diagonal(dim1=1, dim2=2).sum(-1) - 1)
    return torch.stack((distance, angle))


class CoronalObservation:
    def __init__(self, initial, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.constrained = arm in ("constrained", "joint_point", "joint_cell")
        self.joint = arm in ("joint_point", "joint_cell")
        self.cell = arm == "joint_cell"
        self.pose_cor_weight = 10.
        self.window_radius = 3
        initial_motion = adjacent_motion(initial.detach())
        windows = F.pad(initial_motion[None], (self.window_radius,) * 2,
                        mode="replicate")[0].unfold(-1, 2 * self.window_radius + 1, 1)
        self.lower = windows.amin(-1)
        self.upper = windows.amax(-1)
        self.penalty_weights = initial.new_tensor([0.03, 30.])
        self.thickness_mm = 1.5
        self.offsets = initial.new_tensor([-1., 0., 1.]) * (self.thickness_mm / 3)
        self.sigma_mm = self.thickness_mm / 4
        self.weights = torch.softmax(-0.5 * (self.offsets / self.sigma_mm).square(), dim=0)

    def config(self):
        return dict(arm=self.arm, field_loss="masked COR MSE", sag_pose_loss="masked single-scale 1-SSIM",
                    cor_pose_mse_weight=self.pose_cor_weight if self.joint else 0.,
                    distance_angle_weights=self.penalty_weights.tolist() if self.constrained else [0., 0.],
                    window_radius_pairs=self.window_radius if self.constrained else None,
                    cell_thickness_mm=self.thickness_mm if self.cell else 0.,
                    cell_offsets_mm=self.offsets.tolist() if self.cell else [0.],
                    cell_weights=self.weights.tolist() if self.cell else [1.],
                    cell_sigma_mm=self.sigma_mm if self.cell else None,
                    quadrature="deterministic subcell midpoints", cell_thickness_measured=False)

    def motion_penalty(self, matrices):
        values = adjacent_motion(matrices)
        excess = torch.maximum(self.lower - values, values - self.upper).clamp_min(0)
        return (excess.amax(dim=-1).square() * self.penalty_weights).sum()

    def predict(self, model, matrices, local, progress):
        # local [B,P,3] in mm; both centers and cell normals follow each candidate pose.
        xyz = local @ matrices[:, :3, :3].transpose(1, 2) + matrices[:, None, :3, 3]
        if not self.cell:
            return model(xyz, progress)[..., 0]
        normals = F.normalize(matrices[:, :3, 2], dim=-1)
        samples = xyz[:, :, None, :] + normals[:, None, None, :] * self.offsets[None, None, :, None]
        return (model(samples.reshape(-1, 3), progress)[..., 0].reshape(samples.shape[:-1])
                * self.weights).sum(-1)

    def cor_loss(self, model, poses, patch, progress):
        predicted = self.predict(model, poses.matrices()[patch["frames"]], patch["local"], progress)
        residual = predicted.reshape_as(patch["image"]) - patch["image"]
        mask = patch["mask"]
        return (residual.square() * mask).sum() / mask.sum().clamp_min(1)


def main():
    from .trajectory_reconstruction import Inputs, TrajectoryData
    from .workflow import train_coronal_variant
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    data = TrajectoryData(Inputs("cuda"), "CNN")
    trial = argparse.Namespace(output=args.output, mode="train", method="CNN", device="cuda",
                               pose_mode="off" if args.arm == "fixed" else "se3",
                               pose_supervision="observed", steps=20000, seed=3407, encoding="hash",
                               patches=4, patch_size=64, sagittal_patches=2, width=128, bands=10,
                               lr=.0005, pose_lr=.002)
    train_coronal_variant(data, data.guidance, args.output, trial, conditioned=False,
                          observation=CoronalObservation(data.initial, args.arm))


if __name__ == "__main__":
    main()
