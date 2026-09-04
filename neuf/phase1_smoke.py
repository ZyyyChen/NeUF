from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neuf.dataset import Dataset
from neuf.export_full_grid_from_ckpt import query_grid
from neuf.nerf_network import NeRF
from neuf.phase1_data import (
    apply_phase1_training_split,
    current_pose_hash,
    freeze_phase1_manifests,
)
from neuf.phase1_losses import masked_mean
from neuf.slice_renderer import SliceRenderer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_smoke(dataset_path: Path, output_dir: Path) -> dict:
    started = time.time()
    torch.manual_seed(3407)
    np.random.seed(3407)
    dataset = Dataset.open_from_save(dataset_path)
    splits = freeze_phase1_manifests(
        dataset,
        dataset_path,
        output_dir,
        allow_small_dataset=True,
    )
    apply_phase1_training_split(dataset, splits)
    pose_before = current_pose_hash(dataset)

    model = NeRF(
        field_head=NeRF.ANATOMY_SPECKLE_FIELD_HEAD,
        intensity_activation="identity",
    )
    model.init_dual_encoding(
        pe_type="hash",
        bounding_box=dataset.get_bounding_box(),
        n_levels_low=8,
        n_levels_high=8,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution_low=16,
        finest_resolution_low=64,
        base_resolution_high=64,
        finest_resolution_high=512,
        use_gate=False,
        hf_activate_ratio=0.2,
        hf_max_weight=1.0,
    )
    model.init_model()
    renderer = SliceRenderer(dataset)
    optimizer = torch.optim.Adam(model.grad_vars(), lr=5e-4)

    patch_size = 64
    width = int(dataset.px_width)
    row, column = 18, 43
    offsets = (
        torch.arange(patch_size, device=dataset.pixels.device)[:, None] * width
        + torch.arange(patch_size, device=dataset.pixels.device)[None, :]
    ).reshape(-1)
    indices = row * width + column + offsets
    points = dataset.points[indices].unsqueeze(1)
    viewdirs = dataset.viewdirs[indices].unsqueeze(1)
    target = dataset.pixels[indices].reshape(1, 1, patch_size, patch_size)
    mask = dataset.get_sector_mask(flatten=True, device=DEVICE)[indices]
    mask = mask.reshape(1, 1, patch_size, patch_size)

    losses = []
    gradient_paths = {}
    for progress in (0.1, 0.5):
        model.training_progress = progress
        components_flat = renderer.query_point_components(model, points, viewdirs)
        components = {
            name: value.reshape(1, 1, patch_size, patch_size)
            for name, value in components_flat.items()
        }
        loss = masked_mean((components["intensity"] - target).square(), mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_paths = {
            "low_encoder": any(p.grad is not None for p in model.dual_encoder.enc_low.parameters()),
            "high_encoder": any(p.grad is not None for p in model.dual_encoder.enc_high.parameters()),
            "anatomy_head": any(p.grad is not None for p in model.anatomy_head.parameters()),
            "speckle_head": any(p.grad is not None for p in model.speckle_head.parameters()),
        }
        if not all(gradient_paths.values()):
            raise RuntimeError(f"E2 joint gradient path is incomplete: {gradient_paths}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.training_progress = 1.0
    checkpoint = model.get_save_dict()
    checkpoint.update(
        {
            "seed": 3407,
            "start": 2,
            "baked": True,
            "baked_dataset_file": str(dataset_path.resolve()),
            "bounding_box": dataset.get_bounding_box(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ultrasound_sector_mask": dataset.sector_mask_signature(),
            "dataset_physical_calibration": dataset.physical_calibration_signature(),
            "phase1_smoke_only": True,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pkl"
    torch.save(checkpoint, checkpoint_path)

    restored = NeRF(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    restored.training_progress = 1.0
    restored.eval()
    with torch.no_grad():
        validation = renderer.render_slice_from_dataset_valid(
            restored,
            0,
            reshaped=True,
            alpha=0.5,
        )
        point_min = np.asarray(dataset.point_min, dtype=np.float32)
        point_max = np.asarray(dataset.point_max, dtype=np.float32)
        axes = [
            np.linspace(point_min[index], point_max[index], 2, dtype=np.float32)
            for index in range(3)
        ]
        volume = query_grid(
            restored,
            checkpoint,
            axes[0],
            axes[1],
            axes[2],
            chunk_size=8,
            use_bbox_mask=True,
            alpha=0.5,
            component="intensity",
        )

    pose_after = current_pose_hash(dataset)
    result = {
        "status": "PASS (smoke only; not a research result)",
        "dataset": str(dataset_path.resolve()),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "losses": losses,
        "joint_gradient_paths": gradient_paths,
        "use_gate": model.dual_encoder.use_gate,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_roundtrip": True,
        "slice_shape_hw": list(validation.shape),
        "slice_finite": bool(torch.isfinite(validation).all()),
        "volume_shape_zyx": list(volume.shape),
        "volume_finite": bool(np.isfinite(volume).all()),
        "pose_hash_before": pose_before,
        "pose_hash_after": pose_after,
        "pose_unchanged": pose_before == pose_after,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One lightweight Phase 1 smoke run")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/simu_56/us/baked_dataset.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase1_image_quality/logs/smoke"),
    )
    args = parser.parse_args()
    result = run_smoke(args.dataset, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
