from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

from neuf.dataset import Dataset
from neuf.export_full_grid_from_ckpt import query_grid, save_mhd
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


def _independent_stv_inference(checkpoint_path: Path, teacher_path: Path) -> dict:
    """在新进程阻断 STV 导入和 teacher 文件读取，检查部署只依赖三维场。"""
    script = r'''
import builtins
import io
import json
import os
import sys
from pathlib import Path

teacher_path = os.path.abspath(sys.argv[2])
original_import, original_open = builtins.__import__, builtins.open
original_io_open, original_is_file = io.open, Path.is_file

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "stv" or name.startswith("stv.") or name == "neuf.frozen_stv":
        raise ImportError("Neural STV is unavailable in the inference smoke process")
    if name == "neuf" and "frozen_stv" in (fromlist or ()):
        raise ImportError("Teacher module is unavailable")
    return original_import(name, globals, locals, fromlist, level)

def is_teacher(file):
    return isinstance(file, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(file)) == teacher_path

def guarded_open(file, *args, **kwargs):
    if is_teacher(file):
        raise FileNotFoundError("Teacher checkpoint is unavailable")
    return original_open(file, *args, **kwargs)

def guarded_io_open(file, *args, **kwargs):
    if is_teacher(file):
        raise FileNotFoundError("Teacher checkpoint is unavailable")
    return original_io_open(file, *args, **kwargs)

builtins.__import__, builtins.open, io.open = guarded_import, guarded_open, guarded_io_open
Path.is_file = lambda self: False if is_teacher(self) else original_is_file(self)
assert not Path(teacher_path).is_file()

import torch
from neuf.nerf_network import NeRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(sys.argv[1], map_location=device, weights_only=False)
model = NeRF(checkpoint).eval()
bbox = torch.stack([torch.as_tensor(value, device=device) for value in checkpoint["bounding_box"]])
points = (bbox[0] + (bbox[1] - bbox[0]) * torch.tensor([[0.25], [0.5], [0.75]], device=device)).float()
with torch.no_grad():
    result = model.query_components(points)
    torch.testing.assert_close(result["structure"] + result["boundary"] + result["residual"], result["intensity"])
    assert all(torch.isfinite(value).all() for value in result.values())
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        output = model.query_components(points, alpha=alpha)
        torch.testing.assert_close(output["intensity"], result["anatomy"] + alpha * result["residual"])
        for name in ("structure", "boundary", "residual", "anatomy"):
            torch.testing.assert_close(output[name], result[name], rtol=0, atol=0)
assert not any(name == "stv" or name.startswith("stv.") or name == "neuf.frozen_stv" for name in sys.modules)
print(json.dumps({"fresh_process": True, "stv_import_blocked": True, "teacher_file_access_blocked": True, "alpha_sweep": [0, 0.25, 0.5, 0.75, 1], "component_names": sorted(result)}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint_path.resolve()), str(teacher_path.resolve())],
        cwd=Path(__file__).resolve().parents[1],
        # 集群 GPU 为独占进程模式；独立部署检查使用 CPU，避免争用训练进程的设备。
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"独立推理进程失败：\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.splitlines()[-1])


def _plot_frozen_comparison(target, mask, before, after, output_path: Path) -> dict:
    """固定一个 64×64 训练 patch；两步结果只说明实现可运行，不评价重建质量。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def array(value):
        return value.detach().cpu().reshape(64, 64).numpy()

    gt, valid = array(target), array(mask).astype(bool)
    initial, final = array(before["intensity"]), array(after["intensity"])
    signed_limit = 0.5
    error_limit = 0.5
    panels = [
        ("GT", gt, "gray", 0.0, 1.0),
        ("Before: random E1 initialization", initial, "gray", 0.0, 1.0),
        ("After: 2 updates", final, "gray", 0.0, 1.0),
        ("Absolute error: before", np.abs(initial - gt), "magma", 0.0, error_limit),
        ("Structure: after", array(after["structure"]), "gray", 0.0, 1.0),
        ("Signed boundary: after", array(after["boundary"]), "RdBu_r", -max(signed_limit, 1e-8), max(signed_limit, 1e-8)),
        ("Residual: after", array(after["residual"]), "RdBu_r", -max(signed_limit, 1e-8), max(signed_limit, 1e-8)),
        ("Absolute error: after", np.abs(final - gt), "magma", 0.0, error_limit),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for axis, (title, values, cmap, lower, upper) in zip(axes.flat, panels):
        plotted = axis.imshow(np.where(valid, values, np.nan), cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xlabel("Column (px)")
        axis.set_ylabel("Row (px)")
        figure.colorbar(plotted, ax=axis, fraction=0.046)
    figure.suptitle("Frozen Neural STV smoke; same observed patch; reconstruction quality unvalidated")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return {
        "before_masked_mse": float(np.mean((initial[valid] - gt[valid]) ** 2)),
        "after_masked_mse": float(np.mean((final[valid] - gt[valid]) ** 2)),
        "image_display_range": [0.0, 1.0],
        "signed_display_range": [-max(signed_limit, 1e-8), max(signed_limit, 1e-8)],
        "error_display_range": [0.0, error_limit],
        "valid_pixel_count": int(valid.sum()),
    }


def run_frozen_smoke(dataset_path: Path, output_dir: Path, teacher_path: Path) -> dict:
    """复用唯一 smoke 入口验证冻结监督、E1 迁移、三维导出和独立推理。"""
    from neuf.main import NeUFTrainer, ValidationPreview, _save_alpha_previews
    from neuf.frozen_stv import frozen_stv_anatomy_loss, anatomy_spatial_loss

    started = time.time()
    torch.manual_seed(3407)
    np.random.seed(3407)
    checkpoints_dir, metrics_dir = output_dir / "checkpoints", output_dir / "metrics"
    plots_dir, config_dir = output_dir / "plots", output_dir / "run_config"
    for directory in (checkpoints_dir, metrics_dir, plots_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 小 HashGrid 的 E1 只用于验证权重迁移，不冒充已经训练好的重建基线。
    dataset = Dataset.open_from_save(dataset_path)
    e1 = NeRF(field_head=NeRF.MATCHED_FIELD_HEAD, intensity_activation="identity")
    e1.init_dual_encoding(bounding_box=dataset.get_bounding_box(), log2_hashmap_size=8, use_gate=False)
    e1.init_model()
    e1_checkpoint = e1.get_save_dict()
    e1_checkpoint.update({
        "baked": True,
        "baked_dataset_file": str(dataset_path.resolve()),
        "ultrasound_sector_mask": dataset.sector_mask_signature(),
        "dataset_physical_calibration": dataset.physical_calibration_signature(),
    })
    e1_path = checkpoints_dir / "e1_initialization.pkl"
    torch.save(e1_checkpoint, e1_path)
    del dataset, e1_checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer = NeUFTrainer(
        dataset=str(dataset_path), root=str(output_dir),
        phase1_output_dir=str(config_dir), phase1_allow_small_dataset=True,
        field_head=NeRF.FROZEN_STV_FIELD_HEAD, encoding="DUAL_HASH",
        init_e1_checkpoint=str(e1_path), stv_checkpoint=str(teacher_path),
        stv_tile_size=128, hash_log2_hashmap_size=8,
        training_mode="Patch", patch_size=64, points_per_iter=4096,
        nb_iters_max=2, seed=3407,
    )
    teacher = trainer.stv_teacher._model
    teacher_before = {name: value.detach().cpu().clone() for name, value in teacher.state_dict().items()}
    teacher_metadata = trainer.stv_teacher.metadata()
    json.dumps(teacher_metadata)
    assert not teacher.training and all(not parameter.requires_grad for parameter in teacher.parameters())

    target, points, viewdirs, mask = fixed_batch = trainer._sample_batch()
    fixed_teacher_targets = {name: value.detach().clone() for name, value in trainer.stv_batch_targets.items()}
    torch.testing.assert_close(sum(fixed_teacher_targets[name] for name in ("structure", "boundary", "residual"))[mask], target[mask])
    assert all(not value.requires_grad and not torch.is_inference(value) for value in fixed_teacher_targets.values())
    with torch.no_grad():
        _, before = trainer._query_batch(points, viewdirs)
        previous = trainer.renderer.query_points(e1, points, viewdirs)
        torch.testing.assert_close(before["structure"], previous)
        torch.testing.assert_close(before["intensity"], previous)
        assert torch.count_nonzero(before["boundary"]) == 0 and torch.count_nonzero(before["residual"]) == 0
    del e1

    # 首批只有一个 patch，精确记录对应切片与像素起点，前后显示不再重新抽样。
    frame_index = next(iter(trainer.stv_target_cache))
    source_slice = trainer.dataset.slices[frame_index]
    frame_points = trainer.dataset.points[source_slice.start:source_slice.end]
    local_indices = torch.nonzero((frame_points == points.reshape(-1, 3)[0]).all(dim=-1)).reshape(-1)
    assert local_indices.numel() == 1
    row, column = divmod(int(local_indices[0]), int(trainer.dataset.px_width))
    comparison = {
        "dataset": str(dataset_path.resolve()), "comparison_indices": [int(frame_index)],
        "original_frame_index": None if source_slice.frame_index is None else int(source_slice.frame_index),
        "selection": "first sampled patch with seed 3407 before any update",
        "crop_rc_hw": [row, column, 64, 64], "orientation": "native observed transverse slice",
        "normalization": "dataset display intensity [0,1], no per-patch normalization",
        "mask": "dataset ultrasound sector", "steps": [0, 2],
        "model": NeRF.FROZEN_STV_FIELD_HEAD, "quality_status": "尚未验证",
    }
    (config_dir / "comparison_manifest.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    initial_output_weights = trainer.model.components_head.output.weight.detach().clone()
    # 原图误差可由 A/R 抵消；alpha=0 的专门损失必须仍能识别 anatomy 损坏。
    desired = fixed_teacher_targets["structure"] + fixed_teacher_targets["boundary"]
    yy, xx = torch.meshgrid(torch.arange(64, device=DEVICE), torch.arange(64, device=DEVICE), indexing="ij")
    corruption = (((xx + yy) % 2).float() * 0.1 - 0.05).reshape_as(desired)
    corrupted_a = (desired + corruption).detach().requires_grad_()
    endpoint, endpoint_terms = frozen_stv_anatomy_loss(
        {"anatomy": corrupted_a}, fixed_teacher_targets, mask,
        patch_size=64, anatomy_weight=1, edge_weight=0.1,
    )
    assert endpoint_terms["anatomy_mse"] > 0 and endpoint_terms["anatomy_edge_mse"] > 0
    endpoint.backward()
    assert corrupted_a.grad is not None and corrupted_a.grad.abs().sum() > 0
    torch.testing.assert_close(corrupted_a.detach() + target - desired - corruption, target)
    losses, gradients = [], []
    timing_path = metrics_dir / "timing.csv"
    timing_path.write_text("epoch,global_step,step_time_sec,elapsed_sec\n", encoding="utf-8")
    progress = tqdm.trange(2, desc="Frozen STV smoke", mininterval=1.0)
    for step in progress:
        step_started = time.perf_counter()
        batch = fixed_batch if step == 0 else trainer._sample_batch()
        target, points, viewdirs, mask = batch
        prediction, components = trainer._query_batch(points, viewdirs)
        loss, terms = trainer._training_loss(target, prediction, mask, components, stage=0, points=points)
        assert torch.isfinite(loss)
        assert terms["anatomy_spatial_points"] > 0
        if step == 0:
            spatial, _ = anatomy_spatial_loss(
                trainer.model, points, components["anatomy"], mask,
                trainer.stv_batch_targets["smooth_weight"], trainer.dataset.get_bounding_box(),
                step_mm=0.5, max_points=32, generator=trainer.spatial_generator,
            )
            spatial_gradient = torch.autograd.grad(spatial, trainer.model.components_head.output.weight, retain_graph=True)[0]
            assert spatial_gradient[:2].abs().sum() > 0 and torch.count_nonzero(spatial_gradient[2]) == 0
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        output_gradient = trainer.model.components_head.output.weight.grad
        assert output_gradient is not None
        component_gradients = {
            name: float(output_gradient[index].norm().detach().cpu())
            for index, name in enumerate(("structure", "boundary", "residual"))
        }
        encoder_gradients = {
            name: any(parameter.grad is not None and bool(torch.any(parameter.grad != 0)) for parameter in encoder.parameters())
            for name, encoder in (("low_encoder", trainer.model.dual_encoder.enc_low), ("high_encoder", trainer.model.dual_encoder.enc_high))
        }
        assert all(np.isfinite(value) and value > 0 for value in component_gradients.values())
        assert all(encoder_gradients.values())
        assert all(parameter.grad is None for parameter in teacher.parameters())
        trainer.optimizer.step()
        trainer.scheduler.step()
        gradients.append({**component_gradients, **encoder_gradients})
        losses.append({"step": step + 1, "total": float(loss.detach().cpu()), **{name: float(value.detach().cpu()) for name, value in terms.items()}})
        step_seconds = time.perf_counter() - step_started
        with timing_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([1, step + 1, step_seconds, time.time() - started])
        progress.set_postfix(loss=losses[-1]["total"], seconds=step_seconds)
    assert not torch.equal(initial_output_weights, trainer.model.components_head.output.weight.detach())
    assert all(torch.equal(teacher_before[name], value.detach().cpu()) for name, value in teacher.state_dict().items())

    target, points, viewdirs, mask = fixed_batch
    with torch.no_grad():
        _, after = trainer._query_batch(points, viewdirs)
        torch.testing.assert_close(after["structure"] + after["boundary"], after["anatomy"])
        torch.testing.assert_close(after["anatomy"] + after["residual"], after["intensity"])
    checkpoint = trainer._checkpoint_payload(2)
    checkpoint["phase1_smoke_only"] = True
    checkpoint_path = checkpoints_dir / "frozen_stv_step_000002.pkl"
    torch.save(checkpoint, checkpoint_path)
    restored = NeRF(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)).eval()
    with torch.no_grad():
        restored_components = trainer.renderer.query_point_components(restored, points, viewdirs)
        for name, value in after.items():
            torch.testing.assert_close(restored_components[name], value)
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            output = restored.query_components(points, alpha=alpha)
            torch.testing.assert_close(output["intensity"], after["anatomy"] + alpha * after["residual"])
        for alpha in (-0.1, 1.1, float("nan")):
            try:
                restored.query(points, alpha=alpha)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid alpha was accepted")
        default_zero = NeRF(checkpoint, default_alpha=0).eval()
        default_roundtrip = NeRF(default_zero.get_save_dict()).eval()
        torch.testing.assert_close(default_roundtrip.query(points), after["anatomy"])
        del default_zero, default_roundtrip

    bbox = torch.stack([torch.as_tensor(value).detach().cpu() for value in checkpoint["bounding_box"]]).numpy()
    axes = [np.linspace(bbox[0, axis], bbox[1, axis], 2, dtype=np.float32) for axis in range(3)]
    volumes = {name: query_grid(restored, checkpoint, *axes, chunk_size=8, component=name) for name in after}
    assert all(np.isfinite(value).all() for value in volumes.values())
    np.testing.assert_allclose(volumes["structure"] + volumes["boundary"] + volumes["residual"], volumes["intensity"], rtol=1e-5, atol=1e-6)
    for alpha in (0, 0.5, 1):
        actual = query_grid(restored, checkpoint, *axes, chunk_size=8, alpha=alpha)
        np.testing.assert_allclose(actual, volumes["anatomy"] + alpha * volumes["residual"], rtol=1e-5, atol=1e-6)
    float_dir = output_dir / "predictions"
    save_mhd(volumes["anatomy"], float_dir, (1.0, 1.0, 1.0), bbox[0], stem="smoke_anatomy_float")
    assert "ElementType = MET_FLOAT" in (float_dir / "smoke_anatomy_float.mhd").read_text()
    np.testing.assert_array_equal(np.fromfile(float_dir / "smoke_anatomy_float.raw", dtype=np.float32).reshape(2, 2, 2), volumes["anatomy"])
    independent_inference = _independent_stv_inference(checkpoint_path, teacher_path)
    figure_path = plots_dir / "fixed_patch_before_after_step_000002.png"
    comparison_metrics = _plot_frozen_comparison(target, mask, before, after, figure_path)
    _save_alpha_previews(plots_dir, 2, [ValidationPreview(
        slice_id=f"smoke_frame_{frame_index}_r{row}_c{column}",
        target=target.detach().cpu().reshape(64, 64),
        prediction=after["intensity"].detach().cpu().reshape(64, 64),
        anatomy=after["anatomy"].detach().cpu().reshape(64, 64),
        boundary=after["boundary"].detach().cpu().reshape(64, 64),
        residual=after["residual"].detach().cpu().reshape(64, 64),
        mask=mask.detach().cpu().reshape(64, 64),
    )], selection="first sampled training patch with seed 3407; smoke only")
    pose_after = current_pose_hash(trainer.dataset)
    assert pose_after == trainer.pose_hash_before
    trainer.writer.close()
    result = {
        "status": "PASS (smoke only; reconstruction quality unvalidated)",
        "dataset": str(dataset_path.resolve()), "seed": 3407,
        "field_head": trainer.field_head, "steps": 2, "losses": losses,
        "component_and_encoder_gradients": gradients,
        "teacher_parameters_frozen": True, "teacher_state_unchanged": True,
        "teacher_targets_closure": True, "teacher_metadata": teacher_metadata,
        "student_weights_changed": True, "e1_initialization_equivalent": True,
        "checkpoint_roundtrip": True, "checkpoint": str(checkpoint_path.resolve()),
        "alpha_endpoints_and_interpolation": True, "default_alpha_roundtrip": True,
        "anatomy_loss_detects_compensated_corruption": True,
        "spatial_loss_preserves_residual_output_row": True, "float_mhd_roundtrip": True,
        "volume_components_closure": True, "volume_shape_zyx": list(volumes["intensity"].shape),
        "independent_inference": independent_inference,
        "fixed_patch_metrics": comparison_metrics, "comparison": comparison,
        "comparison_plot": str(figure_path.resolve()), "pose_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    (metrics_dir / "smoke_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--teacher-checkpoint", type=Path, help="验证冻结 Neural STV 三分量训练与独立推理；省略时保留 E2 smoke")
    parser.add_argument("--edge-teacher", type=Path, help="验证传统 NLSTV response 的 V0–V2 路径")
    parser.add_argument("--edge-source-images", type=Path, help="teacher 对应的原始 uint8 images.npy")
    args = parser.parse_args()
    if args.edge_teacher is not None:
        if args.teacher_checkpoint is not None or args.edge_source_images is None:
            parser.error("--edge-teacher 需要 --edge-source-images，且不能同时指定 neural teacher")
        from neuf.edge_field.workflow import run_smoke as run_edge_smoke
        result = run_edge_smoke(args.dataset, args.output_dir, args.edge_teacher, args.edge_source_images)
    elif args.teacher_checkpoint is None:
        result = run_smoke(args.dataset, args.output_dir)
    else:
        started = time.perf_counter()
        stamp = datetime.now().astimezone().isoformat()
        result = None
        metrics_dir = args.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_frozen_smoke(args.dataset, args.output_dir, args.teacher_checkpoint)
        finally:
            timing_path = metrics_dir / "timing.csv"
            with timing_path.open(newline="") if timing_path.is_file() else open('/dev/null') as handle:
                rows = list(csv.DictReader(handle))
            summary = {
                "configured_epochs": 1, "completed_epochs": int(result is not None),
                "configured_total_steps": 2, "completed_total_steps": len(rows),
                "start_timestamp": stamp,
                "end_timestamp": datetime.now().astimezone().isoformat(),
                "total_wall_time_sec": time.perf_counter() - started,
                "mean_step_time_sec": sum(float(row["step_time_sec"]) for row in rows) / max(1, len(rows)),
                "status": "complete" if result is not None else "failed",
                "final_result_path": None if result is None else result["checkpoint"],
                "validation_only": True,
            }
            (metrics_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
