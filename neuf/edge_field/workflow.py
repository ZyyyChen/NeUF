from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data import EdgeData, write_json
from .evaluation import compare, evaluate, export_volume, query, write_csv
from .losses import losses, response_losses
from .model import EdgeField, PoseRefiner, load_field, se3_exp
from .response_evaluation import compare_responses, evaluate_response


def timestamp():
    return datetime.now().astimezone().isoformat()


def checkpoint(model, poses, data, optimizers, config, step):
    return dict(schema="nlstv_response_field_v1" if model.response_only else "nlstv_edge_field_v1", model=model.state_dict(), model_config=model.config,
                bounds=data.bounds.cpu(), poses=poses.state_dict(), corrected_poses=poses.matrices().detach().cpu(),
                frame_ids=data.frame_ids.cpu(), train_indices=data.splits["training"].cpu(),
                optimizer_states=[opt.state_dict() for opt in optimizers], config=config, completed_steps=step,
                frequency_progress=min((step-1)/max(config["steps"]-1, 1)/(.5 if model.response_only else .7), 1.),
                data_metadata=data.metadata, local_grid=data.local.cpu(), mask=data.mask.cpu())


def train_variant(data, output, name, args, *, smoke=False):
    output = Path(output)
    for folder in ("checkpoints", "metrics", "plots", "predictions", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    pose_generator = torch.Generator().manual_seed(args.seed+1)
    device = data.images.device
    response_only = name in ("EdgeFixed", "EdgePose")
    model = EdgeField(data.bounds, bands=args.bands, width=args.width, response_only=response_only,
                      plane_resolutions=args.plane_resolutions if response_only else ()).to(device)
    poses = PoseRefiner(data.initial, data.splits["training"].cpu(), data.bounds.mean(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pose_optimizer = torch.optim.Adam([poses.raw], lr=args.pose_lr)
    config = dict(vars(args), variant=name, equal_budget="same field updates and pixel batches; pose updates separately counted")
    if response_only:
        config.update(supervision="traditional teacher response only; original images used solely for input integrity checks",
                      response_loss_weights=dict(response=1., shape=.1, detail=.25),
                      pose_loss_weights=dict(response=1., shape=.5, prior=.01),
                      pose_start=.35, pose_end=.7, pose_ready_correlation_loss=.8,
                      pose_batches="independent stream; not a leave-one-frame-out registration guarantee")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    write_json(output / "run_config" / "config.json", config)
    write_json(output / "run_config" / "comparison_manifest.json", data.metadata)
    start, stamp = time.perf_counter(), timestamp()
    complete, pose_updates, status, pose_grad = 0, 0, "failed", 0.
    history, evaluated = [], []
    path = output / "checkpoints" / "latest.pt"
    progress_bar = tqdm(total=args.steps, desc=name, mininterval=20, file=__import__("sys").stdout)
    with (output / "metrics" / "timing.csv").open("w", newline="") as timing_file:
        loss_keys = ("response", "shape", "detail") if response_only else ("gray", "edge", "couple")
        columns = ["epoch", "global_step", "step_time_sec", "elapsed_sec", "loss", *loss_keys, "pose_updates", "pose_gradient_norm"]
        writer = csv.DictWriter(timing_file, fieldnames=columns)
        writer.writeheader()
        try:
            for step in range(1, args.steps + 1):
                step_start = time.perf_counter()
                ratio = (step-1) / max(args.steps-1, 1)
                frequency_progress = min(ratio / (.5 if response_only else .7), 1.)
                patch = data.patches(args.patches, args.patch_size, generator)
                poses.raw.requires_grad_(False)
                model.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                xyz = poses(patch["frames"], patch["local"])
                channels = 1 if response_only else 2
                prediction = model(xyz, frequency_progress).transpose(1, 2).reshape(args.patches, channels, args.patch_size, args.patch_size)
                if response_only:
                    values = response_losses(prediction, patch, ratio)
                    loss = values["response"] + .1*values["shape"] + .25*values["detail"]
                    if step == 1:
                        # 删除原图后损失必须完全相同，避免灰度监督意外回流。
                        without_gray = {k: v for k, v in patch.items() if k != "image"}
                        checked = response_losses(prediction, without_gray, ratio)
                        for key in values:
                            torch.testing.assert_close(values[key], checked[key], atol=0, rtol=0)
                        assert not hasattr(model, "gray") and prediction.shape[1] == 1
                else:
                    values = losses(prediction, patch, ratio, use_edges=name != "V0")
                    loss = values["gray"] + args.edge_weight * values["edge"] + args.couple_weight * values["couple"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name} step {step}: non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * (1 - .9 * ratio)
                ready = smoke or (history and np.mean([r["shape"] for r in history[-100:]]) < .8) if response_only else True
                pose_stage = ((name == "V2" and .2 <= ratio < .7)
                              or (name == "EdgePose" and .35 <= ratio < .7 and ready))
                if pose_stage and step % args.pose_every == 0:
                    model.requires_grad_(False)
                    poses.raw.requires_grad_(True)
                    pose_optimizer.zero_grad(set_to_none=True)
                    pose_patch = data.patches(args.patches, args.patch_size, pose_generator) if response_only else patch
                    xyz = poses(pose_patch["frames"], pose_patch["local"])
                    prediction = model(xyz, frequency_progress).transpose(1, 2).reshape(args.patches, channels, args.patch_size, args.patch_size)
                    if response_only:
                        pose_values = response_losses(prediction, pose_patch, ratio, pose_step=True)
                        pose_loss = pose_values["response"] + .5*pose_values["shape"]
                    else:
                        pose_values = losses(prediction, pose_patch, ratio, use_edges=True, pose_step=True)
                        pose_loss = pose_values["edge"] + .2 * pose_values["gray"]
                    pose_loss = pose_loss + .01 * poses.prior(data.splits["training"], data.frame_ids)
                    pose_loss.backward()
                    pose_grad = float(torch.nn.utils.clip_grad_norm_([poses.raw], 1., error_if_nonfinite=True))
                    pose_optimizer.step()
                    pose_updates += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
                duration = time.perf_counter() - step_start
                complete = step
                row = dict(epoch=1, global_step=step, step_time_sec=duration, elapsed_sec=time.perf_counter()-start,
                           loss=float(loss.detach()), **{k: float(v.detach()) for k, v in values.items()},
                           pose_updates=pose_updates, pose_gradient_norm=pose_grad)
                writer.writerow(row)
                history.append(row)
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{row['loss']:.5f}", dt=f"{duration:.3f}s", epoch="1/1", pose=pose_updates, refresh=False)
                if step % 100 == 0:
                    timing_file.flush()
                if step in {args.steps//2, args.steps}:
                    torch.save(checkpoint(model, poses, data, (optimizer, pose_optimizer), config, step), path)
                    if not smoke:
                        evaluator = evaluate_response if response_only else evaluate
                        evaluated = evaluator(model, poses, data, output, step, frequency_progress, final=step == args.steps)
                        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
                        for key in loss_keys:
                            series = np.array([r[key] for r in history])
                            window = min(100, len(series))
                            ax.plot(np.arange(window, len(series)+1), np.convolve(series, np.ones(window)/window, mode="valid"), label=key)
                        ax.set(xlabel="Field update", ylabel="Loss (moving mean)", title=f"{name}; pose updates={pose_updates}")
                        ax.legend()
                        fig.savefig(output / "plots" / f"loss_{step:06d}.png", dpi=130)
                        plt.close(fig)
            # checkpoint 重载用独立模型查询，同一坐标必须得到相同结果。
            restored, saved = load_field(path, device)
            points = data.local.reshape(-1, 3)[::4096]
            points = points @ data.initial[0, :3, :3].T + data.initial[0, :3, 3]
            with torch.no_grad():
                torch.testing.assert_close(restored(points), model(points), atol=1e-7, rtol=1e-5)
                heldout = torch.cat((data.splits["validation"], data.splits["test"]))
                torch.testing.assert_close(poses.matrices()[heldout], data.initial[heldout], atol=1e-5, rtol=1e-6)
                anchor = data.splits["training"][0]
                torch.testing.assert_close(poses.matrices()[anchor], data.initial[anchor], atol=1e-5, rtol=1e-6)
            if name in ("V0", "V1", "EdgeFixed"):
                assert poses.raw.count_nonzero() == 0
            if name in ("V2", "EdgePose") and (pose_updates == 0 or pose_grad <= 0 or not poses.raw.count_nonzero()):
                raise AssertionError(f"{name} 位姿路径未实际更新：检查结构场是否达到就绪条件")
            corrected = saved["corrected_poses"].cpu().numpy()
            np.savez_compressed(output / "predictions" / "poses.npz", frame_ids=data.frame_ids.cpu().numpy(),
                                initial=data.initial.cpu().numpy(), corrected=corrected,
                                train_indices=data.splits["training"].cpu().numpy(), twists=poses.twists().detach().cpu().numpy())
            pose_rows = []
            for i, matrix in enumerate(corrected):
                initial = data.initial[i].cpu().numpy()
                angle = math.degrees(float(poses.twists()[i, 3:].detach().norm()))
                pose_rows.append(dict(frame_id=int(data.frame_ids[i]), translation_change_mm=float(np.linalg.norm(matrix[:3,3]-initial[:3,3])), rotation_change_deg=angle))
            write_csv(output / "metrics" / "pose_changes.csv", pose_rows)
            if not smoke:
                export_volume(restored, data, output, args.volume_spacing,
                              corrected_poses=poses.matrices() if response_only else None)
            status = "complete"
        finally:
            progress_bar.close()
            timing_file.flush()
            write_json(output / "metrics" / "training_summary.json", dict(
                status=status, configured_epochs=1, completed_epochs=int(complete == args.steps),
                configured_total_steps=args.steps, completed_total_steps=complete, pose_updates=pose_updates,
                start_timestamp=stamp, end_timestamp=timestamp(), total_wall_time_sec=time.perf_counter()-start,
                mean_step_time_sec=float(np.mean([r["step_time_sec"] for r in history])) if history else None,
                final_checkpoint_path=str(path) if path.exists() else None,
                final_result_path=str(output / "predictions"), smoke_only=smoke,
                real_pose_accuracy="unknown: corrections are not errors against ground truth"))
    return dict(variant=name, checkpoint=str(path), pose_updates=pose_updates, status=status, evaluated_images=len(evaluated))


def geometry_check(device):
    """已知刚体变换的必要回归；不能作为真实超声配准质量证据。"""
    twist = torch.zeros(2, 6, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(se3_exp, (twist,), fast_mode=True)
    true = torch.tensor([[.3, -.2, .1, .015, -.01, .02]], device=device)
    points = torch.tensor([[1., 2., 3.], [-4., 2., 0.], [2., -3., 1.], [0., 1., -4.]], device=device)
    matrix = se3_exp(true)[0]
    target = points @ matrix[:3, :3].T + matrix[:3, 3]
    estimated = torch.zeros_like(true, requires_grad=True)
    optimizer = torch.optim.LBFGS([estimated], max_iter=40, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        m = se3_exp(estimated)[0]
        loss = (points @ m[:3,:3].T + m[:3,3] - target).square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    error = float((se3_exp(estimated)-se3_exp(true)).abs().max().detach())
    if error > 1e-4:
        raise AssertionError(f"SE3 已知变换恢复失败 {error}")
    return dict(zero_twist_gradcheck=True, known_transform_max_matrix_error=error,
                interpretation="geometry implementation check only, not ultrasound pose recovery accuracy")


def run_smoke(dataset_path, output_dir, teacher, source_images):
    args = defaults()
    args.dataset, args.teacher, args.source_images = dataset_path, teacher, source_images
    args.output = output_dir
    args.steps, args.patches, args.patch_size, args.pose_every = 10, 2, 48, 1
    args.width, args.bands = 64, 6
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geometry = geometry_check(device)
    data = EdgeData(dataset_path, teacher, source_images, device=device)
    result = [train_variant(data, output_dir/name, name, args, smoke=True) for name in ("V0", "V1", "V2")]
    result = dict(status="PASS; smoke only", geometry=geometry, variants=result,
                  data_alignment_frames=len(data.frame_ids), heldout_pose_fixed=True, checkpoint_roundtrip=True,
                  teacher_requires_grad=bool(data.edges.requires_grad), quality="unvalidated")
    write_json(output_dir / "comparison" / "metrics" / "smoke_result.json", result)
    return result


def align_poses(data, args):
    from .model import InPlanePoseRefiner
    from .relations import NeighborEdgeVolume, alignment_checks, alignment_loss
    from .response_evaluation import evaluate_alignment, summarize_alignment

    output = args.output / "EdgeAlign"
    for name in ("run_config", "metrics", "predictions", "plots", "checkpoints"):
        (output/name).mkdir(parents=True, exist_ok=True)
    assert not hasattr(data, "images"), "位姿优化数据不应包含灰度缓存"
    reference = NeighborEdgeVolume(data, max_gap_mm=args.alignment_gap_mm)
    write_json(output/"run_config/geometry.json", reference.metadata)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(supervision="NLSTV response only; gray used only for immutable input identity audit",
                  dof="local axial/lateral translation and rotation about the fixed slice normal",
                  loss_weights=dict(shape=1., response=.5, coverage=2., prior=.01, smooth=.05),
                  donor_geometry="fixed initial world coordinates throughout optimization; target frame excluded",
                  sampling="whole-slice grid, stride 4 with cycling pixel offsets; batch dimension is frames",
                  comparison_indices=[*data.metadata["comparison_indices"], 121],
                  normalization="saved response [0,1]; no additional frame normalization")
    write_json(output/"run_config/config.json", config)
    print(f"EdgeAlign geometry: {reference.metadata['valid_frames']} supported frames, "
          f"{reference.metadata['optimized_frames']} active training poses", flush=True)
    if args.smoke:
        write_json(output/"metrics/geometry_check.json", geometry_check(data.edges.device))
        write_json(output/"metrics/edge_recovery_check.json", alignment_checks(data))
    pool = torch.where(reference.active)[0]
    if not len(pool):
        raise ValueError("没有双侧邻域支持；已保存 geometry.json，不能从这些 edge 推断面内修正")
    pose = InPlanePoseRefiner(data.initial, reference.active, reference.center,
                             translation_mm=args.alignment_translation_mm,
                             rotation_deg=args.alignment_rotation_deg)
    optimizer = torch.optim.Adam([pose.raw], lr=args.alignment_lr)
    generator = torch.Generator().manual_seed(args.seed)
    started, stamp, complete, status = time.perf_counter(), timestamp(), 0, "failed"
    history = []
    before = evaluate_alignment(data, reference, pose, output, 0, smoke=args.smoke)
    bar = tqdm(total=args.steps, desc="EdgeAlign", mininterval=20, file=__import__("sys").stdout)
    path = output/"predictions/poses.npz"
    with (output/"metrics/timing.csv").open("w", newline="") as handle:
        columns = ["epoch", "global_step", "step_time_sec", "elapsed_sec", "loss", "shape", "response", "coverage", "prior", "smooth", "gradient_norm", "valid_fraction"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        try:
            for step in range(1, args.steps+1):
                tick = time.perf_counter()
                ratio = (step-1)/max(1, args.steps-1)
                sigma = 2. if ratio < .35 else (1. if ratio < .7 else 0.)
                if reference.sigma != sigma:
                    reference.set_scale(sigma)
                # 整帧分散取样共同约束旋转和平移，避免单个小 patch 的局部平移/旋转退化。
                frames = pool[torch.randint(len(pool), (args.patches,), generator=generator).to(pool.device)]
                rr = torch.arange((step//4) % 4, data.height-3, 4, device=pool.device)
                cc = torch.arange(step % 4, data.width-3, 4, device=pool.device)
                local = data.local[rr[:, None], cc].reshape(1, -1, 3).expand(args.patches, -1, -1)
                optimizer.zero_grad(set_to_none=True)
                prediction, valid, coverage = reference.sample(frames, local, pose.matrices(),
                                                              source_matrices=data.initial)
                target = reference.maps[frames][:, rr[:, None], cc][:, None]
                mask = valid.reshape_as(target)
                terms = alignment_loss(prediction.reshape_as(target), target, mask, coverage.reshape_as(target))
                prior, smooth = pose.prior(data.splits["training"], data.frame_ids)
                loss = terms["shape"]+.5*terms["response"]+2*terms["coverage"]+.01*prior+.05*smooth
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"EdgeAlign step {step}: non-finite loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_([pose.raw], 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.param_groups[0]["lr"] = args.alignment_lr*(1-.8*ratio)
                if data.edges.is_cuda:
                    torch.cuda.synchronize()
                row = dict(epoch=1, global_step=step, step_time_sec=time.perf_counter()-tick,
                           elapsed_sec=time.perf_counter()-started, loss=float(loss.detach()),
                           **{k: float(v.detach()) for k, v in terms.items()}, prior=float(prior.detach()),
                           smooth=float(smooth.detach()), gradient_norm=float(grad), valid_fraction=float(mask.float().mean()))
                history.append(row)
                writer.writerow(row)
                complete = step
                bar.update(1)
                bar.set_postfix(loss=f"{row['loss']:.4f}", valid=f"{row['valid_fraction']:.2f}", dt=f"{row['step_time_sec']:.3f}s", epoch="1/1", refresh=False)
                if step % 50 == 0:
                    handle.flush()
                if step in {args.steps//2, args.steps}:
                    matrices = pose.matrices().detach()
                    torch.testing.assert_close(matrices[~reference.active], data.initial[~reference.active], atol=0, rtol=0)
                    torch.testing.assert_close(matrices[:, :3, 2], data.initial[:, :3, 2], atol=1e-6, rtol=0)
                    normal_motion = ((matrices[:, :3, 3]-data.initial[:, :3, 3])*data.initial[:, :3, 2]).sum(-1)
                    assert normal_motion.abs().max() < 1e-4
                    torch.save(dict(schema="nlstv_edge_pose_alignment_v1", poses=pose.state_dict(), config=config,
                                    optimizer=optimizer.state_dict(), step=step), output/"checkpoints/latest.pt")
                    np.savez_compressed(path, initial=data.initial.cpu().numpy(), corrected=matrices.cpu().numpy(),
                                        frame_ids=data.frame_ids.cpu().numpy(), optimized=reference.active.cpu().numpy(),
                                        local_corrections=pose.corrections().detach().cpu().numpy(),
                                        local_center=reference.center.cpu().numpy(), coordinate_units="mm", rotation_units="radian")
                    after = evaluate_alignment(data, reference, pose, output, step, smoke=args.smoke)
                    summarize_alignment(data, reference, pose, output, before, after, history, step)
            status = "complete"
        finally:
            bar.close()
            handle.flush()
            write_json(output/"metrics/training_summary.json", dict(
                status=status, configured_epochs=1, completed_epochs=int(complete == args.steps),
                configured_total_steps=args.steps, completed_total_steps=complete,
                start_timestamp=stamp, end_timestamp=timestamp(), total_wall_time_sec=time.perf_counter()-started,
                mean_step_time_sec=float(np.mean([r["step_time_sec"] for r in history])) if history else None,
                final_result_path=str(path) if path.exists() else None, smoke_only=args.smoke,
                real_pose_accuracy="unknown; only edge consistency and controlled perturbation recovery can be evaluated"))


def parser():
    parser = argparse.ArgumentParser(description="固定 NLSTV teacher 的灰度/response 三维场与位姿修正")
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    parser.add_argument("--dataset", type=Path, default=root/"data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original/baked_dataset_physical.pkl")
    parser.add_argument("--teacher", type=Path, default=workspace/"NLSTV/logs/20260903_train03/cerebral/index_0/nlstv_lambda009/predictions/cerebral_edges.mat")
    parser.add_argument("--source-images", type=Path, default=workspace/"UltraNeRF-Studio/data/cerebral/patient_0_convex/images.npy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--bands", type=int, default=10)
    parser.add_argument("--plane-resolutions", nargs="*", type=int, default=[], help="response 场的多分辨率特征平面；空值沿用纯 MLP")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pose-lr", type=float, default=2e-3)
    parser.add_argument("--pose-every", type=int, default=5)
    parser.add_argument("--edge-weight", type=float, default=.2)
    parser.add_argument("--couple-weight", type=float, default=.02)
    parser.add_argument("--volume-spacing", type=float, default=.75)
    parser.add_argument("--variants", nargs="+", choices=("V0", "V1", "V2", "EdgeFixed", "EdgePose"), default=["V0", "V1", "V2"])
    parser.add_argument("--smoke", action="store_true", help="已有轻量检查的 response 路径，不评价重建质量")
    parser.add_argument("--checkpoint", type=Path, help="独立推理使用 checkpoint 与世界坐标 points.npy，不读取 teacher")
    parser.add_argument("--points", type=Path)
    parser.add_argument("--align-poses", action="store_true", help="仅使用其他切片的 edge 插值优化面内探头坐标")
    parser.add_argument("--alignment-gap-mm", type=float, default=3.)
    parser.add_argument("--alignment-translation-mm", type=float, default=1.)
    parser.add_argument("--alignment-rotation-deg", type=float, default=1.)
    parser.add_argument("--alignment-lr", type=float, default=.005)
    return parser


def defaults():
    return parser().parse_args([])


def main():
    args = parser().parse_args()
    if args.output is None:
        raise ValueError("必须指定 --output")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        if args.points is None:
            raise ValueError("独立推理需要 --points，世界坐标单位 mm")
        model, _ = load_field(args.checkpoint, device)
        prediction = query(model, torch.tensor(np.load(args.points), dtype=torch.float32, device=device))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, prediction)
        return
    if args.steps < 10 or args.patch_size < 48 or args.patches < 1 or args.pose_every < 1 or args.volume_spacing <= 0:
        raise ValueError("要求 steps>=10、patch_size>=48、patches/pose_every>=1、volume_spacing>0")
    response_only = all(name in ("EdgeFixed", "EdgePose") for name in args.variants)
    if not response_only and any(name in ("EdgeFixed", "EdgePose") for name in args.variants):
        raise ValueError("灰度和 response 实验分开运行，禁止混用评价目标")
    if any(r < 2 for r in args.plane_resolutions) or (args.plane_resolutions and not response_only):
        raise ValueError("特征平面仅用于 response 场，且分辨率至少为 2")
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有实验目录: {args.output}")
    args.output.mkdir(parents=True)
    source = args.output/"comparison/run_config/source"
    source.mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source/path.name)
    if args.align_poses and any(not math.isfinite(v) or v <= 0 for v in (
            args.alignment_gap_mm, args.alignment_translation_mm, args.alignment_rotation_deg, args.alignment_lr)):
        raise ValueError("位姿优化间距、幅度限制和学习率必须为正有限数")
    data = EdgeData(args.dataset, args.teacher, args.source_images, device=device, cache_images=not args.align_poses)
    write_json(args.output/"comparison/run_config/data_manifest.json", data.metadata)
    if args.align_poses:
        align_poses(data, args)
        return
    geometry = geometry_check(device) if args.smoke else None
    results = [train_variant(data, args.output/name, name, args, smoke=args.smoke) for name in args.variants]
    if args.smoke:
        write_json(args.output/"comparison/metrics/smoke_result.json", dict(
            status="PASS; execution only", geometry=geometry, variants=results,
            response_only=response_only, gray_independence_checked=response_only,
            quality="not evaluated"))
    else:
        comparison = compare_responses if response_only else compare
        comparison(data, args.output, args.variants)


if __name__ == "__main__":
    main()
