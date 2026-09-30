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

from .data import EdgeData, SagittalIntersection, write_json
from .evaluation import compare, compare_sweeps, evaluate, evaluate_sweep, export_volume, query, write_csv
from .losses import edge_guided_losses, edge_preserving_losses, losses, response_losses
from .model import EdgeField, PoseRefiner, load_field, se3_exp
from .response_evaluation import compare_responses, evaluate_response
from .sweep_geometry import SweepPoseRefiner, load_sweep


SWEEP_VARIANTS = {"HashObservedL1Angle": "angle", "HashObservedL1Velocity": "velocity"}


def timestamp():
    return datetime.now().astimezone().isoformat()


def checkpoint(model, poses, data, optimizers, config, step, *, generators=None):
    return dict(schema="nlstv_response_field_v1" if model.response_only else "nlstv_edge_field_v1", model=model.state_dict(), model_config=model.config,
                bounds=data.bounds.cpu(), poses=poses.state_dict(), corrected_poses=poses.matrices().detach().cpu(),
                frame_ids=data.frame_ids.cpu(), train_indices=data.splits["training"].cpu(),
                optimizer_states=[opt.state_dict() for opt in optimizers], config=config, completed_steps=step,
                generator_states={name: generator.get_state() for name, generator in (generators or {}).items()},
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                frequency_progress=min((step-1)/max(config["steps"]-1, 1)/(.5 if model.response_only else .7), 1.),
                data_metadata=data.metadata, local_grid=data.local.cpu(), mask=data.mask.cpu(),
                sweep_metadata=poses.metadata if isinstance(poses, SweepPoseRefiner) else None)


def train_variant(data, output, name, args, *, smoke=False, preview_steps=(), sagittal=None):
    output = Path(output)
    stop_step = args.stop_step or args.steps
    # 按完整训练预算向上取整；中途停止不改变原定的 50% 和 75% 位置。
    halfway_step = (args.steps + 1) // 2
    checkpoint_steps = {halfway_step, (3 * args.steps + 3) // 4, stop_step}
    extra_checkpoint_steps = getattr(args, "checkpoint_steps", [])
    if any(not 1 <= step <= stop_step for step in extra_checkpoint_steps):
        raise ValueError("额外 checkpoint 步数必须在本次实际训练范围内")
    checkpoint_steps.update(extra_checkpoint_steps)
    preview_steps = tuple(sorted(set(preview_steps)))
    if any(not isinstance(step, int) or not 1 <= step <= args.steps for step in preview_steps):
        raise ValueError("预览步数必须是训练范围内的整数")
    for folder in ("checkpoints", "metrics", "plots", "predictions", "run_config"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    pose_generator = torch.Generator().manual_seed(args.seed+1)
    spatial_generator = torch.Generator().manual_seed(args.seed+2)
    device = data.images.device
    response_only = name in ("EdgeFixed", "EdgePose")
    sweep_mode = SWEEP_VARIANTS.get(name)
    pure_l1 = name == "HashObservedL1" or sweep_mode is not None
    preserve_edges = name in ("HashObservedEdgePreserve", "HashObservedEdgePreserve3D")
    guided_names = (
        "ObservedUniform", "ObservedEdgeSample", "ObservedEdgeGuided",
        "HashObservedL1", "HashObservedUniform", "HashObservedEdgeSample", "HashObservedEdgeGuided",
        "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
        "HashObservedEdgeGatedSharpProfile",
    )
    guided_experiment = name in guided_names or preserve_edges or sweep_mode is not None
    guided_sampling = preserve_edges or name in (
        "ObservedEdgeSample", "ObservedEdgeGuided",
        "HashObservedEdgeSample", "HashObservedEdgeGuided",
        "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
        "HashObservedEdgeGatedSharpProfile",
    )
    guided_loss = preserve_edges or name in (
        "ObservedEdgeGuided", "HashObservedEdgeGuided",
        "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
        "HashObservedEdgeGatedSharpProfile",
    )
    edge_conditioned = preserve_edges or name in (
        "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
        "HashObservedEdgeGatedSharpProfile",
    )
    sharpness_loss = name in ("HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
                              "HashObservedEdgeGatedSharpProfile")
    focus_observed_edges = name == "HashObservedEdgeGatedSharpFocused"
    profile_loss = name == "HashObservedEdgeGatedSharpProfile"
    model = EdgeField(
        data.bounds, bands=args.bands, width=args.width, response_only=response_only,
        plane_resolutions=args.plane_resolutions if response_only else (), encoding=args.encoding,
        hash_levels=args.hash_levels, hash_features=args.hash_features,
        log2_hashmap_size=args.log2_hashmap_size,
        hash_base_resolution=args.hash_base_resolution,
        hash_finest_resolution=args.hash_finest_resolution,
        edge_conditioned=edge_conditioned,
        coarse_hash_levels=args.coarse_hash_levels,
        detail_scale=args.detail_scale,
    ).to(device)
    if sweep_mode:
        poses = SweepPoseRefiner(
            data.initial, data.splits["training"], data.frame_ids,
            radial_offset_mm=args.sweep_radial_offset_px * data.spacing[0], mode=sweep_mode,
        ).to(device)
    else:
        poses = PoseRefiner(data.initial, data.splits["training"].cpu(), data.bounds.mean(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pose_optimizer = torch.optim.Adam([poses.raw], lr=args.sweep_pose_lr if sweep_mode else args.pose_lr)
    spatial_regularizer = None
    if name == "HashObservedEdgePreserve3D":
        from .relations import EdgeAwareSpatialRegularizer
        spatial_regularizer = EdgeAwareSpatialRegularizer(data)
    config = dict(
        vars(args), variant=name, preview_steps=list(preview_steps),
        checkpoint_milestones=sorted(step for step in checkpoint_steps if step <= stop_step),
        equal_budget="same field updates and pixel batches; pose updates separately counted",
        architecture=dict(
            input=("multiresolution trilinear 3D HashGrid" if model.encoding == "hash"
                   else "normalized world xyz plus coarse-to-fine Fourier features"),
            encoding=model.encoding,
            encoded_dimension=model.encoded_dimension,
            edge_conditioned=edge_conditioned,
            coarse_encoded_dimension=model.coarse_encoded_dimension,
            trunk=(
                f"coarse {model.config['layers']}-layer trunk from first {model.config['coarse_hash_levels']} hash levels; "
                f"full-resolution {model.config['layers']}-layer detail trunk; width {model.config['width']}, SiLU"
                if edge_conditioned else
                f"{model.config['layers']} fully connected layers, width {model.config['width']}, SiLU"
            ),
            heads=(
                "coarse gray logit + predicted NLSTV gate * bounded fine residual; edge gate is supervised"
                if edge_conditioned else
                "shared trunk -> sigmoid gray head and sigmoid NLSTV response head"
            ),
            trainable_field_parameters=sum(parameter.numel() for parameter in model.parameters()),
            feature_planes=list(model.config["plane_resolutions"]),
        ),
    )
    if response_only:
        config.update(supervision="traditional teacher response only; original images used solely for input integrity checks",
                      response_loss_weights=dict(response=1., shape=.1, detail=.25),
                      pose_loss_weights=dict(response=1., shape=.5, prior=.01),
                      pose_start=.35, pose_end=.7, pose_ready_correlation_loss=.8,
                      pose_batches="independent stream; not a leave-one-frame-out registration guarantee")
    elif guided_experiment:
        config.update(
            supervision=(
                "observed B-mode intensity; supervised NLSTV response gates the learned high-frequency gray residual"
                if edge_conditioned else
                "observed B-mode intensity; NLSTV response is an auxiliary target and soft gradient weight"
            ),
            loss_weights=dict(
                photo=1., gradient=.1 if guided_loss else 0., edge=.02 if guided_loss else 0.,
                sharpness=args.sharpness_weight if sharpness_loss else 0.,
            ),
            response_weight="1 + 2 * response",
            sharpness_target="edge-response-weighted observed B-mode Laplacian" if sharpness_loss else None,
            gradient_focus=(
                "base gradient + 0.5 * mean gradient error where NLSTV response > 0.15 "
                "and observed gray gradient magnitude > 0.01; target is observed gray"
                if focus_observed_edges else None
            ),
            normal_profile=(
                "observed-gray normal-direction +/-1,2,3 native-pixel contrast; response > 0.15 "
                "and Gaussian-smoothed observed gradient > 0.01 select valid fan-interior centers; weight 0.1"
                if profile_loss else None
            ),
            edge_warmup="zero through 10%; linear ramp to full weight at 20%",
            edge_centered_patch_fraction=args.guided_fraction if guided_sampling else 0.,
            fixed_poses=True,
        )
    if pure_l1:
        config.update(
            supervision="observed B-mode intensity only; strict masked L1",
            photo_loss="mean(abs(gray - observed)) over valid mask",
            response_weight=None, edge_warmup=None, auxiliary_response_supervised=False,
            teacher_usage="loader identity checks and diagnostics only; no loss, gate, or edge sampling",
        )
    if sagittal is not None:
        config.update(
            pose_mode=sweep_mode or "fixed", fixed_poses=sweep_mode is None,
            sagittal=sagittal.metadata,
            sweep_geometry=poses.metadata if sweep_mode else None,
            sweep_pose_loss=dict(photo=1., intersection=args.sagittal_weight, prior=args.sweep_prior_weight),
            sweep_schedule=dict(start=.2, end=.7, every=args.pose_every),
            angular_velocity_units="radian / original frame; acquisition timestamps unavailable",
            heldout_poses="interpolated from training corrections; held-out images excluded from losses" if sweep_mode else "fixed input",
            sagittal_role="training reference for angle/velocity variants; not independent test ground truth",
            equal_budget="same field initialization, field batches and field updates; extra pose updates/time reported",
        )
    if preserve_edges:
        config.update(
            equal_budget="same field updates and supervised pixel patches; extra spatial queries recorded separately",
            loss_weights=dict(photo=1., gradient=args.preserve_edge_weight,
                              profile=args.preserve_profile_weight, edge=.02,
                              spatial=args.spatial_weight if spatial_regularizer else 0.),
            response_weight="fixed E * (E > 0.15) * (smoothed observed gradient > 0.01); normalized per region",
            sharpness_target=None, gradient_focus="edge region only; no background gradient matching",
            normal_profile="soft-E-weighted observed normal contrasts at +/-1,2,3 native pixels",
            spatial_regularization=dict(
                enabled=spatial_regularizer is not None, points=args.spatial_points,
                step_mm=args.spatial_step_mm, tangent_weight=args.spatial_tangent_weight,
                guidance="fixed training observations only; two-sided support and five-point edge protection",
                support=spatial_regularizer.metadata if spatial_regularizer is not None else None,
                limitation="weak local smoothness; no anatomical surface connectivity guarantee",
            ),
        )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    write_json(output / "run_config" / "config.json", config)
    write_json(output / "run_config" / "comparison_manifest.json", data.metadata)
    start, stamp = time.perf_counter(), timestamp()
    complete, pose_updates, status, pose_grad = 0, 0, "failed", 0.
    history, evaluated = [], []
    if sagittal is not None and not smoke:
        evaluate_sweep(model, poses, data, sagittal, output, 0, 0.)
    path = output / "checkpoints" / "latest.pt"
    progress_bar = tqdm(total=stop_step, desc=name, mininterval=20, file=__import__("sys").stdout)
    with (output / "metrics" / "timing.csv").open("w", newline="") as timing_file:
        if response_only:
            loss_keys = ("response", "shape", "detail")
        elif preserve_edges:
            loss_keys = ("photo", "gradient", "profile", "edge", "ramp", "spatial",
                         "spatial_background", "spatial_tangent", "spatial_valid_fraction",
                         "spatial_active_fraction")
        elif guided_experiment:
            loss_keys = ("photo", "gradient", "edge", "sharpness", "ramp")
            if profile_loss:
                loss_keys += ("profile",)
        else:
            loss_keys = ("gray", "edge", "couple")
        columns = ["epoch", "global_step", "step_time_sec", "elapsed_sec", "loss", *loss_keys,
                   "edge_centered_fraction", "pose_updates", "pose_gradient_norm"]
        if sagittal is not None:
            columns += ["pose_photo", "sagittal_loss", "sagittal_edge_ncc", "sagittal_ssim",
                        "sagittal_coverage", "pose_prior", "angle_step_max_deg", "angle_step_rms_deg"]
        writer = csv.DictWriter(timing_file, fieldnames=columns)
        writer.writeheader()
        try:
            for step in range(1, stop_step + 1):
                step_start = time.perf_counter()
                pose_metrics = {key: 0. for key in columns if key.startswith(("pose_photo", "sagittal_", "pose_prior", "angle_step_"))}
                ratio = (step-1) / max(args.steps-1, 1)
                frequency_progress = min(ratio / (.5 if response_only else .7), 1.)
                sampling_fraction = args.guided_fraction if guided_sampling else 0.
                patch = data.patches(
                    args.patches, args.patch_size, generator,
                    edge_fraction=sampling_fraction,
                )
                poses.raw.requires_grad_(False)
                model.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                xyz = poses(patch["frames"], patch["local"])
                channels = 1 if response_only else 2
                prediction = model(xyz, frequency_progress).transpose(1, 2).reshape(args.patches, channels, args.patch_size, args.patch_size)
                if smoke and edge_conditioned and step == 1:
                    # 只对灰度求导，确认 edge head 确实位于重建路径而非仅共享 trunk。
                    gate_gradient = torch.autograd.grad(
                        prediction[:, :1].sum(), model.edge.weight, retain_graph=True,
                    )[0]
                    if not torch.isfinite(gate_gradient).all() or gate_gradient.abs().sum() <= 0:
                        raise AssertionError("edge head 未实际控制灰度输出")
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
                elif preserve_edges:
                    values = edge_preserving_losses(prediction, patch)
                    ramp = min(1., max(0., (ratio - .1) / .1))
                    values["ramp"] = values["photo"].new_tensor(ramp)
                    zero = prediction.sum() * 0
                    spatial_values = dict.fromkeys(
                        ("spatial", "spatial_background", "spatial_tangent",
                         "spatial_valid_fraction", "spatial_active_fraction"), zero)
                    if spatial_regularizer is not None and ramp > 0:
                        spatial_values = spatial_regularizer.loss(
                            model, patch, spatial_generator, points=args.spatial_points,
                            step_mm=args.spatial_step_mm, tangent_weight=args.spatial_tangent_weight,
                        )
                    values.update(spatial_values)
                    loss = values["photo"] + ramp * (
                        args.preserve_edge_weight * values["gradient"]
                        + args.preserve_profile_weight * values["profile"]
                        + .02 * values["edge"] + args.spatial_weight * values["spatial"]
                    )
                elif guided_experiment:
                    values = edge_guided_losses(
                        prediction, patch, ratio, use_guidance=guided_loss,
                        use_sharpness=sharpness_loss, focus_observed_edges=focus_observed_edges,
                        use_profile=profile_loss, use_l1=pure_l1,
                    )
                    loss = values["photo"] + values["ramp"] * (
                        .1 * values["gradient"] + .02 * values["edge"]
                        + args.sharpness_weight * values["sharpness"]
                        + (.1 * values["profile"] if profile_loss else 0.)
                    )
                    if pure_l1 and step == 1:
                        # 首个真实批次确认灰度绝对误差、均匀采样以及 teacher 独立性。
                        expected = (prediction[:, :1] - patch["image"]).abs()[patch["mask"]].mean()
                        torch.testing.assert_close(loss, expected)
                        without_teacher = {key: value for key, value in patch.items() if key != "edge"}
                        checked = edge_guided_losses(prediction, without_teacher, ratio,
                                                     use_guidance=False, use_l1=True)
                        torch.testing.assert_close(checked["photo"], loss, atol=0, rtol=0)
                        assert not model.edge_conditioned and patch["edge_centered_count"] == 0
                        assert all(float(values[key].detach()) == 0 for key in ("gradient", "edge", "sharpness", "ramp"))
                        print("Strict L1 first-batch check: PASS; no gate, edge sampling, or teacher loss", flush=True)
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
                              or (name == "EdgePose" and .35 <= ratio < .7 and ready)
                              or (sweep_mode is not None and .2 <= ratio < .7))
                if pose_stage and step % args.pose_every == 0:
                    model.requires_grad_(False)
                    poses.raw.requires_grad_(True)
                    pose_optimizer.zero_grad(set_to_none=True)
                    pose_patch = data.patches(args.patches, args.patch_size, pose_generator) if response_only or sweep_mode else patch
                    xyz = poses(pose_patch["frames"], pose_patch["local"])
                    prediction = model(xyz, frequency_progress).transpose(1, 2).reshape(args.patches, channels, args.patch_size, args.patch_size)
                    if sweep_mode:
                        # 冻结网络参数但保留坐标梯度；运动更新同时受场重建和实采交线约束。
                        pose_photo = (prediction[:, :1] - pose_patch["image"]).abs()[pose_patch["mask"]].mean()
                        structure = sagittal.loss(poses.matrices(), data.splits["training"], ratio)
                        prior = poses.prior()
                        pose_loss = pose_photo + args.sagittal_weight * structure["loss"] + args.sweep_prior_weight * prior
                        previous_raw = poses.raw.detach().clone()
                        if smoke and pose_updates == 0:
                            # 分别确认图像场和独立 sagittal 都能更新轨迹，避免只有先验在反传。
                            for label, term in (("field_photo", pose_photo), ("sagittal_edge_ncc", structure["edge_ncc"])):
                                gradient = torch.autograd.grad(term, poses.raw, retain_graph=True)[0]
                                if not torch.isfinite(gradient).all() or gradient.abs().sum() <= 0:
                                    raise AssertionError(f"{name}: {label} 未提供有效角度梯度")
                            print(f"{name}: field and sagittal coordinate gradients PASS", flush=True)
                        pose_metrics.update(pose_photo=float(pose_photo.detach()),
                                            sagittal_loss=float(structure["loss"].detach()),
                                            sagittal_edge_ncc=float(structure["edge_ncc"].detach()),
                                            sagittal_ssim=float(structure["ssim"].detach()),
                                            sagittal_coverage=float(structure["coverage"].detach()),
                                            pose_prior=float(prior.detach()))
                    elif response_only:
                        pose_values = response_losses(prediction, pose_patch, ratio, pose_step=True)
                        pose_loss = pose_values["response"] + .5*pose_values["shape"]
                    else:
                        pose_values = losses(prediction, pose_patch, ratio, use_edges=True, pose_step=True)
                        pose_loss = pose_values["edge"] + .2 * pose_values["gray"]
                    if not sweep_mode:
                        pose_loss = pose_loss + .01 * poses.prior(data.splits["training"], data.frame_ids)
                    if not torch.isfinite(pose_loss):
                        raise FloatingPointError(f"{name} step {step}: non-finite pose loss")
                    pose_loss.backward()
                    pose_grad = float(torch.nn.utils.clip_grad_norm_([poses.raw], 1., error_if_nonfinite=True))
                    pose_optimizer.step()
                    if sweep_mode:
                        movement = poses.limit_update_(previous_raw, max_angle_step_deg=args.sweep_max_step_deg,
                                                       max_angle_offset_deg=args.sweep_max_offset_deg)
                        pose_metrics.update(angle_step_max_deg=movement["actual_angle_step_max_deg"],
                                            angle_step_rms_deg=movement["actual_angle_step_rms_deg"])
                    pose_updates += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
                duration = time.perf_counter() - step_start
                complete = step
                row = dict(epoch=1, global_step=step, step_time_sec=duration, elapsed_sec=time.perf_counter()-start,
                           loss=float(loss.detach()), **{k: float(v.detach()) for k, v in values.items()},
                           edge_centered_fraction=patch["edge_centered_count"] / args.patches,
                           pose_updates=pose_updates, pose_gradient_norm=pose_grad, **pose_metrics)
                writer.writerow(row)
                history.append(row)
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{row['loss']:.5f}", dt=f"{duration:.3f}s", epoch="1/1", pose=pose_updates, refresh=False)
                if step % 100 == 0:
                    timing_file.flush()
                if step in checkpoint_steps:
                    payload = checkpoint(model, poses, data, (optimizer, pose_optimizer), config, step,
                                         generators=dict(patch=generator, pose=pose_generator, spatial=spatial_generator))
                    torch.save(payload, path.with_name(f"step_{step:06d}.pt"))
                    torch.save(payload, path)
                    if sagittal is not None and (not smoke or step == stop_step):
                        evaluate_sweep(model, poses, data, sagittal, output, step, frequency_progress)
                if step in {halfway_step, stop_step, *preview_steps}:
                    if not smoke:
                        if response_only:
                            evaluated = evaluate_response(
                                model, poses, data, output, step, frequency_progress,
                                final=step == stop_step,
                            )
                        else:
                            evaluated = evaluate(
                                model, poses, data, output, step, frequency_progress,
                                final=step == stop_step,
                                save_training_preview=bool(preview_steps),
                            )
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
                if not sweep_mode:
                    torch.testing.assert_close(poses.matrices()[heldout], data.initial[heldout], atol=1e-5, rtol=1e-6)
                else:
                    restored_poses = load_sweep(saved, device)
                    torch.testing.assert_close(restored_poses.matrices(), poses.matrices(), atol=1e-6, rtol=1e-6)
                anchor = data.splits["training"][0]
                torch.testing.assert_close(poses.matrices()[anchor], data.initial[anchor], atol=1e-5, rtol=1e-6)
            if preserve_edges or name in (
                "V0", "V1", "EdgeFixed", "ObservedUniform", "ObservedEdgeSample", "ObservedEdgeGuided",
                "HashObservedL1", "HashObservedUniform", "HashObservedEdgeSample", "HashObservedEdgeGuided",
                "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
                "HashObservedEdgeGatedSharpProfile",
            ):
                assert poses.raw.count_nonzero() == 0
            if (name in ("V2", "EdgePose") or sweep_mode) and (pose_updates == 0 or pose_grad <= 0 or not poses.raw.count_nonzero()):
                raise AssertionError(f"{name} 位姿路径未实际更新：检查结构场是否达到就绪条件")
            corrected = saved["corrected_poses"].cpu().numpy()
            pose_payload = dict(frame_ids=data.frame_ids.cpu().numpy(), initial=data.initial.cpu().numpy(),
                                corrected=corrected, train_indices=data.splits["training"].cpu().numpy())
            if sweep_mode:
                pose_payload.update(initial_angles=poses.initial_angles.cpu().numpy(),
                                    corrected_angles=poses.angles().detach().cpu().numpy(),
                                    angular_increments=poses.increments().detach().cpu().numpy())
            else:
                pose_payload["twists"] = poses.twists().detach().cpu().numpy()
            np.savez_compressed(output / "predictions" / "poses.npz", **pose_payload)
            pose_rows = []
            for i, matrix in enumerate(corrected):
                initial = data.initial[i].cpu().numpy()
                angle = math.degrees(float(poses.angle_offsets()[i].detach().abs())) if sweep_mode else math.degrees(float(poses.twists()[i, 3:].detach().norm()))
                pose_rows.append(dict(frame_id=int(data.frame_ids[i]), translation_change_mm=float(np.linalg.norm(matrix[:3,3]-initial[:3,3])), rotation_change_deg=angle))
            write_csv(output / "metrics" / "pose_changes.csv", pose_rows)
            if not smoke:
                export_volume(restored, data, output, args.volume_spacing,
                              corrected_poses=poses.matrices() if response_only or sweep_mode else None)
            status = "complete"
        finally:
            progress_bar.close()
            timing_file.flush()
            write_json(output / "metrics" / "training_summary.json", dict(
                status=status, configured_epochs=1, completed_epochs=int(complete == stop_step),
                configured_total_steps=args.steps, requested_stop_step=stop_step,
                stopped_early=stop_step < args.steps, completed_total_steps=complete, pose_updates=pose_updates,
                start_timestamp=stamp, end_timestamp=timestamp(), total_wall_time_sec=time.perf_counter()-start,
                mean_step_time_sec=float(np.mean([r["step_time_sec"] for r in history])) if history else None,
                final_checkpoint_path=str(path) if path.exists() else None,
                final_result_path=str(output / "predictions"), smoke_only=smoke,
                edge_centered_fraction=(
                    args.guided_fraction if guided_sampling else 0.
                ),
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
    # 复用现有 smoke，确认结束后仍可独立读取两个中期状态。
    for variant in result:
        checkpoint_dir = Path(variant["checkpoint"]).parent
        retained_steps = sorted({(args.steps + 1) // 2, (3 * args.steps + 3) // 4, args.steps})
        for step in retained_steps:
            saved = torch.load(checkpoint_dir / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
            assert saved["completed_steps"] == step
            assert saved["model"] and len(saved["optimizer_states"]) == 2
        variant["retained_checkpoint_steps"] = retained_steps
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
    halfway_step = (args.steps + 1) // 2
    checkpoint_steps = {halfway_step, (3 * args.steps + 3) // 4, args.steps}
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
                if step in checkpoint_steps:
                    payload = dict(schema="nlstv_edge_pose_alignment_v1", poses=pose.state_dict(), config=config,
                                   optimizer=optimizer.state_dict(), step=step)
                    torch.save(payload, output / "checkpoints" / f"step_{step:06d}.pt")
                    torch.save(payload, output / "checkpoints" / "latest.pt")
                if step in {halfway_step, args.steps}:
                    matrices = pose.matrices().detach()
                    torch.testing.assert_close(matrices[~reference.active], data.initial[~reference.active], atol=0, rtol=0)
                    torch.testing.assert_close(matrices[:, :3, 2], data.initial[:, :3, 2], atol=1e-6, rtol=0)
                    normal_motion = ((matrices[:, :3, 3]-data.initial[:, :3, 3])*data.initial[:, :3, 2]).sum(-1)
                    assert normal_motion.abs().max() < 1e-4
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
    parser.add_argument("--stop-step", type=int,
                        help="在指定步数保存最终结果并停止；学习率和损失调度仍以 --steps 为准")
    parser.add_argument("--checkpoint-steps", nargs="*", type=int, default=[],
                        help="额外独立保存的实际步数；不改变学习率、损失日程或默认里程碑")
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--bands", type=int, default=10)
    parser.add_argument("--train-frame", type=int, help="诊断模式：只用指定训练帧拟合 EdgeFixed")
    parser.add_argument("--train-frames", nargs="+", type=int,
                        help="灰度消融：只用指定 observed 帧训练；原测试帧会从测试统计中移除")
    parser.add_argument("--fit-frames", nargs="+", type=int,
                        help="只用指定原始帧共同拟合 EdgeFixed；所选帧从验证/测试集移入训练集")
    parser.add_argument("--encoding", choices=("fourier", "hash"), default="fourier")
    parser.add_argument("--hash-levels", type=int, default=16)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--log2-hashmap-size", type=int, default=19)
    parser.add_argument("--hash-base-resolution", type=int, default=16)
    parser.add_argument("--hash-finest-resolution", type=int, default=512)
    parser.add_argument("--coarse-hash-levels", type=int, default=8)
    parser.add_argument("--detail-scale", type=float, default=2.)
    parser.add_argument("--plane-resolutions", nargs="*", type=int, default=[], help="response 场的多分辨率特征平面；空值沿用纯 MLP")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pose-lr", type=float, default=2e-3)
    parser.add_argument("--pose-every", type=int, default=5)
    parser.add_argument("--sagittal-reference", type=Path,
                        default=root/"data/cerebral_data/Pre_traitement_echo_v2/Repositionnement/Patient0/data_repos_Patient0_J35_2_sag.mat")
    parser.add_argument("--sagittal-calibration", type=Path,
                        default=workspace/"logs/20260924_train13/cerebral/index_all/IntersectionRegistration/run_config/similarity_fixed_support.npz")
    parser.add_argument("--sagittal-weight", type=float, default=.05)
    parser.add_argument("--sagittal-ssim-weight", type=float, default=0.)
    parser.add_argument("--sweep-pose-lr", type=float, default=.01)
    parser.add_argument("--sweep-prior-weight", type=float, default=.002)
    parser.add_argument("--sweep-max-step-deg", type=float, default=.01)
    parser.add_argument("--sweep-max-offset-deg", type=float, default=3.)
    parser.add_argument("--sweep-radial-offset-px", type=float, default=28.1915615,
                        help="原 sagittal 轨迹标定的 delta；乘轴向像素间距得到固定中心到探头原点的毫米距离")
    parser.add_argument("--edge-weight", type=float, default=.2)
    parser.add_argument("--couple-weight", type=float, default=.02)
    parser.add_argument("--volume-spacing", type=float, default=.75)
    parser.add_argument("--guided-fraction", type=float, default=.4)
    parser.add_argument("--sharpness-weight", type=float, default=.05)
    parser.add_argument("--preserve-edge-weight", type=float, default=.1)
    parser.add_argument("--preserve-profile-weight", type=float, default=.1)
    parser.add_argument("--spatial-weight", type=float, default=.001)
    parser.add_argument("--spatial-points", type=int, default=512)
    parser.add_argument("--spatial-step-mm", type=float, default=.24)
    parser.add_argument("--spatial-tangent-weight", type=float, default=.1)
    parser.add_argument("--variants", nargs="+", choices=(
        "V0", "V1", "V2", "EdgeFixed", "EdgePose",
        "ObservedUniform", "ObservedEdgeSample", "ObservedEdgeGuided",
        "HashObservedL1", "HashObservedUniform", "HashObservedEdgeSample", "HashObservedEdgeGuided",
        "HashObservedL1Angle", "HashObservedL1Velocity",
        "HashObservedEdgeGated", "HashObservedEdgeGatedSharp", "HashObservedEdgeGatedSharpFocused",
        "HashObservedEdgeGatedSharpProfile",
        "HashObservedEdgePreserve", "HashObservedEdgePreserve3D",
    ), default=["V0", "V1", "V2"])
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
    if args.stop_step is not None and not 10 <= args.stop_step <= args.steps:
        raise ValueError("--stop-step 必须在 [10, steps] 内")
    if not 0 <= args.guided_fraction <= 1:
        raise ValueError("guided-fraction 必须位于 [0,1]")
    if (args.hash_levels < 2 or args.hash_features < 1 or args.log2_hashmap_size < 1
            or args.hash_base_resolution < 2
            or args.hash_finest_resolution < args.hash_base_resolution):
        raise ValueError("HashGrid 参数要求 levels>=2、features/log2_size>=1、2<=base<=finest")
    if not 1 <= args.coarse_hash_levels < args.hash_levels:
        raise ValueError("要求 1<=coarse-hash-levels<hash-levels")
    if args.detail_scale <= 0 or args.sharpness_weight < 0:
        raise ValueError("detail-scale 必须为正且 sharpness-weight 不能为负")
    if any(not math.isfinite(v) or v < 0 for v in (
            args.preserve_edge_weight, args.preserve_profile_weight,
            args.spatial_weight, args.spatial_tangent_weight)):
        raise ValueError("保边和三维正则权重必须为非负有限数")
    if args.spatial_points < 1 or not math.isfinite(args.spatial_step_mm) or args.spatial_step_mm <= 0:
        raise ValueError("三维采样点数和毫米差分步长必须为正")
    response_only = all(name in ("EdgeFixed", "EdgePose") for name in args.variants)
    sweep_experiment = any(name in SWEEP_VARIANTS for name in args.variants)
    if sweep_experiment:
        if not set(args.variants) <= {"HashObservedL1", *SWEEP_VARIANTS}:
            raise ValueError("平面角度/角速度对照只与相同的纯 L1 基线比较")
        if args.align_poses or args.train_frame is not None or args.train_frames is not None or args.fit_frames is not None:
            raise ValueError("平面轨迹对照保持原训练/验证划分，不能混用其他位姿或选帧模式")
        if any(not math.isfinite(v) or v <= 0 for v in (
                args.sweep_pose_lr, args.sweep_max_step_deg, args.sweep_max_offset_deg, args.sweep_radial_offset_px)):
            raise ValueError("平面轨迹学习率、角度范围与半径须为正有限数")
        if any(not math.isfinite(v) or v < 0 for v in (
                args.sagittal_weight, args.sagittal_ssim_weight, args.sweep_prior_weight)):
            raise ValueError("sagittal 和轨迹先验权重须为非负有限数")
    if not response_only and any(name in ("EdgeFixed", "EdgePose") for name in args.variants):
        raise ValueError("灰度和 response 实验分开运行，禁止混用评价目标")
    if args.train_frame is not None and (args.variants != ["EdgeFixed"] or args.align_poses):
        raise ValueError("单帧拟合只支持 EdgeFixed，且不启用位姿优化")
    if args.train_frames is not None and (args.train_frame is not None or args.align_poses or response_only):
        raise ValueError("--train-frames 只支持灰度模型，且不能与单帧 response 或位姿优化混用")
    if args.train_frames is not None and len(args.train_frames) != len(set(args.train_frames)):
        raise ValueError("--train-frames 中帧编号重复")
    if args.fit_frames is not None:
        if args.train_frame is not None or args.variants not in (["EdgeFixed"], ["EdgePose"]) or args.align_poses:
            raise ValueError("多帧拟合只支持 EdgeFixed 或 EdgePose，且不能与单帧或独立位姿对齐同时使用")
        if len(args.fit_frames) != len(set(args.fit_frames)):
            raise ValueError("拟合帧编号不能重复")
    if any(r < 2 for r in args.plane_resolutions) or (args.plane_resolutions and not response_only):
        raise ValueError("特征平面仅用于 response 场，且分辨率至少为 2")
    if args.encoding == "hash" and args.plane_resolutions:
        raise ValueError("HashGrid 与 feature planes 禁止混用")
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
    if args.train_frame is not None:
        training = data.splits["training"]
        selected = training[data.frame_ids[training] == args.train_frame]
        if selected.numel() != 1:
            raise ValueError(f"帧 {args.train_frame} 不在原始训练集内")
        # 保持全数据的世界包围盒与固定验证帧，只限制优化时抽取的训练帧。
        data.splits["training"] = selected
        data.metadata["splits"]["training"] = [args.train_frame]
        data.metadata["train_frame_only"] = args.train_frame
    if args.train_frames is not None:
        lookup = {int(frame): index for index, frame in enumerate(data.frame_ids.cpu().tolist())}
        if any(frame not in lookup for frame in args.train_frames):
            raise ValueError("--train-frames 包含数据集中不存在的帧")
        selected = torch.tensor([lookup[frame] for frame in args.train_frames], dtype=torch.long,
                                device=data.frame_ids.device)
        if any(len(data.edge_centers[index]) == 0 for index in selected.cpu().tolist()):
            raise ValueError("指定训练帧缺少可用的 response>0.15 边缘中心")
        original_splits = {name: frames.copy() for name, frames in data.metadata["splits"].items()}
        data.splits["training"] = selected
        for name in ("validation", "test"):
            data.splits[name] = data.splits[name][~torch.isin(data.splits[name], selected)]
        data.comparison = selected.cpu().tolist()
        data.metadata["original_splits"] = original_splits
        data.metadata["splits"] = {
            name: data.frame_ids[frames].cpu().tolist() for name, frames in data.splits.items()
        }
        data.metadata["comparison_indices"] = args.train_frames
        data.metadata["comparison_roi_xywh"] = [350, 190, 250, 270]
        data.metadata["train_frames_only"] = args.train_frames
    elif args.fit_frames is not None:
        frame_lookup = {frame: index for index, frame in enumerate(data.frame_ids.cpu().tolist())}
        missing = sorted(set(args.fit_frames) - frame_lookup.keys())
        if missing:
            raise ValueError(f"拟合帧不存在: {missing}")
        selected_ids = sorted(args.fit_frames)
        selected = torch.tensor([frame_lookup[frame] for frame in selected_ids],
                                dtype=torch.long, device=data.frame_ids.device)
        if any(index in data.comparison for index in selected.cpu().tolist()):
            raise ValueError("拟合帧包含固定验证比较帧，无法保持原比较清单")
        original_splits = data.metadata["splits"]
        data.metadata["fit_original_splits"] = {
            frame: next(name for name, frames in original_splits.items() if frame in frames)
            for frame in selected_ids
        }
        data.splits["training"] = selected
        for name in ("validation", "test"):
            data.splits[name] = data.splits[name][
                ~torch.isin(data.frame_ids[data.splits[name]], data.frame_ids[selected])
            ]
        data.metadata["splits"] = {
            name: data.frame_ids[indices].cpu().tolist() for name, indices in data.splits.items()
        }
        data.metadata["fit_frames_only"] = selected_ids
    write_json(args.output/"comparison/run_config/data_manifest.json", data.metadata)
    if args.align_poses:
        align_poses(data, args)
        return
    sagittal = (SagittalIntersection(data, args.sagittal_reference, args.sagittal_calibration,
                                    ssim_weight=args.sagittal_ssim_weight) if sweep_experiment else None)
    if args.smoke and sweep_experiment:
        from .sweep_geometry import smoke_check
        geometry = smoke_check(device)
    else:
        geometry = geometry_check(device) if args.smoke else None
    results = [train_variant(data, args.output/name, name, args, smoke=args.smoke, sagittal=sagittal)
               for name in args.variants]
    if args.smoke:
        # 所有里程碑在后续更新完成后仍需独立重载；不以这个检查判断重建质量。
        for result in results:
            retained = sorted({(args.steps+1)//2, (3*args.steps+3)//4, args.stop_step or args.steps})
            paths = []
            for step in retained:
                if step > (args.stop_step or args.steps):
                    continue
                path = Path(result["checkpoint"]).with_name(f"step_{step:06d}.pt")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert saved["completed_steps"] == step and saved["model"] and saved["optimizer_states"]
                if saved.get("sweep_metadata"):
                    torch.testing.assert_close(load_sweep(saved).matrices(), saved["corrected_poses"].cpu())
                paths.append(str(path))
            result["retained_checkpoints"] = paths
        if sagittal is not None:
            compare_sweeps(data, sagittal, args.output, args.variants)
        write_json(args.output/"comparison/metrics/smoke_result.json", dict(
            status="PASS; execution only", geometry=geometry, variants=results,
            response_only=response_only, gray_independence_checked=response_only,
            quality="not evaluated"))
    else:
        comparison = compare_responses if response_only else compare
        comparison(data, args.output, args.variants)
        if sagittal is not None:
            compare_sweeps(data, sagittal, args.output, args.variants)


if __name__ == "__main__":
    main()
