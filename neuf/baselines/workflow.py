"""用固定版本官方网络逐序列重建；统一预算、留存及独立加载。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import trange

from .data import ROOT, cases, load_case, local_grid, write_json

VERSIONS = {"ImplicitVol": "996f6ea5a4c818b150333890c2db9feb6346e2f6",
            "NeSVoR": "730ddaa3711a2304386de34193ea4b957892fe7b"}


def upstream(name, filename):
    """仅加载用户批准下载的固定版本文件。"""
    module_name = "baseline_" + name + "_" + filename
    if module_name not in sys.modules:
        path = ROOT / "third_party" / name / (filename + ".py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def build_model(name, config):
    device = torch.device("cuda")
    if name == "ImplicitVol":
        official = upstream(name, "model")
        initial = np.asarray(config["initial_matrices"])
        store = {}
        for i, matrix in enumerate(initial):
            angles = Rotation.from_matrix(matrix[:, :3]).as_euler("xyz")
            store[i] = dict(rot_pred=angles, rot_ground=angles,
                            trans_pred=matrix[:, 3], trans_ground=matrix[:, 3])
        model = torch.nn.ModuleDict({"field": official.OfficialNerf_siren(63, 128),
                                     "pose": official.LearnPose(store)})
    else:
        from nesvor.cli.parsers import build_parser_training
        from nesvor.inr.models import NeSVoR
        from nesvor.transform import RigidTransform
        args = build_parser_training().parse_args([])
        args.device, args.dtype = device, torch.float32
        # 小批量 PSF 积分保留官方成像与损失；不启用 MRI 专用预处理。
        args.n_samples = config["psf_samples"]
        args.single_precision = True
        args.coarsest_resolution = config["coarsest_resolution"]
        args.finest_resolution = config["finest_resolution"]
        matrix = torch.tensor(config["initial_matrices"], dtype=torch.float32, device=device)
        resolution = torch.tensor(config["resolution"], device=device).expand(len(matrix), -1)
        model = NeSVoR(RigidTransform(matrix, trans_first=False), resolution,
                       config["intensity_mean"], torch.tensor(config["bbox_normalized"], device=device),
                       config["scale"], args)
    return model.to(device)


def query(model, name, config, xyz):
    normalized = (xyz - xyz.new_tensor(config["center"])) / config["scale"]
    if name == "ImplicitVol":
        return model["field"](upstream(name, "model").encode_position(normalized, 10, True)).squeeze(-1)
    return model.inr(normalized)


def load_model(path):
    """只依赖 checkpoint 与固定官方源码，可独立查询世界坐标。"""
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


def configure(data, name, steps, seed):
    bbox = data["bbox"]
    center = bbox.mean(0)
    scale = float(np.max(bbox[1] - bbox[0]) / 2) if name == "ImplicitVol" else 30.0
    matrices = np.concatenate((data["rotations"][data["train"]],
                               ((data["translations"][data["train"]] - center) / scale)[..., None]), axis=2)
    meta = data["metadata"]
    return dict(model_name=name, upstream_commit=VERSIONS[name], planned_steps=steps, seed=seed,
                center=center.tolist(), scale=scale, initial_matrices=matrices.tolist(),
                bbox_normalized=((bbox - center) / scale).tolist(),
                resolution=(np.array([meta["dx"], meta["dy"], meta["assumed_thickness"]]) / scale).tolist(),
                intensity_mean=float(data["images"][data["train"]].mean() / 255),
                psf_samples=32, coarsest_resolution=16.0, finest_resolution=0.5,
                patch_size=64, batch_size=1024, input=meta,
                evaluation="固定留出帧原始位姿；输入观测不是三维真值",
                training="逐病例从头优化；ImplicitVol 原网络/SSIM，NeSVoR 原网络/成像与损失")


def save_checkpoint(path, step, model, name, config, optimizers, schedulers, rng):
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已保留 checkpoint：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema="neuf_official_baselines_v1", model_name=name, completed_steps=step,
                   planned_steps=config["planned_steps"], config=config,
                   model_state_dict=model.state_dict(),
                   optimizer_states=[o.state_dict() for o in optimizers],
                   scheduler_states=[s.state_dict() for s in schedulers],
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                            sampler=rng.bit_generator.state, torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all()))
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"Retained checkpoint: {path} (step={step})", flush=True)


def sample_batch(data, rng, patch=False):
    frame_id = int(rng.integers(len(data["train"])))
    frame = int(data["train"][frame_id])
    h, w = data["images"].shape[1:]
    if patch:
        ys, xs = data["mask_indices"]
        pick = int(rng.integers(len(xs)))
        top, left = np.clip([ys[pick] - 32, xs[pick] - 32], [0, 0], [h - 64, w - 64])
        rows, cols = np.mgrid[top:top + 64, left:left + 64]
        ids = np.full(rows.size, frame_id)
        values = data["images"][frame, rows, cols].reshape(1, 1, 64, 64)
    else:
        ids = rng.integers(len(data["train"]), size=1024)
        ys, xs = data["mask_indices"]
        selected = rng.integers(len(xs), size=len(ids))
        rows, cols = ys[selected], xs[selected]
        values = data["images"][data["train"][ids], rows, cols]
    xyz = local_grid(data, rows, cols).reshape(-1, 3)
    return frame_id, torch.as_tensor(xyz, device="cuda"), torch.as_tensor(values, dtype=torch.float32, device="cuda") / 255, torch.as_tensor(ids, device="cuda")


def train_case(args, case_id, name):
    from .evaluation import evaluate, export_volume
    data = load_case(case_id, args.pixel_geometry)
    data["mask_indices"] = np.nonzero(data["mask"])
    steps = 20 if args.mode == "smoke" else args.steps
    output = ROOT / "logs" / args.run_id / data["metadata"]["cohort"] / ("index_" + case_id) / (name + ("_smoke" if args.mode == "smoke" else ""))
    if (output / "run_config/config.json").exists() and not args.resume:
        raise FileExistsError(f"已有实验，不能覆盖：{output}")
    for part in ("run_config", "metrics", "checkpoints"):
        (output / part).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    config = configure(data, name, steps, args.seed)
    config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    config["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*Path(__file__).parent.glob("*.py"), ROOT / "qsub/neuf/baselines.sh"]}
    model = build_model(name, config)
    if name == "ImplicitVol":
        optimizers = [torch.optim.Adam(model[k].parameters(), lr=0.001) for k in ("field", "pose")]
        schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[0], list(range(10, 10001, 10)), gamma=0.9954),
                      torch.optim.lr_scheduler.MultiStepLR(optimizers[1], list(range(100, 10001, 100)), gamma=0.9)]
        ssim = upstream(name, "ssim").SSIM(window_size=5)
    else:
        from nesvor.inr.models import USE_TORCH
        config["backend"] = "official_pytorch" if USE_TORCH else "tinycudann"
        groups = [{"params": [p for k, p in model.named_parameters() if p.numel() and "_net" not in k]},
                  {"params": [p for k, p in model.named_parameters() if p.numel() and "_net" in k], "weight_decay": 1e-2}]
        optimizers = [torch.optim.AdamW(groups, lr=0.005, betas=(0.9, 0.99), eps=1e-15)]
        schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[0], [int(steps * m) for m in (0.5, 0.75, 0.9)], gamma=0.33)]
    completed, status, retained = 0, "running", []
    previous_wall = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        previous = checkpoint["config"]
        if previous["input"] != config["input"] or previous["planned_steps"] != steps or checkpoint["model_name"] != name:
            raise ValueError("续训数据、模型或原总预算不匹配")
        config = previous
        model.load_state_dict(checkpoint["model_state_dict"])
        for optimizer, state in zip(optimizers, checkpoint["optimizer_states"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, checkpoint["scheduler_states"]):
            scheduler.load_state_dict(state)
        state = checkpoint["rng"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        rng.bit_generator.state = state["sampler"]
        torch.set_rng_state(state["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        completed = checkpoint["completed_steps"]
        retained = [str(p) for p in sorted((output / "checkpoints").glob("step_*.pt"))]
        previous_summary = output / "metrics/training_summary.json"
        if previous_summary.exists():
            previous_wall = json.loads(previous_summary.read_text()).get("cumulative_wall_sec", 0)
    write_json(output / "run_config/config.json", config)
    write_json(output / "run_config/comparison_manifest.json", data["metadata"])
    initial_queries = torch.tensor(data["bbox"], dtype=torch.float32, device="cuda")
    start, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    milestones = {(steps + 1) // 2, (3 * steps + 3) // 4, steps}
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    timing_path = output / "metrics/timing.csv"
    total_step_time, session_steps = 0.0, 0
    try:
        evaluate(model, name, config, data, output, completed, smoke=args.mode == "smoke")
        with timing_path.open("a" if args.resume else "w", newline="") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["epoch", "global_step", "step_time_sec", "elapsed_sec", "loss", "session_start"])
            progress = trange(completed + 1, steps + 1, desc=f"{case_id}/{name}", mininterval=10, file=sys.stdout)
            for step in progress:
                model.train()
                tick = time.perf_counter()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                frame, xyz, target, ids = sample_batch(data, rng, name == "ImplicitVol")
                if name == "ImplicitVol":
                    rotation, translation = model["pose"](frame)
                    world = xyz / config["scale"] @ rotation.T + translation
                    prediction = model["field"](upstream(name, "model").encode_position(world, 10, True))
                    loss = -ssim(prediction.reshape(1, 1, 64, 64), target)
                else:
                    losses = model(xyz=xyz / config["scale"], v=target, slice_idx=ids)
                    weights = {"MSE": 1, "logVar": 1, "transReg": 0.1, "biasReg": 100, "imageReg": 1, "deformReg": 0.1}
                    loss = sum(weights.get(key, 0) * value for key, value in losses.items())
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f"step={step} 出现非有限损失")
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if name == "NeSVoR" or step % len(data["train"]) == 0:
                    for scheduler in schedulers:
                        scheduler.step()
                torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                completed, session_steps = step, session_steps + 1
                total_step_time += duration
                value = float(loss.detach())
                elapsed = time.perf_counter() - start
                writer.writerow([step // len(data["train"]), step, duration, elapsed, value, started])
                progress.set_postfix(epoch=f"{step // len(data['train'])}/{steps // len(data['train'])}", loss=f"{value:.5f}", step_sec=f"{duration:.3f}", refresh=False)
                if step % 100 == 0 or step in milestones:
                    stream.flush()
                    write_json(output / "metrics/progress.json", dict(step=step, total=steps, loss=value, elapsed_sec=elapsed))
                if step in milestones or stop_requested:
                    path = output / "checkpoints" / f"step_{step:06d}.pt"
                    model.eval()
                    with torch.no_grad():
                        reference = query(model, name, config, initial_queries).clone()
                    save_checkpoint(path, step, model, name, config, optimizers, schedulers, rng)
                    retained.append(str(path))
                    # 新建网络独立加载，不能用仍在内存的训练模型冒充重载检查。
                    loaded, checkpoint = load_model(path)
                    with torch.no_grad():
                        torch.testing.assert_close(query(loaded, name, config, initial_queries), reference)
                    del loaded, checkpoint
                    # 重建网络会消耗随机数；恢复保存点状态，让验证不改变后续 PSF 采样。
                    saved_rng = torch.load(path, map_location="cpu", weights_only=False)["rng"]
                    torch.set_rng_state(saved_rng["torch"])
                    torch.cuda.set_rng_state_all(saved_rng["cuda"])
                    evaluate(model, name, config, data, output, step, smoke=args.mode == "smoke")
                    if stop_requested:
                        status = "stopped"
                        break
            else:
                status = "complete"
        if status == "complete" and args.mode != "smoke":
            export_volume(model, name, config, data, output)
    finally:
        if status == "running":
            status = "failed"
        summary = dict(status=status, configured_epochs=steps / len(data["train"]), completed_epochs=completed / len(data["train"]),
                       configured_total_steps=steps, completed_total_steps=completed,
                       start_timestamp=started, end_timestamp=datetime.now(timezone.utc).isoformat(),
                       session_wall_sec=time.perf_counter() - start, session_completed_steps=session_steps,
                       cumulative_wall_sec=previous_wall + time.perf_counter() - start,
                       mean_step_sec=total_step_time / max(session_steps, 1), checkpoints=retained,
                       final_checkpoint=retained[-1] if retained else None,
                       result_dir=str(output), seed=args.seed, unit=data["metadata"]["unit"])
        write_json(output / "metrics/training_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if status != "complete":
        raise RuntimeError(f"训练未完成：{status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train", "report"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=["all", "ImplicitVol", "NeSVoR"], default="all")
    parser.add_argument("--case", default="us_recal_original_cropped")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pixel-geometry", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "report":
        from .evaluation import report
        report(args.run_id)
        return
    for name, commit in VERSIONS.items():
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party" / name, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"官方源码版本不匹配：{name}")
    if not torch.cuda.is_available():
        raise RuntimeError("训练必须使用调度器分配的 CUDA GPU")
    aggregate = ROOT / "logs" / args.run_id / "baselines/index_all/comparison/run_config"
    write_json(aggregate / "input_manifest.json", dict(
        cases=cases(), excluded=[dict(case="Patient106_J0_0", reason="扫描数组为空")],
        aliases={"data_recal_Patient0_J35_2_d_0.5_matlab": "data_recal_Patient0_J35_2_d_0.5"},
        pixel_geometry_authorized=args.pixel_geometry, seed=args.seed))
    records = cases() if args.case == "all" else [next(c for c in cases() if c["case"] == args.case)]
    for record in records:
        for name in VERSIONS if args.model == "all" else [args.model]:
            train_case(args, record["case"], name)


if __name__ == "__main__":
    main()
