"""Known-volume acquisition and matched reconstruction controls (mm, native pixels)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation, Slerp

from .data import write_json
from .trajectory_reconstruction import Inputs, TrajectoryData, ReferenceGuidance


ARMS = ("oracle", "noisy_fixed", "noisy_6dof")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Volume:
    def __init__(self, root, name):
        self.root = Path(root)
        self.grid = json.loads((self.root / "grid.json").read_text())
        self.shape = tuple(self.grid["shape_zyx"])
        self.origin = np.array(self.grid["origin_xyz_mm"])
        self.spacing = self.grid["spacing_mm"]
        dtype = np.uint8 if name == "support" else np.float32
        self.values = np.memmap(self.root / f"{name}.raw", mode="r", dtype=dtype, shape=self.shape)

    def sample(self, xyz, values=None):
        xyz = np.asarray(xyz)
        coords = ((xyz.reshape(-1, 3) - self.origin) / self.spacing)[:, ::-1].T
        result = ndi.map_coordinates(self.values if values is None else values, coords,
                                     order=1, mode="constant", cval=0, prefilter=False)
        return result.reshape(xyz.shape[:-1])


def pose_metrics(estimated, truth, probes):
    est, true = np.asarray(estimated, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    ep = probes @ est[:, :3, :3].transpose(0, 2, 1) + est[:, None, :3, 3]
    tp = probes @ true[:, :3, :3].transpose(0, 2, 1) + true[:, None, :3, 3]
    tre = np.linalg.norm(ep - tp, axis=-1)
    angles = np.degrees(Rotation.from_matrix(est[:, :3, :3] @ true[:, :3, :3].transpose(0, 2, 1)).magnitude())
    origins = np.linalg.norm(est[:, :3, 3] - true[:, :3, 3], axis=-1)
    return dict(tre_mean_mm=float(tre[1:].mean()), tre_rms_mm=float(np.sqrt((tre[1:]**2).mean())),
                tre_p95_mm=float(np.percentile(tre[1:], 95)), rotation_mean_deg=float(angles[1:].mean()),
                origin_mean_mm=float(origins[1:].mean()), frame_tre_mm=tre.mean(1).tolist(),
                frame_rotation_deg=angles.tolist(), scope="frames 1..241; frame0 is fixed exact anchor")


def prepare(args):
    from .model import PoseRefiner, se3_exp
    from .coronal_features import coronal_projection
    root = args.dataset
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    original = TrajectoryData(Inputs("cpu"), "CNN3D")
    source = original.initial.numpy().astype(np.float64)
    n = len(source)
    # Smooth the original sweep, then resample at new times and change the global pose.
    quat = Rotation.from_matrix(source[:, :3, :3]).as_quat()
    for i in range(1, n):
        if np.dot(quat[i], quat[i-1]) < 0:
            quat[i] *= -1
    quat = ndi.gaussian_filter1d(quat, 4., axis=0, mode="nearest")
    times = np.linspace(2.5, n - 3.5, n)
    truth = np.tile(np.eye(4), (n, 1, 1))
    truth[:, :3, :3] = Slerp(np.arange(n), Rotation.from_quat(quat))(times).as_matrix()
    smooth_origin = ndi.gaussian_filter1d(source[:, :3, 3], 4., axis=0, mode="nearest")
    truth[:, :3, 3] = np.column_stack([np.interp(times, np.arange(n), smooth_origin[:, j]) for j in range(3)])
    support = Volume(args.volumes, "support")
    bounds = np.stack((support.origin, support.origin + (np.array(support.shape[::-1])-1)*support.spacing))
    center = bounds.mean(0)
    global_rotation = Rotation.from_euler("xyz", [.4, -.3, .6], degrees=True).as_matrix()
    truth[:, :3, :3] = global_rotation @ truth[:, :3, :3]
    truth[:, :3, 3] = (truth[:, :3, 3]-center) @ global_rotation.T + center + [.2, -.15, .25]
    truth = truth.astype(np.float32)
    rng = np.random.default_rng(20261117)
    correction = np.concatenate((np.clip(rng.normal(0, .25, (n, 3)), -.75, .75),
                                 np.deg2rad(np.clip(rng.normal(0, .4, (n, 3)), -1.2, 1.2))), axis=1).astype(np.float32)
    correction[0] = 0
    inverse = se3_exp(-torch.from_numpy(correction)).numpy()
    noisy = truth.copy()
    noisy[:, :3, :3] = inverse[:, :3, :3] @ truth[:, :3, :3]
    noisy[:, :3, 3] = (inverse[:, :3, :3] @ (truth[:, :3, 3]-center)[..., None])[..., 0] + center + inverse[:, :3, 3]
    refiner = PoseRefiner(torch.from_numpy(noisy), torch.arange(n), center, translation_mm=1., rotation_deg=2.)
    with torch.no_grad():
        refiner.raw.copy_(torch.atanh(torch.from_numpy(correction) / refiner.limits))
    residual = float((refiner.matrices()-torch.from_numpy(truth)).abs().max())
    assert residual < 2e-5, residual
    local = original.local.numpy()
    fov = original.mask.numpy().mean(0) > .9
    # Exclude the artificial zero exterior before interpolation; never use image intensity to select evaluation points.
    interior = ndi.binary_erosion(support.values.astype(bool), iterations=2).astype(np.float32)
    masks = np.empty((n, *fov.shape), dtype=bool)
    reference_xyz = original.guidance.xyz_grid.numpy()
    reference_mask = support.sample(reference_xyz, interior) > .999
    with h5py.File(root / "acquisition.h5", "x") as out:
        out.create_dataset("initial", data=noisy)
        out.create_dataset("local", data=local)
        out.create_dataset("bounds", data=bounds.astype(np.float32))
        out.create_dataset("reference_mask", data=reference_mask, compression="gzip", compression_opts=1)
        for name in ("inr", "knn"):
            vol = Volume(args.volumes, name)
            images = out.create_dataset(f"{name}/images", shape=(n, *fov.shape), dtype="f4",
                                        chunks=(1, *fov.shape), compression="gzip", compression_opts=1)
            for i, matrix in enumerate(truth):
                xyz = local @ matrix[:3, :3].T + matrix[:3, 3]
                if name == "inr":
                    masks[i] = fov & (support.sample(xyz, interior) > .999)
                images[i] = np.where(masks[i], vol.sample(xyz), 0)
                if i % 60 == 0:
                    print("sampled", name, i, flush=True)
            out.create_dataset(f"{name}/reference", data=np.where(reference_mask, vol.sample(reference_xyz), 0))
        out.create_dataset("mask", data=masks, chunks=(1, *fov.shape), compression="gzip", compression_opts=1)
    probe_indices = np.flatnonzero(ndi.binary_erosion(fov, iterations=10))[::997]
    probes = local.reshape(-1, 3)[probe_indices]
    # Fixed random 3D locations away from all acquisition planes and from the supervised SAG.
    candidates = np.flatnonzero(interior.ravel())
    chosen = rng.choice(candidates, size=min(350000, len(candidates)), replace=False)
    xyz = np.column_stack(np.unravel_index(chosen, support.shape))[:, ::-1] * support.spacing + support.origin
    xyz = xyz[np.abs(xyz[:, 2]) > .3].astype(np.float32)
    heldout = []
    ttruth, tmask = torch.from_numpy(truth), torch.from_numpy(masks).float()
    with torch.no_grad():
        for batch in torch.from_numpy(xyz).split(2048):
            projected, weights, valid = coronal_projection(batch, ttruth, original.local, tmask)
            distance = (projected - batch[None]).norm(dim=-1)
            minimum = distance.masked_fill(weights <= 0, float("inf")).amin(0)
            heldout.append(batch[valid & (minimum >= .06) & (minimum <= 1.5)].numpy())
    xyz = np.concatenate(heldout)[:100000]
    assert len(xyz) >= 20000, len(xyz)
    np.savez_compressed(root / "truth.npz", poses=truth, correction=correction, probes=probes,
                        holdout_xyz=xyz, inr=Volume(args.volumes, "inr").sample(xyz), knn=Volume(args.volumes, "knn").sample(xyz))
    manifest = dict(schema="synthetic_volume_v1", volumes=str(args.volumes), volume_grid=support.grid,
                    source_raw_sha256={name: digest(args.volumes / f"{name}.raw") for name in ("inr", "knn", "support")},
                    acquisition_sha256=digest(root / "acquisition.h5"), truth_sha256=digest(root / "truth.npz"),
                    new_trajectory="quaternion/translation Gaussian sigma4; 242 new times 2.5..238.5; global xyz Euler[.4,-.3,.6]deg and translation[.2,-.15,.25]mm about volume center",
                    source_trajectory=original.metadata["pose_source"], seed=20261117,
                    noise="inverse SE3 about shared volume center: iid t sigma.25mm clip.75mm; r sigma.4deg clip1.2deg; frame0 exact",
                    exact_correction_representable_max_matrix_error=residual,
                    image_model="trilinear samples of float32 B-mode volume; no added intensity noise or PSF",
                    mask_policy="90% common native acoustic FOV intersect support eroded2 voxels and trilinear support>.999",
                    shape_hw=list(fov.shape), frames=n, source_spacing_hw_mm=list(original.spacing),
                    reference_spacing_mm=float(original.guidance.pose.pitch), reference_geometry="native_sag_pixel_pitch_v1",
                    field_bounds_mm=bounds.tolist(), holdout_count=len(xyz),
                    holdout_scope="fixed volume voxels; >=.06mm from nearest supported acquisition plane, <=1.5mm; abs(z)>.3mm; no target intensity/learnt pose selection",
                    initial_pose_error=pose_metrics(noisy, truth, probes),
                    valid_pixels_per_frame=masks.sum((1, 2)).tolist(),
                    interpretation="two same-patient reconstructed volumes are synthetic truth, not independent clinical 3D truth")
    write_json(root / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("holdout_count", "exact_correction_representable_max_matrix_error")}), flush=True)


class SyntheticData:
    patches = TrajectoryData.patches

    def __init__(self, root, teacher, arm, device="cuda"):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        with h5py.File(root / "acquisition.h5", "r") as h:
            masks = h["mask"][:]
            self.images = torch.tensor(h[f"{teacher}/images"][:], device=device)
            self.mask = torch.tensor(masks, device=device, dtype=torch.float32)
            self.local = torch.tensor(h["local"][:], device=device)
            self.bounds = torch.tensor(h["bounds"][:], device=device)
            initial = h["initial"][:]
            inputs = SimpleNamespace(reference=torch.tensor(h[f"{teacher}/reference"][:], device=device),
                                     reference_mask=torch.tensor(h["reference_mask"][:], device=device))
        if arm == "oracle":
            with np.load(root / "truth.npz") as truth:
                initial = truth["poses"]
        self.initial = torch.tensor(initial, device=device)
        self.height, self.width = self.images.shape[-2:]
        self.spacing = manifest["source_spacing_hw_mm"]
        self.valid_centers = [np.flatnonzero(ndi.binary_erosion(m, iterations=10)).astype(np.int32) for m in masks]
        assert all(len(c) for c in self.valid_centers)
        self.frame_ids = torch.arange(len(self.images), device=device)
        self.splits = dict(training=self.frame_ids, validation=self.frame_ids[:0], test=self.frame_ids[:0])
        self.comparison = [236, 166, 96, 26]
        self.guidance = ReferenceGuidance(inputs, manifest["reference_spacing_mm"])
        self.guidance.metadata["roi"] = "synthetic volume support; real SAG manual R1-R5 not used"
        self.metadata = dict(manifest, method=f"synthetic_{teacher}_{arm}", teacher=teacher, arm=arm,
                             acquisition_path=str(root / "acquisition.h5"),
                             frame_ids=list(range(len(self.images))),
                             splits=dict(training=list(range(len(self.images))), validation=[], test=[]),
                             normalization="float32 teacher intensities in [0,1], no rescaling",
                             reference_image="synthetic SAG z=0 from same teacher volume")


def train(args):
    from .cell_experiment import CoronalObservation
    from .workflow import train_coronal_variant
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 20:
        raise ValueError("steps must be >=20")
    torch.set_num_threads(4)
    data = SyntheticData(args.dataset, args.teacher, args.arm)
    trial = argparse.Namespace(output=args.output, mode="train", method=data.metadata["method"], device="cuda",
                               pose_mode="se3" if args.arm == "noisy_6dof" else "off", pose_supervision="observed",
                               steps=args.steps, seed=3407, encoding="hash", extra_checkpoints=[],
                               patches=4, patch_size=64, sagittal_patches=2, width=128, bands=10, lr=.0005, pose_lr=.002)
    train_coronal_variant(data, data.guidance, args.output, trial, conditioned=False,
                          observation=CoronalObservation(data.initial, "online" if args.arm == "noisy_6dof" else "fixed"))


def query(model, xyz, device):
    with torch.no_grad():
        return torch.cat([model(p.to(device))[..., 0].cpu() for p in torch.from_numpy(np.asarray(xyz, dtype=np.float32)).reshape(-1, 3).split(16384)]).numpy().reshape(xyz.shape[:-1])


def image_metrics(pred, truth, mask):
    from .losses import masked_ssim
    mask = mask.astype(bool)
    err = pred[mask].astype(np.float64)-truth[mask]
    mse = float(np.mean(err**2))
    tensor = lambda x: torch.from_numpy(np.asarray(x)).float()[None, None]
    score, count = masked_ssim(tensor(pred), tensor(truth), tensor(mask))
    gm = ndi.binary_erosion(mask, iterations=1)
    gp, gt = [np.hypot(ndi.sobel(x, 0), ndi.sobel(x, 1))[gm] for x in (pred, truth)]
    return dict(mse=mse, psnr_db=float(-10*np.log10(max(mse, 1e-20))), ssim=float(score),
                valid_windows=int(count), edge_ncc=float(np.corrcoef(gp, gt)[0, 1]))


def evaluate(args):
    from .model import load_field
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    volume = Volume(Path(manifest["volumes"]), args.teacher)
    support = Volume(Path(manifest["volumes"]), "support")
    interior = ndi.binary_erosion(support.values.astype(bool), iterations=2).astype(np.float32)
    with np.load(root / "truth.npz") as h:
        truth, probes, xyz, target = (h[k] for k in ("poses", "probes", "holdout_xyz", args.teacher))
    rows, saved_images = [], {}
    for checkpoint in sorted((args.output / "checkpoints").glob("step_*.pt")):
        model, payload = load_field(checkpoint, device)
        prediction = query(model, xyz, device)
        mse = float(np.mean((prediction.astype(np.float64)-target)**2))
        row = dict(teacher=args.teacher, arm=args.arm, step=payload["completed_steps"],
                   volume_holdout_mse=mse, volume_holdout_psnr_db=float(-10*np.log10(mse)),
                   volume_holdout_mae=float(np.mean(np.abs(prediction-target))), holdout_count=len(xyz),
                   **pose_metrics(payload["corrected_poses"].numpy(), truth, probes), slices={})
        # Fixed world planes selected before model fitting; z=2 is not supervised SAG z=0.
        for name, fixed_axis, position in (("sag_z2", 2, 2.), ("axial_y35_5", 1, 35.5), ("cor_x64", 0, 64.)):
            axes = [a for a in range(3) if a != fixed_axis]
            horizontal, vertical = axes
            shape = np.array(volume.shape[::-1])
            xx = np.arange(shape[horizontal])*volume.spacing+volume.origin[horizontal]
            yy = np.arange(shape[vertical])*volume.spacing+volume.origin[vertical]
            meshx, meshy = np.meshgrid(xx, yy)
            plane = np.empty((*meshx.shape, 3), dtype=np.float32)
            plane[..., horizontal], plane[..., vertical], plane[..., fixed_axis] = meshx, meshy, position
            mask = support.sample(plane, interior) > .999
            ref, pred = volume.sample(plane), query(model, plane, device)
            row["slices"][name] = image_metrics(pred, ref, mask)
            saved_images.update({f"{name}_prediction": pred, f"{name}_truth": ref, f"{name}_mask": mask})
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("frame_") and k != "slices"}), flush=True)
        del model, payload
    write_json(args.output / "metrics/synthetic_truth.json", dict(rows=rows, manifest=str(root / "manifest.json")))
    np.savez_compressed(args.output / "predictions/synthetic_truth.npz", **saved_images)


def compare(args):
    import matplotlib.pyplot as plt
    root = args.output
    records = {(t, a): json.loads((root / f"{t}_{a}/metrics/synthetic_truth.json").read_text())["rows"] for t in ("inr", "knn") for a in ARMS}
    final = [r[-1] for r in records.values()]
    write_json(root / "comparison.json", dict(rows=final, histories={f"{t}_{a}": v for (t, a), v in records.items()}))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for row, teacher in enumerate(("inr", "knn")):
        for col, arm in enumerate(ARMS, 1):
            images = np.load(root / f"{teacher}_{arm}/predictions/synthetic_truth.npz")
            mask = images["sag_z2_mask"]
            valid_y, valid_x = np.where(mask)
            crop = np.s_[valid_y.min():valid_y.max()+1, valid_x.min():valid_x.max()+1]
            if col == 1:
                axes[row, 0].imshow(np.where(mask, images["sag_z2_truth"], np.nan)[crop], cmap="gray", vmin=0, vmax=1)
                axes[row, 0].set_title(f"{teacher.upper()} synthetic truth: SAG z=2 mm")
            axes[row, col].imshow(np.where(mask, images["sag_z2_prediction"], np.nan)[crop], cmap="gray", vmin=0, vmax=1)
            metrics = records[teacher, arm][-1]
            axes[row, col].set_title(f"{arm}\nSSIM {metrics['slices']['sag_z2']['ssim']:.4f}; TRE {metrics['tre_mean_mm']:.3f} mm")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for col, teacher in enumerate(("inr", "knn")):
        for arm in ARMS:
            history = records[teacher, arm]
            metrics = history[-1]
            axes[0, col].plot(metrics["frame_tre_mm"], label=arm, alpha=.8)
            axes[1, col].plot([r["step"] for r in history], [r["volume_holdout_psnr_db"] for r in history], "o-", label=arm)
        axes[0, col].set(title=teacher.upper(), ylabel="True pixel TRE [mm]", xlabel="Acquisition frame")
        axes[1, col].set(ylabel="Held-out 3D PSNR [dB]", xlabel="INR updates")
        axes[0, col].legend()
        axes[1, col].legend()
    fig.savefig(root / "geometry_quality.png", dpi=150)
    plt.close(fig)


def visualize(args):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    root = args.dataset
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "truth.npz") as h:
        truth, probes, correction = (h[k] for k in ("poses", "probes", "correction"))
    with h5py.File(root / "acquisition.h5", "r") as h:
        noisy, local = h["initial"][:], h["local"][:]
        masks = np.stack([h["mask"][i] for i in (0, 48, 96, 144, 192, 241)])
        sample_images = {teacher: np.stack([h[f"{teacher}/images"][i] for i in (26, 126, 226)]) for teacher in ("inr", "knn")}
        references = {teacher: h[f"{teacher}/reference"][:] for teacher in ("inr", "knn")}
    with h5py.File(manifest["source_trajectory"]["path"], "r") as h:
        source = h["poses"][:]
    source[:, :3, :3] = source[:, :3, :3] @ np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    # A separate lightweight artifact makes these explicit matrices inspectable without training data.
    np.savez_compressed(output / "trajectories.npz", source=source, truth=truth, noisy=noisy,
                        correction=correction, local_probes=probes)
    import csv
    with (output / "trajectory_centers.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"]+[f"{name}_{axis}_mm" for name in ("source", "truth", "noisy") for axis in "xyz"])
        writer.writerows([[i]+[float(matrix[i, j, 3]) for matrix in (source, truth, noisy) for j in range(3)] for i in range(len(truth))])
    fig = plt.figure(figsize=(15, 7), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    colors = cm.viridis(np.linspace(0, 1, 6))
    all_points = []
    for i, mask, color in zip((0, 48, 96, 144, 192, 241), masks, colors):
        pts = local[::8, ::8][mask[::8, ::8]]
        hull = pts[ConvexHull(pts[:, :2]).vertices]
        world = hull @ truth[i, :3, :3].T + truth[i, :3, 3]
        all_points.append(world)
        ax.add_collection3d(Poly3DCollection([world], facecolors=[color], edgecolors=[color], alpha=.13, linewidths=1.2))
        p = world[np.argmax(world[:, 1])]
        ax.text(*p, f"#{i}", color=color, fontsize=9)
    ax.plot(*truth[:, :3, 3].T, color="#147bb8", linewidth=2.5, label="New true image centers")
    lo, hi = np.concatenate(all_points).min(0), np.concatenate(all_points).max(0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel="SAG right x [mm]", ylabel="SAG depth y [mm]", zlabel="SAG normal z [mm]",
           title="New acquisition: 242 planes\nSix acoustic footprints shown (convex outlines)")
    ax.set_box_aspect(hi-lo)
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc="upper left", fontsize=9)
    ax = fig.add_subplot(122, projection="3d")
    for matrices, color, label, width in ((source, "#777777", "Original Edge + 3DCNN", 1.2),
                                        (truth, "#147bb8", "New ground-truth path", 2.5),
                                        (noisy, "#e58a25", "Perturbed initial path", .8)):
        ax.plot(*matrices[:, :3, 3].T, color=color, label=label, linewidth=width, alpha=.9)
    for i in (0, 48, 96, 144, 192, 241):
        ax.scatter(*truth[i, :3, 3], s=18, color="#147bb8")
        ax.text(*truth[i, :3, 3], str(i), fontsize=8)
    ax.set_box_aspect((1.5, 1., .65))
    ax.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title="Image-center paths (unequal axis scales)\nCenters are not measured probe positions")
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=25, azim=-50)
    ax.legend(fontsize=9)
    fig.suptitle("Same acquisition geometry for both INR and KNN synthetic truth volumes", fontsize=15)
    fig.savefig(output / "sampling_trajectory.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, layout="constrained")
    for matrices, color, label, width in ((source, "#777777", "Original Edge + 3DCNN", 1.),
                                        (truth, "#147bb8", "New truth", 2.), (noisy, "#e58a25", "Perturbed initialization", .9)):
        rotation = np.degrees(Rotation.from_matrix(matrices[:, :3, :3] @ truth[0, :3, :3].T).as_rotvec())
        for j, axis in enumerate("xyz"):
            axes[0, j].plot(matrices[:, j, 3], color=color, label=label, linewidth=width)
            axes[1, j].plot(rotation[:, j], color=color, linewidth=width)
            axes[0, j].set(ylabel=f"Center {axis} [mm]")
            axes[1, j].set(ylabel=f"Relative rotation vector {axis} [deg]", xlabel="Acquisition frame")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("New 6DoF acquisition poses and injected tracking error; rotations relative to true frame 0")
    fig.savefig(output / "trajectory_components.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for row, teacher in enumerate(("inr", "knn")):
        panels = [references[teacher], *sample_images[teacher]]
        titles = ["Synthetic SAG z=0", "New COR #26", "New COR #126", "New COR #226"]
        for ax, panel, title in zip(axes[row], panels, titles):
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{teacher.upper()} truth: {title}", fontsize=10)
            ax.axis("off")
    fig.savefig(output / "sampled_frames.png", dpi=160)
    plt.close(fig)
    write_json(output / "trajectory_summary.json", dict(initial_error=pose_metrics(noisy, truth, probes),
                new_vs_source=pose_metrics(truth, source, probes),
                truth_center_min_xyz_mm=truth[:, :3, 3].min(0).tolist(),
                truth_center_max_xyz_mm=truth[:, :3, 3].max(0).tolist(),
                frames=len(truth), **{k: manifest[k] for k in ("new_trajectory", "noise", "exact_correction_representable_max_matrix_error")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate", "compare", "visualize"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--volumes", type=Path)
    parser.add_argument("--teacher", choices=("inr", "knn"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
