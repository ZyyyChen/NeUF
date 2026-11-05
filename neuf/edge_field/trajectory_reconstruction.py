"""在现有完整轨迹上比较同一个坐标隐式场与四种附加位姿修正。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn

from .data import write_json
from .workflow import train_coronal_variant


METHODS = ("FourP", "SevenP", "Edge", "CNN", "EdgeDT", "Annotation", "AnnotationDT")
MODES = ("off", "translation", "se3", "sweep")
WORKSPACE = Path(__file__).resolve().parents[3]
AUTO_ROOT = "logs/20261029_train01/cerebral/index_patient0/NetworkInitializationComparison"
ANNOTATION_ROOT = "logs/20261104_train01/cerebral/index_patient0/AnnotationSevenDTBaseline"
GEOMETRY_ROOT = "logs/20261026_train08/cerebral_patient0/index_x0472/EdgeRegistration"
ANNOTATION_PREP = "logs/20261030_train03/cerebral/index_patient0/AnnotationDTComparison/run_config/preparation.json"
AUTO_KEYS = dict(FourP="four_parameter", SevenP="seven_parameter", Edge="edge",
                 CNN="edge_cnn2d", EdgeDT="edge_dualtrack")


def workspace_path(value):
    path = (WORKSPACE / value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"路径越出已批准工作区：{path}")
    return path


class Inputs:
    """七种轨迹共用同一原生图像、逐帧声学掩膜及随机采样候选。"""

    def __init__(self, device):
        self.hashes = {}
        manifest_path = workspace_path(f"{ANNOTATION_ROOT}/run_config/input_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        auto_manifest = json.loads(workspace_path(
            "logs/20261026_train23/cerebral/index_patient0/CNN2D6D/run_config/input_manifest.json").read_text())
        self.sources = {"annotation_manifest": self.identify(manifest_path)}
        for name in ("coronal", "sagittal", "metadata", "reference_mask"):
            path = workspace_path(manifest[name]["path"])
            record = self.identify(path)
            if record["sha256"] != manifest[name]["sha256"] or record["sha256"] != auto_manifest[name]["sha256"]:
                raise ValueError(f"自动与标注方法的 {name} 输入身份不一致")
            self.sources[name] = record
        with h5py.File(self.sources["coronal"]["path"], "r") as handle:
            frames = np.ascontiguousarray(handle["data_recal"][:].transpose(0, 2, 1))
        with h5py.File(self.sources["sagittal"]["path"], "r") as handle:
            reference = np.asarray(handle["data_sag"]).T.copy()
        if frames.shape != (242, 708, 944) or reference.shape != (708, 944):
            raise ValueError("当前实验要求 Patient0 的242张708×944原生图及同尺寸SAG")
        if frames.dtype != np.uint8 or reference.dtype != np.uint8:
            raise ValueError("原始灰度须为uint8，不能自行重新归一化")
        reference_mask = np.asarray(Image.open(self.sources["reference_mask"]["path"]).convert("L")) > 0
        if reference_mask.shape != reference.shape:
            raise ValueError("参考掩膜尺寸不匹配")
        self.roi_path = workspace_path("EdgeReg/configs/patient0_rois.json")
        roi = json.loads(self.roi_path.read_text())
        roi_image = workspace_path(roi["image_path"])
        if self.identify(roi_image)["sha256"] != roi["image_sha256"]:
            raise ValueError("固定R1–R5所对应的SAG图像哈希不匹配")
        if roi["image_shape_hw"] != list(reference.shape) or not np.array_equal(
                np.asarray(Image.open(roi_image).convert("L")), reference):
            raise ValueError("当前原生SAG与ROI参考像素不一致")
        if roi["roi_ids"] != ["R1", "R2", "R3", "R4", "R5"]:
            raise ValueError("固定ROI顺序改变")
        self.sources.update(roi_config=self.identify(self.roi_path), roi_reference=self.identify(roi_image))
        pitches = json.loads(Path(self.sources["metadata"]["path"]).read_text())["infos"]["px_size_cm"]
        self.spacing = float(pitches["height"]) * 10, float(pitches["width"]) * 10
        np.testing.assert_allclose(self.spacing, [.119928, .119928], atol=1e-12, rtol=0)
        masks = np.empty(frames.shape, dtype=bool)
        self.centers = []
        for index, frame in enumerate(frames):
            labels, count = ndimage.label(frame > 0)
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            if count == 0 or sizes.max() <= frame.size / 10:
                raise ValueError(f"第{index}帧没有可靠的声学支持域")
            masks[index] = ndimage.binary_fill_holes(labels == sizes.argmax())
            # 只需保存一维int32索引；同一帧和随机数在所有方法中选到同一个图块。
            centers = np.flatnonzero(ndimage.binary_erosion(masks[index], iterations=10)).astype(np.int32)
            if not len(centers):
                raise ValueError(f"第{index}帧没有完整SSIM窗口候选")
            self.centers.append(centers)
        self.mask_hash = hashlib.sha256(masks.tobytes()).hexdigest()
        self.images = torch.as_tensor(frames, device=device, dtype=torch.float32).div_(255)
        self.mask = torch.as_tensor(masks, device=device, dtype=torch.float32)
        self.reference = torch.as_tensor(reference, device=device, dtype=torch.float32).div_(255)
        self.reference_mask = torch.as_tensor(reference_mask, device=device)
        self.roi = roi
        self.geometry_path = workspace_path(f"{GEOMETRY_ROOT}/run_config/initialization.json")
        self.geometry = json.loads(self.geometry_path.read_text())
        self.annotation_path = workspace_path(ANNOTATION_PREP)
        self.annotation = json.loads(self.annotation_path.read_text())
        self.sources.update(automatic_geometry=self.identify(self.geometry_path),
                            annotation_geometry=self.identify(self.annotation_path))

    def identify(self, path):
        path = workspace_path(path)
        if path not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self.hashes[path] = digest.hexdigest()
        return dict(path=str(path), sha256=self.hashes[path])


class FixedReference(nn.Module):
    """当前方法已冻结的SAG像素标定；matrix表示像素原点的毫米坐标框架。"""

    def __init__(self, pitch, device):
        super().__init__()
        self.register_buffer("initial", torch.eye(4, device=device))
        self.register_buffer("pitch", torch.tensor(pitch, device=device))
        self.register_buffer("native_shape", torch.tensor([708, 944], device=device))

    def matrix(self):
        return self.initial

    def world(self, xy):
        local = torch.cat((xy * self.pitch, torch.zeros_like(xy[..., :1])), -1)
        return local @ self.initial[:3, :3].T + self.initial[:3, 3]

    def grid(self):
        height, width = self.native_shape.tolist()
        row, column = torch.meshgrid(torch.arange(height, device=self.pitch.device),
                                     torch.arange(width, device=self.pitch.device), indexing="ij")
        return self.world(torch.stack((column, row), -1).float())


def load_fixed_reference(checkpoint, device="cpu"):
    """仅从里程碑恢复参考网格，不读取原始MAT、标定文件或旧SAG适配类。"""
    if checkpoint["data_metadata"].get("reference_geometry") != "native_sag_pixel_pitch_v1":
        raise ValueError("checkpoint不是当前原生SAG固定标定接口")
    state = checkpoint["sagittal_pose"]
    pose = FixedReference(float(state["pitch"]), device)
    pose.load_state_dict(state, strict=True)
    return pose


class ReferenceGuidance:
    def __init__(self, inputs, pitch):
        self.reference = inputs.reference
        self.roi = self.reference_mask = inputs.reference_mask
        device = self.reference.device
        self.pose = FixedReference(pitch, device)
        self.xyz_grid = self.pose.grid()
        self.uniform_centers = torch.nonzero(self.roi).cpu()
        self.metadata = dict(reference_spacing_mm=pitch, reference_plane="SAG world z=0; x right, y down",
                             roi="full original SAG acoustic mask; fixed R1-R5 for reporting")


class TrajectoryData:
    def __init__(self, inputs, method):
        self.images, self.mask = inputs.images, inputs.mask
        self.height, self.width = self.images.shape[-2:]
        self.spacing, self.valid_centers = inputs.spacing, inputs.centers
        device = self.images.device
        self.frame_ids = torch.arange(len(self.images), device=device)
        self.splits = dict(training=self.frame_ids, validation=self.frame_ids[:0], test=self.frame_ids[:0])
        self.comparison = [236, 226, 216, 206]
        automatic = method in AUTO_KEYS
        pitch = .133849221834 if automatic else float(inputs.annotation["reference_spacing_mm"])
        column = 472 if automatic else 477
        if automatic:
            source = workspace_path(f"{AUTO_ROOT}/predictions/initialization_comparison.h5")
            key = f"{AUTO_KEYS[method]}/poses"
        else:
            step = 0 if method == "Annotation" else 200
            source = workspace_path(f"{ANNOTATION_ROOT}/predictions/step_{step:06d}.h5")
            key = "reference_pose_matrices"
        with h5py.File(source, "r") as handle:
            original = np.asarray(handle[key], dtype=np.float64)
            if automatic:
                np.testing.assert_allclose(handle["reference"][:], inputs.reference.cpu().numpy(), atol=1e-7, rtol=0)
        if original.shape != (242, 4, 4) or not np.isfinite(original).all():
            raise ValueError(f"{method} 缺少242个有效的完整刚体位姿")
        np.testing.assert_allclose(original[:, 3], np.tile([0, 0, 0, 1], (242, 1)), atol=1e-7, rtol=0)
        np.testing.assert_allclose(original[:, :3, :3].transpose(0, 2, 1) @ original[:, :3, :3],
                                   np.broadcast_to(np.eye(3), (242, 3, 3)), atol=3e-5, rtol=0)
        np.testing.assert_allclose(np.linalg.det(original[:, :3, :3]), 1, atol=3e-5, rtol=0)
        # 已归档完整矩阵以图像中心为原点，x477的偏移已包含，不能再次补偿。
        # 交换轴向/横向，同时反转局部法向，保持右手正交基及所有像素世界位置。
        basis = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        converted = original.copy()
        converted[:, :3, :3] = original[:, :3, :3] @ basis
        probes = np.array([[0., 0.], [472., 100.], [477., 353.5], [943., 707.]])
        old_local = np.column_stack(((probes[:, 0] - 472) * self.spacing[1],
                                    (probes[:, 1] - 353.5) * self.spacing[0], np.zeros(len(probes))))
        new_local = old_local @ basis
        old_world = old_local @ original[:, :3, :3].transpose(0, 2, 1) + original[:, None, :3, 3]
        new_world = new_local @ converted[:, :3, :3].transpose(0, 2, 1) + converted[:, None, :3, 3]
        np.testing.assert_allclose(old_world, new_world, atol=1e-11, rtol=0)
        self.initial = torch.as_tensor(converted, device=device, dtype=torch.float32)
        row, col = torch.meshgrid(torch.arange(self.height, device=device),
                                  torch.arange(self.width, device=device), indexing="ij")
        self.local = torch.stack(((row - 353.5) * self.spacing[0],
                                  (col - 472) * self.spacing[1], row * 0), -1).float()
        if method == "FourP":
            center_xy = np.tile(inputs.geometry["fan_apex"], (242, 1))
            center_source = "saved 4P detected fan_apex; mathematical rotation center"
        else:
            parameters = (inputs.geometry["parameters"] if automatic
                          else inputs.annotation["seven_parameters_reference"])
            center_xy = np.asarray(parameters[:2]) + np.arange(242)[:, None] * np.asarray(parameters[2:4])
            center_source = ("saved 7P cx/cz + frame*vx/vz" if method == "SevenP" else
                             "inherited automatic 7P centers; Edge depth origin unknown, not a measured probe center"
                             if automatic else "saved annotation 7P cx/cz + frame*vx/vz; not a measured probe center")
        self.sweep_centers = torch.as_tensor(np.column_stack((center_xy * pitch, np.zeros(242))),
                                             device=device, dtype=torch.float32)
        self.sweep_axis = torch.tensor([0., 0., 1.], device=device)
        self.guidance = ReferenceGuidance(inputs, pitch)
        corners = self.local[[0, 0, self.height-1, self.height-1], [0, self.width-1, 0, self.width-1]]
        world_corners = corners @ self.initial[:, :3, :3].transpose(1, 2) + self.initial[:, None, :3, 3]
        sagittal_corners = self.guidance.xyz_grid[[0, 0, 707, 707], [0, 943, 0, 943]]
        extent = torch.cat((world_corners.reshape(-1, 3), sagittal_corners))
        self.bounds = torch.stack((extent.amin(0) - 5, extent.amax(0) + 5))
        self.metadata = dict(method=method, pose_source=inputs.identify(source), h5_dataset=key,
            inputs=inputs.sources, shape_hw=[708, 944], frame_ids=list(range(242)),
            splits=dict(training=list(range(242)), validation=[], test=[]), comparison_indices=self.comparison,
            comparison_roi_xywh=inputs.roi["roi_xywh"], roi_config=str(inputs.roi_path),
            roi_config_sha256=inputs.identify(inputs.roi_path)["sha256"], roi_reference_verified=True,
            reference_spacing_mm=pitch, source_spacing_hw_mm=list(self.spacing), registration_column=column,
            reference_geometry="native_sag_pixel_pitch_v1",
            normalization="raw uint8 / 255; no per-method rescaling", mask_sha256=inputs.mask_hash,
            mask_policy="per-frame largest nonblack component with holes filled; shared across 28 trials",
            coordinate_convention="local=(axial centered353.5,lateral centered472,0); world=(SAG right,down,normal) mm",
            basis_conversion=basis.tolist(), basis_probe_max_error_mm=float(np.abs(old_world-new_world).max()),
            source_column_offset_already_applied=True, sweep_center_source=center_source,
            sweep_centers_mm=self.sweep_centers.cpu().tolist(), sweep_axis=[0., 0., 1.],
            field_bounds_mm=self.bounds.cpu().tolist(), field_bound_margin_mm=5.,
            interpretation="same-case fitted reconstruction; each method retains its frozen calibration; no independent 3D ground truth")

    def patches(self, batch_size, size, generator):
        frames_cpu = torch.randint(len(self.images), (batch_size,), generator=generator)
        centers = []
        for index in frames_cpu.tolist():
            valid = self.valid_centers[index]
            pick = int(torch.randint(len(valid), (1,), generator=generator))
            centers.append(divmod(int(valid[pick]), self.width))
        device = self.images.device
        frames = frames_cpu.to(device)
        centers = torch.tensor(centers, device=device)
        rr = (centers[:, 0] - size//2).clamp(0, self.height-size)[:, None, None] + torch.arange(size, device=device)[None, :, None]
        cc = (centers[:, 1] - size//2).clamp(0, self.width-size)[:, None, None] + torch.arange(size, device=device)[None, None, :]
        return dict(frames=frames, local=self.local[rr, cc].reshape(batch_size, -1, 3),
                    image=self.images[frames[:, None, None], rr, cc][:, None],
                    mask=self.mask[frames[:, None, None], rr, cc][:, None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "smoke", "train", "compare"), required=True)
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--pose-mode", choices=("all", *MODES), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--encoding", choices=("fourier", "hash"), default="hash")
    args = parser.parse_args()
    args.output = workspace_path(args.output)
    if not args.output.is_relative_to(WORKSPACE / "logs"):
        raise ValueError("实验输出必须位于工作区logs层级")
    if args.steps < 20:
        raise ValueError("至少20次INR更新以覆盖既定位置调度")
    methods = METHODS if args.method == "all" else (args.method,)
    modes = MODES if args.pose_mode == "all" else (args.pose_mode,)
    if args.mode == "compare":
        from .evaluation import compare_trajectory_trials
        compare_trajectory_trials(args.output, args.steps, methods, modes,
                                  workspace_path("EdgeReg/configs/patient0_rois.json"))
        return
    inputs = Inputs(args.device)
    if args.mode == "prepare":
        reports = {method: TrajectoryData(inputs, method).metadata for method in METHODS}
        directory = args.output / "comparison/run_config"
        write_json(directory / "input_audit.json", dict(status="passed", methods=reports,
                   source_images_identical=True, roi_reference_verified=True,
                   scope="input identity, SO3 matrices, handedness and unchanged pixel projections only; not quality validation"))
        write_json(directory / "comparison_manifest.json", dict(methods=METHODS, modes=MODES,
                   expected_trials=28, source_frames=list(range(242)), comparison_indices=[236, 226, 216, 206],
                   steps=args.steps, seed=args.seed, reference_shape_hw=[708, 944], intensity_range=[0, 1],
                   roi_config=str(inputs.roi_path), roi=inputs.roi, inputs=inputs.sources,
                   calibration_policy="retain each method's existing frozen SAG pitch; compare in common native pixels"))
        for filename in ("trajectory_reconstruction.py", "model.py", "workflow.py", "evaluation.py", "coronal_features.py", "losses.py"):
            (directory / "source").mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__).with_name(filename), directory / "source" / filename)
        print(f"全部七组输入核验通过：{directory / 'input_audit.json'}", flush=True)
        return
    if args.mode == "smoke" and args.method == "all":
        methods = ("AnnotationDT",)
    for method in methods:
        data = TrajectoryData(inputs, method)
        for mode in modes:
            output = args.output / f"{method}_{mode}"
            if output.exists():
                raise FileExistsError(f"拒绝覆盖已有训练：{output}")
            trial = copy.copy(args)
            trial.method, trial.pose_mode = method, mode
            trial.patches, trial.patch_size, trial.sagittal_patches = 4, 64, 2
            trial.width, trial.bands, trial.lr, trial.pose_lr = 128, 10, .0005, .002
            print(f"开始 {method}/{mode}；steps={trial.steps}；输出={output}", flush=True)
            train_coronal_variant(data, data.guidance, output, trial, conditioned=False,
                                  smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
