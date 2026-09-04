from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import binary_erosion

from neuf.dataset import Dataset, Slice


MINIMUM_TRANSVERSE_SLICES = 50


class Phase1DataBlockedError(RuntimeError):
    """Raised when frozen Phase 1 data rules cannot be satisfied."""


@dataclass(frozen=True)
class FrameRef:
    source: str
    source_index: int
    original_frame_index: int
    stable_slice_id: str
    slice_info: Slice
    pixel_hash: str
    pose_hash: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(value) -> str:
    array = np.ascontiguousarray(torch.as_tensor(value).detach().cpu().numpy())
    descriptor = f"{array.dtype.str}|{array.shape}|".encode("utf-8")
    return sha256_bytes(descriptor + array.tobytes())


def hash_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pose_array(slice_info: Slice) -> np.ndarray:
    rotation = slice_info.rotation
    quaternion = [rotation.w, rotation.x, rotation.y, rotation.z]
    return np.concatenate(
        [np.asarray(slice_info.position, dtype=np.float32), np.asarray(quaternion, dtype=np.float32)]
    )


def pose_hash(slice_info: Slice) -> str:
    return hash_array(pose_array(slice_info))


def metric_mask(dataset: Dataset, erosion_px: int = 3) -> np.ndarray:
    base = dataset.get_sector_mask().detach().cpu().numpy().astype(bool)
    if erosion_px < 0:
        raise ValueError("erosion_px must be >= 0")
    if erosion_px == 0:
        return base
    structure = np.ones((3, 3), dtype=bool)
    return binary_erosion(base, structure=structure, iterations=erosion_px, border_value=0)


def mask_signature(mask: np.ndarray, *, erosion_px: int) -> dict:
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    return {
        "shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "valid_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "valid_fraction": float(mask.mean()),
        "erosion_px": int(erosion_px),
        "sha256": hash_array(mask),
    }


def _slice_pixels(dataset: Dataset, source: str, index: int) -> torch.Tensor:
    slices = dataset.slices if source == "training_pool" else dataset.slices_valid
    pixels = dataset.pixels if source == "training_pool" else dataset.pixels_valid
    item = slices[index]
    return pixels[item.start:item.end]


def _frame_refs(dataset: Dataset) -> tuple[list[FrameRef], list[FrameRef]]:
    def build(source: str, slices: list[Slice]) -> list[FrameRef]:
        refs: list[FrameRef] = []
        for source_index, item in enumerate(slices):
            pixels = _slice_pixels(dataset, source, source_index)
            image_hash = hash_array(pixels)
            frame_index = getattr(item, "frame_index", None)
            if frame_index is None:
                frame_index = source_index
            stable_id = f"{source}_{int(frame_index):06d}_{image_hash[:12]}"
            refs.append(
                FrameRef(
                    source=source,
                    source_index=source_index,
                    original_frame_index=int(frame_index),
                    stable_slice_id=stable_id,
                    slice_info=item,
                    pixel_hash=image_hash,
                    pose_hash=pose_hash(item),
                )
            )
        return refs

    return build("training_pool", dataset.slices), build("held_out_pool", dataset.slices_valid)


def frame_content_key(ref: FrameRef) -> tuple[str, str]:
    """Return the exact image-and-pose identity used to prevent split leakage."""
    return ref.pixel_hash, ref.pose_hash


def split_content_overlaps(
    splits: dict[str, list[FrameRef]],
) -> dict[str, int]:
    """Count exact image-and-pose duplicates shared by each pair of splits."""
    content_sets = {
        name: {frame_content_key(ref) for ref in refs}
        for name, refs in splits.items()
    }
    return {
        f"{left}_vs_{right}": len(content_sets[left] & content_sets[right])
        for left, right in (
            ("training", "validation"),
            ("training", "test"),
            ("validation", "test"),
        )
    }


def _require_unique_content(refs: list[FrameRef], source: str) -> None:
    seen: dict[tuple[str, str], str] = {}
    for ref in refs:
        key = frame_content_key(ref)
        if key in seen:
            raise Phase1DataBlockedError(
                f"{source} contains duplicate image-and-pose content: "
                f"{seen[key]} and {ref.stable_slice_id}"
            )
        seen[key] = ref.stable_slice_id


def build_phase1_split(
    dataset: Dataset,
    *,
    allow_small_dataset: bool = False,
) -> dict[str, list[FrameRef]]:
    training_pool, held_out_pool = _frame_refs(dataset)
    _require_unique_content(training_pool, "training_pool")
    _require_unique_content(held_out_pool, "held_out_pool")
    total = len(
        {
            frame_content_key(ref)
            for ref in training_pool + held_out_pool
        }
    )
    if total < MINIMUM_TRANSVERSE_SLICES and not allow_small_dataset:
        raise Phase1DataBlockedError(
            f"Phase 1 requires at least {MINIMUM_TRANSVERSE_SLICES} transverse slices; "
            f"dataset contains {total}"
        )

    if held_out_pool:
        test = held_out_pool
        test_content = {frame_content_key(ref) for ref in test}
        validation_ids = {
            ref.stable_slice_id
            for pool_index, ref in enumerate(training_pool)
            if pool_index % 10 == 5 and frame_content_key(ref) not in test_content
        }
        validation = [ref for ref in training_pool if ref.stable_slice_id in validation_ids]
        training = [
            ref
            for ref in training_pool
            if ref.stable_slice_id not in validation_ids
            and frame_content_key(ref) not in test_content
        ]
    else:
        validation = [ref for index, ref in enumerate(training_pool) if index % 10 == 5]
        test = [ref for index, ref in enumerate(training_pool) if index % 10 == 0]
        held_out = {ref.stable_slice_id for ref in validation + test}
        training = [ref for ref in training_pool if ref.stable_slice_id not in held_out]

    splits = {"training": training, "validation": validation, "test": test}
    id_sets = {name: {ref.stable_slice_id for ref in refs} for name, refs in splits.items()}
    if any(
        id_sets[left] & id_sets[right]
        for left, right in (("training", "validation"), ("training", "test"), ("validation", "test"))
    ):
        raise RuntimeError("Phase 1 split construction produced overlapping frame IDs")
    content_overlaps = split_content_overlaps(splits)
    if any(content_overlaps.values()):
        raise RuntimeError(
            "Phase 1 split construction produced overlapping image-and-pose content: "
            f"{content_overlaps}"
        )
    if not training or not validation or not test:
        raise Phase1DataBlockedError(
            "Phase 1 split requires non-empty training, validation, and test sets"
        )
    return splits


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _freeze_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise Phase1DataBlockedError(
                    f"Frozen manifest differs from current data/configuration: {path}"
                )
            return

        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            _json_dump(temporary_path, payload)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def freeze_phase1_manifests(
    dataset: Dataset,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    allow_small_dataset: bool = False,
) -> dict[str, list[FrameRef]]:
    """Freeze split, data/geometry hashes, mask, and display IDs before training."""
    output_dir = Path(output_dir)
    manifests_dir = output_dir / "manifests"
    splits = build_phase1_split(dataset, allow_small_dataset=allow_small_dataset)

    manifest_rows = []
    for split_name in ("training", "validation", "test"):
        for ref in splits[split_name]:
            manifest_rows.append(
                {
                    "original_frame_index": ref.original_frame_index,
                    "source_pool": ref.source,
                    "source_index": ref.source_index,
                    "stable_slice_id": ref.stable_slice_id,
                    "pixel_hash": ref.pixel_hash,
                    "pose_hash": ref.pose_hash,
                    "split": split_name,
                }
            )
    split_manifest = {
        "schema_version": 2,
        "dataset_path": str(Path(dataset_path).resolve()),
        "minimum_slice_requirement": MINIMUM_TRANSVERSE_SLICES,
        "small_dataset_smoke_only": bool(
            sum(len(values) for values in splits.values()) < MINIMUM_TRANSVERSE_SLICES
        ),
        "counts": {name: len(values) for name, values in splits.items()},
        "content_identity": ["pixel_hash", "pose_hash"],
        "normalization": {"kind": "fixed", "minimum": 0.0, "maximum": 1.0},
        "frames": manifest_rows,
    }
    _freeze_json(manifests_dir / "split_manifest.json", split_manifest)

    base_mask = dataset.get_sector_mask().detach().cpu().numpy().astype(bool)
    eroded_mask = metric_mask(dataset, erosion_px=3)
    _freeze_json(
        manifests_dir / "metric_mask_signature.json",
        {
            "base_sector_mask": mask_signature(base_mask, erosion_px=0),
            "metric_mask": mask_signature(eroded_mask, erosion_px=3),
            "finite_pixel_intersection": "applied_per_slice_during_evaluation",
            "static_exclusion_mask": None,
        },
    )

    all_refs = splits["training"] + splits["validation"] + splits["test"]
    geometry_hashes = {
        "dataset_file": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": hash_file(dataset_path),
        },
        "all_pose_arrays_sha256": hash_array(np.stack([pose_array(ref.slice_info) for ref in all_refs])),
        "all_pixel_hashes_sha256": sha256_bytes(
            "".join(ref.pixel_hash for ref in all_refs).encode("ascii")
        ),
        "point_min_sha256": hash_array(dataset.point_min),
        "point_max_sha256": hash_array(dataset.point_max),
        "sector_mask_sha256": hash_array(base_mask.astype(np.uint8)),
        "metric_mask_sha256": hash_array(eroded_mask.astype(np.uint8)),
        "image_shape_hw": [int(dataset.px_height), int(dataset.px_width)],
        "pixel_spacing_mm_wh": [
            float(dataset.roi_px_size_width_mm),
            float(dataset.roi_px_size_height_mm),
        ],
    }
    _freeze_json(manifests_dir / "data_geometry_hashes.json", geometry_hashes)

    test_ids = [ref.stable_slice_id for ref in splits["test"]]
    quantile_indices = [int(math.floor(q * (len(test_ids) - 1))) for q in (0.25, 0.50, 0.75)]
    _freeze_json(
        manifests_dir / "display_slice_ids.json",
        {
            "selection": "floor(q * (n_test - 1)) in frozen test order",
            "quantiles": [0.25, 0.50, 0.75],
            "indices": quantile_indices,
            "stable_slice_ids": [test_ids[index] for index in quantile_indices],
        },
    )
    return splits


def _partition_tensor(
    dataset: Dataset,
    refs: Iterable[FrameRef],
    training_name: str,
    held_out_name: str,
) -> torch.Tensor:
    values = []
    for ref in refs:
        source_tensor = getattr(
            dataset,
            training_name if ref.source == "training_pool" else held_out_name,
        )
        values.append(source_tensor[ref.slice_info.start:ref.slice_info.end])
    if not values:
        source = getattr(dataset, training_name)
        return torch.empty((0,) + tuple(source.shape[1:]), dtype=source.dtype, device=source.device)
    return torch.cat(values, dim=0)


def _rebased_slices(refs: Iterable[FrameRef], pixels_per_slice: int) -> list[Slice]:
    output = []
    for index, ref in enumerate(refs):
        output.append(
            Slice(
                start=index * pixels_per_slice,
                end=(index + 1) * pixels_per_slice,
                position=np.array(ref.slice_info.position, dtype=np.float32),
                rotation=ref.slice_info.rotation,
                frame_index=ref.original_frame_index,
            )
        )
    return output


def apply_phase1_training_split(dataset: Dataset, splits: dict[str, list[FrameRef]]) -> None:
    """Replace trainer partitions while retaining a read-only copy of the frozen test set."""
    train_refs = splits["training"]
    valid_refs = splits["validation"]
    test_refs = splits["test"]
    partition_names = (("pixels", "pixels_valid"), ("points", "points_valid"), ("viewdirs", "viewdirs_valid"))
    if getattr(dataset, "has_gt", False):
        partition_names += (("gt", "gt_valid"),)

    tensors = {}
    for training_name, held_out_name in partition_names:
        tensors[f"train_{training_name}"] = _partition_tensor(
            dataset, train_refs, training_name, held_out_name
        )
        tensors[f"valid_{training_name}"] = _partition_tensor(
            dataset, valid_refs, training_name, held_out_name
        )
        tensors[f"test_{training_name}"] = _partition_tensor(
            dataset, test_refs, training_name, held_out_name
        )

    for training_name, held_out_name in partition_names:
        setattr(dataset, training_name, tensors[f"train_{training_name}"])
        setattr(dataset, held_out_name, tensors[f"valid_{training_name}"])
        setattr(dataset, f"phase1_test_{training_name}", tensors[f"test_{training_name}"])
    dataset.slices = _rebased_slices(train_refs, dataset.pixels_per_slice)
    dataset.slices_valid = _rebased_slices(valid_refs, dataset.pixels_per_slice)
    setattr(dataset, "phase1_test_slices", _rebased_slices(test_refs, dataset.pixels_per_slice))
    setattr(
        dataset,
        "phase1_split_ids",
        {name: [ref.stable_slice_id for ref in refs] for name, refs in splits.items()},
    )
    dataset._valid_patch_origins_cache = {}


def current_pose_hash(dataset: Dataset) -> str:
    slices = list(dataset.slices) + list(dataset.slices_valid)
    slices += list(getattr(dataset, "phase1_test_slices", []))
    return hash_array(np.stack([pose_array(item) for item in slices]))
