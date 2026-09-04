from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tqdm

from neuf.dataset import Dataset, validate_checkpoint_dataset_geometry
from neuf.nerf_network import NeRF


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Export a retained HASH/DUAL_HASH checkpoint on a Cartesian grid."
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--output", default="exports/full_grid", type=Path)
    parser.add_argument("--dataset", type=Path, help="Override checkpoint dataset path")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Grid spacing; defaults to the baked in-plane spacing for all axes",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--component",
        choices=["intensity", "anatomy", "speckle"],
        default="intensity",
    )
    parser.add_argument("--save-float-output", action="store_true")
    parser.add_argument("--no-bbox-mask", dest="use_bbox_mask", action="store_false", default=True)
    return parser.parse_args(argv)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def load_dataset(checkpoint: dict, override: Path | None = None) -> tuple[Dataset, Path]:
    value = override or checkpoint.get("baked_dataset_file") or checkpoint.get("dataset")
    if value is None:
        raise KeyError("Checkpoint does not record a baked dataset path; pass --dataset")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Baked dataset not found: {path}")
    dataset = Dataset.open_from_save(path)
    return dataset, path


def build_axis(minimum: float, maximum: float, spacing: float) -> np.ndarray:
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Grid spacing must be positive and finite")
    count = max(2, int(np.ceil((maximum - minimum) / spacing)) + 1)
    return np.linspace(minimum, maximum, count, dtype=np.float32)


def resolve_axes(dataset: Dataset, spacing_xyz: tuple[float, float, float] | None):
    if spacing_xyz is None:
        in_plane = min(
            float(dataset.roi_px_size_width_mm),
            float(dataset.roi_px_size_height_mm),
        )
        spacing_xyz = (in_plane, in_plane, in_plane)
    point_min = np.asarray(dataset.point_min, dtype=np.float32)
    point_max = np.asarray(dataset.point_max, dtype=np.float32)
    axes = tuple(
        build_axis(float(point_min[index]), float(point_max[index]), spacing_xyz[index])
        for index in range(3)
    )
    actual_spacing = tuple(
        float(axis[1] - axis[0]) if len(axis) > 1 else float(spacing_xyz[index])
        for index, axis in enumerate(axes)
    )
    return axes, actual_spacing


def query_model_component(
    model: NeRF,
    points: torch.Tensor,
    directions: torch.Tensor,
    *,
    alpha: float = 1.0,
    component: str = "intensity",
) -> torch.Tensor:
    if component == "intensity":
        return model.query(points, directions, alpha=alpha)
    if model.field_head != NeRF.ANATOMY_SPECKLE_FIELD_HEAD:
        raise ValueError(
            f"field_head={model.field_head} only supports component='intensity'"
        )
    return model.query_components(points, directions, alpha=alpha)[component]


@torch.no_grad()
def query_grid(
    model: NeRF,
    checkpoint: dict,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    *,
    chunk_size: int = 65536,
    use_bbox_mask: bool = True,
    alpha: float = 1.0,
    component: str = "intensity",
) -> np.ndarray:
    """Return a float volume in ``[z, y, x]`` order."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    model.eval()
    model.training_progress = 1.0
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    xy = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    slices = []
    bbox = checkpoint.get("bounding_box")
    if bbox is not None:
        bbox_min = torch.as_tensor(bbox[0], dtype=torch.float32, device=DEVICE)
        bbox_max = torch.as_tensor(bbox[1], dtype=torch.float32, device=DEVICE)

    for z_value in tqdm.tqdm(z_axis, desc="Querying volume"):
        coordinates = np.column_stack(
            (xy, np.full(len(xy), z_value, dtype=np.float32))
        ).astype(np.float32, copy=False)
        output = np.zeros(len(coordinates), dtype=np.float32)
        for start in range(0, len(coordinates), chunk_size):
            stop = min(start + chunk_size, len(coordinates))
            points = torch.as_tensor(coordinates[start:stop], device=DEVICE)
            if use_bbox_mask and bbox is not None:
                valid = torch.all((points >= bbox_min) & (points <= bbox_max), dim=-1)
            else:
                valid = torch.ones(len(points), dtype=torch.bool, device=DEVICE)
            if torch.any(valid):
                directions = torch.zeros_like(points[valid])
                values = query_model_component(
                    model,
                    points[valid],
                    directions,
                    alpha=alpha,
                    component=component,
                )
                chunk_output = torch.zeros(len(points), device=DEVICE)
                chunk_output[valid] = values.reshape(-1)
                output[start:stop] = chunk_output.cpu().numpy()
        slices.append(output.reshape(len(y_axis), len(x_axis)))
    return np.stack(slices, axis=0)


def display_uint8(volume: np.ndarray, component: str) -> tuple[np.ndarray, list[float]]:
    if component == "speckle":
        window = (-0.5, 0.5)
    else:
        window = (0.0, 1.0)
    scaled = (np.clip(volume, *window) - window[0]) / (window[1] - window[0])
    return np.rint(scaled * 255.0).astype(np.uint8), [float(window[0]), float(window[1])]


def save_mhd(
    volume_zyx: np.ndarray,
    output_dir: Path,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "volume.raw"
    mhd_path = output_dir / "volume.mhd"
    np.ascontiguousarray(volume_zyx).tofile(raw_path)
    z_size, y_size, x_size = volume_zyx.shape
    element_type = "MET_UCHAR" if volume_zyx.dtype == np.uint8 else "MET_FLOAT"
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            f"Offset = {origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            f"ElementSpacing = {spacing_xyz[0]} {spacing_xyz[1]} {spacing_xyz[2]}",
            f"DimSize = {x_size} {y_size} {z_size}",
            f"ElementType = {element_type}",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    mhd_path.write_text(header, encoding="ascii")


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt)
    dataset, dataset_path = load_dataset(checkpoint, args.dataset)
    validate_checkpoint_dataset_geometry(
        checkpoint,
        dataset,
        checkpoint_path=args.ckpt,
    )
    model = NeRF(checkpoint)
    if (
        args.component != "intensity"
        and model.field_head != NeRF.ANATOMY_SPECKLE_FIELD_HEAD
    ):
        raise ValueError(
            f"field_head={model.field_head} only supports --component intensity"
        )
    axes, spacing_xyz = resolve_axes(
        dataset,
        None if args.spacing is None else tuple(args.spacing),
    )
    volume = query_grid(
        model,
        checkpoint,
        axes[0],
        axes[1],
        axes[2],
        chunk_size=args.chunk_size,
        use_bbox_mask=args.use_bbox_mask,
        alpha=args.alpha,
        component=args.component,
    )
    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_float_output:
        np.save(output_dir / "volume_float.npy", volume.astype(np.float32, copy=False))
    display, display_window = display_uint8(volume, args.component)
    save_mhd(display, output_dir, spacing_xyz, np.asarray(dataset.point_min))
    metadata = {
        "checkpoint": str(args.ckpt.resolve()),
        "dataset": str(dataset_path.resolve()),
        "encoding": model.encoding_type,
        "field_head": model.field_head,
        "component": args.component,
        "alpha": float(args.alpha),
        "shape_zyx": list(volume.shape),
        "spacing_xyz_mm": list(spacing_xyz),
        "point_min_xyz_mm": np.asarray(dataset.point_min).tolist(),
        "point_max_xyz_mm": np.asarray(dataset.point_max).tolist(),
        "display_window": display_window,
        "float_output_saved": bool(args.save_float_output),
        "quantitative_warning": (
            "MHD is fixed-window display output; use volume_float.npy for analysis."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
