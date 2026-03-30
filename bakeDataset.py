from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset import Dataset

DEFAULT_INPUT_DIR = "/home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0"
DEFAULT_OUTPUT_PATH = (
    "/home/zchen/Code/NeUF/data/cerebral_data/Pre_traitement_echo_v2/"
    "Recalage/Patient0/us_recal_original/baked_dataset.pkl"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bake a raw dataset into a serialized .pkl file")
    parser.add_argument(
        "--input-dir",
        "-i",
        default=DEFAULT_INPUT_DIR,
        help="Dataset root folder containing image and infos.json subfolders",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the baked dataset (.pkl)",
    )
    parser.add_argument("--img-folder", default="us_recal_original", help="Relative folder containing input images")
    parser.add_argument("--info-folder", default="us_recal_original", help="Relative folder containing infos.json")
    parser.add_argument("--prefix", default="us", help="Image filename prefix")
    parser.add_argument("--suffix", default=".jpg", help="Image filename suffix")
    parser.add_argument("--image-step", type=int, default=1, help="Load one frame every N frames")
    parser.add_argument("--name", default=None, help="Optional dataset display name")
    parser.add_argument("--reverse-quat", action="store_true", help="Use reversed quaternion convention")
    parser.add_argument(
        "--exclude-valid",
        action="store_true",
        help="Exclude validation slices from the training split in the baked dataset",
    )
    parser.add_argument(
        "--copy-name",
        default="dataset.pkl",
        help="Optional filename copied into the input dataset root after baking",
    )
    return parser.parse_args()


def build_dataset(args) -> Dataset:
    dataset_root = Path(args.input_dir).expanduser()
    dataset_name = args.name or dataset_root.name
    return Dataset(
        str(dataset_root),
        img_folder=args.img_folder,
        info_folder=args.info_folder,
        prefix=args.prefix,
        suffix=args.suffix,
        reverse_quat=args.reverse_quat,
        exclude_valid=args.exclude_valid,
        name=dataset_name,
        image_step=args.image_step,
    )


def main():
    args = parse_args()
    dataset = build_dataset(args)

    output_path = Path(args.output).expanduser()
    if output_path.suffix != ".pkl":
        output_path = output_path.with_suffix(".pkl")

    dataset.save(str(output_path))
    print(f"Baked dataset written to {output_path}")

    if args.copy_name:
        copy_target = Path(args.input_dir).expanduser() / args.copy_name
        shutil.copy2(output_path, copy_target)
        print(f"Copied baked dataset to {copy_target}")


if __name__ == "__main__":
    main()
