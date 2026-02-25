import argparse
import os
import sys
from pathlib import Path
import time

# Allow importing project modules (dataset, nerf_network, slice_renderer)
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from export_slices import run_export_slices
from segment_roi import run_segment_roi
from export_segmentation import run_export_segmentation


def main(args):
    # Step 1: Export slices
    volume_data_path = os.path.join(args.input_folder,"volume_data.json")
    if not args.skip_slices:
        do_slices = input("Do you want to export slices? (y/n): ")
        if do_slices.lower() != 'n':
            print("[1/3] Exporting slices ...", flush=True)
            start = time.time()
            volume_data_path_str = run_export_slices(
                input_folder=args.input_folder,
                model_path=args.model_path,
                dataset_path=args.dataset_path,
                output_dir=args.slices_out,
                num_slices=args.num_slices,
                axis=args.axis,
            )
            volume_data_path = Path(volume_data_path_str)
            print(f"Slices saved to {args.slices_out}, took {int(time.time()-start)} seconds")
    else:
        print("[1/3] Skipped export slices.")

    # Step 2: ROI segmentation
    phantoms = {1: "bluephantom",
                2: "jfr",
                3: "simu"}
    phantom_num = int(input("Which phantom are you working on (1=bluephantom ; 2=jfr ; 3=simu)?  "))
    if not args.skip_seg:
        print("[2/3] Running ROI segmentation ...", flush=True)
        seed_tuple = (args.seed_x, args.seed_y) if args.seed_x is not None and args.seed_y is not None else None
        if args.select_seed_index:
            seed_index = input(f"Enter seed slice index (0 to {args.num_slices - 1}): ")
            args.seed_index = int(seed_index)
        start = time.time()
        run_segment_roi(
            slices_dir=args.slices_out,
            seg_out_dir=args.seg_out,
            seed=seed_tuple,
            seed_index=args.seed_index,
            interactive_seed=args.interactive_seed,
            threshold=args.threshold,
            max_distance=args.max_distance,
            delta_frames=args.delta_frames,
            keep_largest=args.keep_largest,
            phantom_type=phantoms[phantom_num],
            show=False,
        )
        print(f"Segmentation saved to {args.seg_out}, took {round(time.time()-start,2)} seconds")
    else:
        print("[2/3] Skipped ROI segmentation.")

    # Step 3: Mesh export
    if not args.skip_mesh:
        print("[3/3] Exporting mesh from segmentation ...", flush=True)
        start = time.time()
        mesh_path_str = run_export_segmentation(
            seg_dir=args.seg_out,
            volume_data_path=volume_data_path,
            mesh_out=args.mesh_out,
            phantom_type=phantoms[phantom_num],
            step_size=args.mesh_step_size,
        )
        mesh_path = Path(mesh_path_str)
        print(f"Mesh exported to {mesh_path}, took {round(time.time()-start,2)} seconds")
    else:
        print("[3/3] Skipped mesh export.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run slices -> ROI segmentation -> mesh export pipeline")

    # Inputs / Outputs
    parser.add_argument("--input-folder", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56"), help="Folder containing the acquisition")
    # parser.add_argument("--model-path", type=Path, default=Path("D:\\0-Code\\NeUF\\logs\\04-01-2026\\NONE_simu_56_0\\checkpoints\\3000.pkl"), help="Path to the ckpt model file")
    parser.add_argument("--model-path", type=Path, default=Path("D:\\0-Code\\NeUF\\logs\\05-01-2026\\NONE_simu_56_3\\checkpoints\\3000.pkl"), help="Path to the ckpt model file")

    # parser.add_argument("--dataset-path", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56\\baked_simu_56.pkl"), help="Path to the dataset JSON file")
    parser.add_argument("--dataset-path", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56\\init.pkl"), help="Path to the dataset JSON file")
    
    parser.add_argument("--slices-out", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56\\out"), help="Output directory for rendered slices")
    parser.add_argument("--seg-out", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56\\out_seg"), help="Output directory for segmentation masks")
    parser.add_argument("--mesh-out", type=Path, default=Path("D:\\0-Code\\NeUF\\data\\simu_56\\isosurface.glb"), help="Output GLB path for exported mesh")

    # Slices params
    parser.add_argument("--num-slices", type=int, default=150)
    parser.add_argument("--axis", type=str, choices=["x", "y", "z"], default="x")
    parser.add_argument("--skip-slices", action="store_true")

    # Segmentation params
    parser.add_argument("--seed-x", type=int, default=None)
    parser.add_argument("--seed-y", type=int, default=None)
    parser.add_argument("--seed-index", type=int, default=None, help="Slice index for the seed (0-based)")
    parser.add_argument("--select-seed-index", action="store_true", help="Prompt the user to select the seed index interactively")
    parser.add_argument("--interactive-seed", action="store_false")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--max-distance", type=int, default=30)
    parser.add_argument("--delta-frames", type=int, default=None)
    parser.add_argument("--keep-largest", action="store_true")
    # parser.add_argument("--phantom_type", type=str, default="bluephantom")
    parser.add_argument("--skip-seg", action="store_true")

    # Mesh params
    parser.add_argument("--mesh-step-size", type=int, default=1)
    parser.add_argument("--skip-mesh", action="store_true")

    cli_args = parser.parse_args()
    sys.exit(main(cli_args))


