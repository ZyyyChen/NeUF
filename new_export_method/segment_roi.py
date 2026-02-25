# segment_roi.py
# This code aims to segment the different ROIs using a 3D region growing algorithm.

import cv2
import numpy as np
import shutil
from itertools import product
from collections import deque

from volume_data import get_frames
from pathlib import Path


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def get_26neighbors(volume_shape, x, y, z):
    max_z, max_y, max_x = (d - 1 for d in volume_shape)
    neighbors = []
    for dx, dy, dz in product((-1, 0, 1), repeat=3):
        if dx == dy == dz == 0:
            continue
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx <= max_x and 0 <= ny <= max_y and 0 <= nz <= max_z:
            neighbors.append((nx, ny, nz))
    return neighbors


def region_growing_3d(volume, seed, threshold=5, max_distance=50):
    output = np.zeros_like(volume, dtype=np.uint8)
    queue = deque([seed])
    visited = set(seed)
    
    seed_intensity = int(volume[seed[2], seed[1], seed[0]])
    min_int = max(seed_intensity - threshold, 0)
    max_int = min(seed_intensity + threshold, 255)

    while queue:
        x, y, z = queue.popleft()
        # check distance from seed
        if np.sqrt((x-seed[0])**2 + (y-seed[1])**2) > max_distance:
        # if np.sqrt((x-seed[0])**2 + (y-seed[1])**2 + (z-seed[2])**2) > max_distance:
            continue
        output[z, y, x] = 255
        for coords in get_26neighbors(volume.shape, x, y, z):
            if coords not in visited:
                intensity = int(volume[coords[2], coords[1], coords[0]])
                if min_int <= intensity <= max_int:
                    output[coords[2], coords[1], coords[0]] = 255
                    queue.append(coords)
                visited.add(coords)

    return output


def get_seed(image):
    clicks = []
    def on_click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # convert rotated coords back to original coords
            X = y
            Y = image.shape[0] - 1 - x
            print('Seed (original):', X, Y)
            clicks.append((X, Y))

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_click)
    cv2.imshow('image', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey()
    cv2.destroyAllWindows()

    seed = clicks[-1]

    return seed, clicks


def draw_contours(image, output):
    image = image.copy()

    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (255, 255, 255), 2)
    
    return image


def keep_largest_component(mask, diameter=100):
    # get connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # get largest connected component
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 if num_labels > 1 else 0
    largest_component_mask = (labels == largest_component).astype(np.uint8)

    # check if the largest component is larger than the specified diameter
    if stats[largest_component, cv2.CC_STAT_WIDTH] > diameter or stats[largest_component, cv2.CC_STAT_HEIGHT] > diameter:
        return np.zeros_like(mask, dtype=np.uint8)

    return largest_component_mask * 255


def overlay_mask_on_image(image, mask, name="output"):
    overlay = cv2.addWeighted(image, 1, mask, 1, 0)
    cv2.imshow(name, overlay)


def get_seg_parameters(phantom_type, index):

    parameters = {
        "bluephantom": {
            "threshold": 5,
            "max_distance": 35,
        },
        "jfr": {
            "target": {
                "threshold": 15,
                "max_distance": 15,
            },
            "obstacles": {
                "threshold": 40,
                "max_distance": 12,
            }
        },
        "simu": {
            "threshold": 20,
            "max_distance": 100,
        },
    }

    seg_parameters = parameters[phantom_type]
    
    # param selection
    if phantom_type == "jfr":
        if index == 0: 
            params = seg_parameters["target"]
        else: 
            params = seg_parameters["obstacles"]
    elif phantom_type == "bluephantom":
        params = seg_parameters
    elif phantom_type == "simu":
        params = seg_parameters
    else:
        raise RuntimeError(f"Phantom type {phantom_type} undefined")
    
    return params


def compute_region_growing_3d(frames, img_index, seed, delta_frames=None, threshold=5, max_distance=10, show=False):

    # Create a 3D volume from the selected frames
    input_volume = np.stack(frames[img_index - delta_frames:img_index + delta_frames + 1], axis=0) if delta_frames else np.stack(frames, axis=0)
    print(input_volume.shape)

    seed_3d = (seed[0], seed[1], delta_frames) if delta_frames else (seed[0], seed[1], seed[2])

    # run region growing on the 3D volume
    output_volume = region_growing_3d(input_volume, seed_3d, threshold=threshold, max_distance=max_distance)

    if show:
        for i in range(len(input_volume)):
            overlay_mask_on_image(input_volume[i], keep_largest_component(output_volume[i]), "output_volume_" + str(i))

        cv2.waitKey()
        cv2.destroyAllWindows()
    
    return output_volume


def run_segment_roi(
        slices_dir,
        seg_out_dir,
        seed=None,
        seed_index=None,
        interactive_seed: bool = False,
        threshold=5,
        max_distance=30,
        delta_frames=None,
        keep_largest=False,
        phantom_type="bluephantom",
        show=False,
    ):

    slices_dir = Path(slices_dir)
    seg_out_dir = Path(seg_out_dir)
    if seg_out_dir.exists():
        shutil.rmtree(seg_out_dir, ignore_errors=True)
    seg_out_dir.mkdir(parents=True, exist_ok=True)

    frames = get_frames(str(slices_dir))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {slices_dir}")

    if seed_index is None:
        seed_index = len(frames) // 2

    clicks = []
    if seed is None and interactive_seed:
        seed, clicks = get_seed(frames[seed_index])
    elif seed is not None:
        clicks = [seed]
    else:
        raise RuntimeError("Seed not provided: pass --seed-x/--seed-y or enable --interactive-seed")

    nb_seeds = len(clicks)

    for i, seed in enumerate(clicks):

        params = get_seg_parameters(phantom_type, i)

        print(f"Processing seed {i+1}/{nb_seeds} at {seed}")
        seed_3d = (seed[0], seed[1], seed_index)
        volume = compute_region_growing_3d(
            frames=frames,
            img_index=seed_index,
            seed=seed_3d,
            delta_frames=delta_frames,
            threshold=params['threshold'],
            max_distance=params['max_distance'],
            show=show,
        )
        seg_out_dir_seed = seg_out_dir.parent / f"{seg_out_dir.name}/seed{i}"
        seg_out_dir_seed.mkdir(parents=True, exist_ok=True)
        for j in range(volume.shape[0]):
            slice_mask = volume[j, :, :]
            if keep_largest:
                slice_mask = keep_largest_component(slice_mask)
            cv2.imwrite(str(seg_out_dir_seed / f"us{j}.jpg"), slice_mask)

    return str(seg_out_dir)