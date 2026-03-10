# export_slices.py
# This code aims to export a series of reconstructed US slices (using the .pkl of the trained nerf model as input).
# The slices are exported alongside a chosen axis (X-axis in our case).


import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import math
import sys
import cv2

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from dataset_1 import Dataset, Quat, get_base_points, get_oriented_points_and_views
from slice_renderer import SliceRenderer
from nerf_network import NeRF
from volume_data import VolumeData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pos_and_rot(infos_json, num_image):
    info_image = infos_json[num_image]
    
    pos = np.array([info_image["x"], info_image["y"], info_image["z"]])
    rot = Quat(float(info_image["w0"]), -float(info_image["w1"]), -float(info_image["w2"]), -float(info_image["w3"]))

    return pos, rot


def get_scan_corners(dataset, pos, rot, volume_data=None):
    if volume_data:
        bbox_size = volume_data.volume_size
        # X, Y = get_base_points(dataset.width,dataset.height,dataset.px_width,dataset.px_height)
        X, Y = get_base_points(bbox_size[2],bbox_size[1],dataset.px_width,dataset.px_height)
    else:
        X, Y = get_base_points(dataset.width,dataset.height,dataset.px_width,dataset.px_height)

    p, v = get_oriented_points_and_views(X, Y, pos, rot)

    # Reshape into image grid
    points_img = p.reshape((int(dataset.px_height), int(dataset.px_width), 3))
    viewdir_img = v.reshape((int(dataset.px_height), int(dataset.px_width), 3))

    # get the viewdir of the center of the image
    center_x = int(dataset.px_width / 2)
    center_y = int(dataset.px_height / 2)
    center_viewdir = viewdir_img[center_y, center_x]

    # Extract corners
    top_left     = points_img[0, 0]
    top_right    = points_img[0, -1]
    bottom_left  = points_img[-1, 0]
    bottom_right = points_img[-1, -1]

    return np.array([top_left,top_right,bottom_left,bottom_right]), center_viewdir


def get_quat_from_angle_and_axis(theta, axis="x"):
    all_axis = {
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1]
    }
    c, s = math.cos(theta), math.sin(theta)
    return Quat(c, all_axis[axis][0] * s, all_axis[axis][1] * s, all_axis[axis][2] * s)


def get_rotation():
    rot = get_quat_from_angle_and_axis(theta=-math.pi/4, axis='z') # rotate around Z-axis
    rot.normalize()
    return rot


def get_all_scan_corners_and_viewdirs(dataset, volume_data, new_points=None, from_dataset=False, infos_json=None, nb_images=None):
    """Get the different scan viewdirs and corners for the different positions"""
    positions, rotations, scan_corners, scan_viewdirs = [], [], [], []
    if from_dataset:
        for i in tqdm(range(0, nb_images, 5)):
            pos, rot = get_pos_and_rot(infos_json, num_image=str(i))
            p, v = get_scan_corners(dataset, pos, rot)
            positions.append(pos)
            rotations.append(rot)
            scan_corners.append(p)
            scan_viewdirs.append(v)
    else:
        rot = get_rotation()
        for pos in tqdm(new_points):
            p, v = get_scan_corners(dataset, pos, rot, volume_data)
            positions.append(pos)
            rotations.append(rot)
            scan_corners.append(p)
            scan_viewdirs.append(v)
    
    return positions, rotations, scan_corners, scan_viewdirs


def get_new_acquisiton_pts(point_min, point_max, axis="x", num_points=50):
    x_min, x_max = sorted([point_min[0], point_max[0]])
    y_min, y_max = sorted([point_min[1], point_max[1]])
    z_min, z_max = sorted([point_min[2], point_max[2]])
    
    if axis == "x":
        axis_points = np.linspace(x_min, x_max, num_points)
        # y, z = (y_min + y_max) / 2, z_min
        y, z = y_min, (z_min + z_max) / 2

    new_points = np.array([[x, y, z] for x in axis_points])

    return new_points


def plot_orientation(pos, quat, ax, length=10.0):
    # Convert quaternion to rotation matrix
    rot_matrix = quat.as_rotmat()
    # Basis vectors
    basis = np.eye(3)
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    for i in range(3):
        vec = rot_matrix @ basis[i] * length
        ax.quiver(
            pos[0], pos[1], pos[2],
            vec[0], vec[1], vec[2],
            color=colors[i], label=f'{labels[i]}-axis'
        )


def plot_3d(*args):
    inputs = list(args)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pos in inputs:
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()

    return ax


def run_export_slices(
        input_folder,
        model_path,
        dataset_path,
        output_dir,
        num_slices=150,
        axis= "x"
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)

    # export_folder = os.path.join(input_folder, "sync", "export")
    export_folder = input_folder
    with open(os.path.join(export_folder, "infos.json")) as f:
        infos = json.load(f)

    ckpt = torch.load(model_path, map_location=device)
    nerf = NeRF(ckpt)
    dataset = Dataset.open_from_save(dataset_path)
    slice_renderer = SliceRenderer(dataset)

    nb_images = len(os.listdir(os.path.join(export_folder,"us")))
    print("nb_images: ", nb_images)

    volume_data = VolumeData(
        point_min=ckpt["bounding_box"][0].detach().cpu().numpy(),
        point_max=ckpt["bounding_box"][1].detach().cpu().numpy(),
        volume_shape=(num_slices, dataset.px_height, dataset.px_width),
        metadata={
            "dataset_path": str(dataset_path),
            "ckpt_path": str(model_path),
            "nb_images": nb_images,
            "dataset_info": {
                "width": dataset.width,
                "height": dataset.height,
                "px_width": dataset.px_width,
                "px_height": dataset.px_height,
            },
        },
    )

    volume_data_path = os.path.join(input_folder, "volume_data.json")
    volume_data.save(volume_data_path)

    new_points = get_new_acquisiton_pts(volume_data.point_min, volume_data.point_max, axis, num_slices)
    new_rot = get_rotation()

    bbox_size = volume_data.volume_size
    # new_pix_grid = get_base_points(dataset.width,dataset.height,dataset.px_width,dataset.px_height)
    new_pix_grid = get_base_points(bbox_size[2],bbox_size[1],dataset.px_width,dataset.px_height)


    output_dir.mkdir(parents=True, exist_ok=True)
    rotated_output_dir = output_dir.parent / "rotated"
    rotated_output_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(new_points))):
        rendered_slice = (
            slice_renderer.render_slice_for_chosen_grid(nerf, 
                                              X=new_pix_grid[0], 
                                              Y=new_pix_grid[1], 
                                              pos=new_points[i], 
                                              rot=new_rot, 
                                              reshaped=True)
            .detach()
            .cpu()
            .numpy()
        )
        plt.imsave(os.path.join(output_dir, f"us{i}.jpg"), rendered_slice, cmap="gray")
        # save a rotated version of the image
        plt.imsave(os.path.join(rotated_output_dir, f"us{i}.jpg"), cv2.rotate(rendered_slice, cv2.ROTATE_90_CLOCKWISE), cmap="gray")

    return str(volume_data_path)


if __name__ == "__main__":
    test_data_folder = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/"

    folder = os.path.join(test_data_folder, "latest")
    export_folder = os.path.join(folder,"sync","export")
    
    with open(os.path.join(export_folder, "infos.json")) as f:
        infos = json.load(f)

    nb_images = len(os.listdir(os.path.join(export_folder,"us")))
    print("nb Images: ",nb_images)

    ckpt = torch.load("latest/ckpt.pkl", map_location=device)
    nerf = NeRF(ckpt)
    dataset = Dataset.open_from_save("datasets/baked/test/latest.pkl")
    slice_renderer = SliceRenderer(dataset)
    
    nb_desired_slices = 5
    
    volume_data = VolumeData(
        point_min=ckpt["bounding_box"][0].detach().cpu().numpy(),
        point_max=ckpt["bounding_box"][1].detach().cpu().numpy(),
        volume_shape=(nb_desired_slices, dataset.px_height, dataset.px_width),
        metadata={
            "dataset_path": os.path.join(folder, "dataset.pkl"),
            "ckpt_path": os.path.join(folder, "ckpt.pkl"),
            "nb_images": nb_images,
            "dataset_info": {
                "width": dataset.width,
                "height": dataset.height,
                "px_width": dataset.px_width,
                "px_height": dataset.px_height
            }
        }
    )
    # volume_data.save(os.path.join(folder, "volume_data.json"))

    positions, rotations, scan_corners, scan_viewdirs = get_all_scan_corners_and_viewdirs(dataset, 
                                                                                          volume_data=volume_data, 
                                                                                          from_dataset=True, 
                                                                                          infos_json=infos, 
                                                                                          nb_images=nb_images)

    print("width, height:",
          np.linalg.norm(scan_corners[0][1] - scan_corners[0][0]),
          np.linalg.norm(scan_corners[0][2] - scan_corners[0][0]))

    new_points = get_new_acquisiton_pts(volume_data.point_min, volume_data.point_max, "x", nb_desired_slices)

    new_positions, new_rotations, new_scan_corners, new_scan_viewdirs = get_all_scan_corners_and_viewdirs(dataset, 
                                                                                          volume_data=volume_data, 
                                                                                          new_points=new_points, 
                                                                                          from_dataset=False)

    width_distance = np.linalg.norm(new_scan_corners[0][1] - new_scan_corners[0][0])
    height_distance = np.linalg.norm(new_scan_corners[0][2] - new_scan_corners[0][0])
    print(width_distance, height_distance)
    print(dataset.width, dataset.height)

    ax = plot_3d(np.array(positions), 
                 np.concatenate(scan_corners),
                 np.concatenate(new_scan_corners), 
                 volume_data.get_corners(), 
                 new_points)

    for pos, viewdir in zip(positions, scan_viewdirs):
        imdir = -viewdir
        ax.quiver(pos[0], pos[1], pos[2], imdir[0], imdir[1], imdir[2], length=10, color="red")

    for pos, viewdir in zip(new_points, new_scan_viewdirs):
        imdir = -viewdir
        ax.quiver(pos[0], pos[1], pos[2], imdir[0], imdir[1], imdir[2], length=10, color="red")

    bbox_size = volume_data.volume_size
    print(bbox_size[1],bbox_size[2],dataset.px_width,dataset.px_height)
    plt.show()

    # os.makedirs(os.path.join("test_data", "out"), exist_ok=True)
    # new_pix_grid = get_base_points(dataset.width,dataset.height,dataset.px_width,dataset.px_height)
    # rot = new_rotations[0]
    # for i in tqdm(range(len(new_points))):
    #     # slice = slice_renderer.render_slice(nerf, pos=new_points[i], rot=rot, reshaped=True).detach().cpu().numpy()
    #     slice = slice_renderer.render_slice_for_chosen_grid(nerf, X=new_pix_grid[0], Y=new_pix_grid[1], pos=new_points[i], rot=rot, reshaped=True).detach().cpu().numpy()
    #     plt.imsave(os.path.join("test_data", "out", f"us{i}.jpg"), slice, cmap="gray")