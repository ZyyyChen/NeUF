import json
import numpy as np
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                           device=device)


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result



def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0],device=device) * grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

def get_base_points(width, height, width_px, height_px):
    pixel_size_W = width / width_px
    pixel_size_H = height / height_px

    start_x = - width / 2 + pixel_size_W / 2
    start_y = 0 + pixel_size_H / 2

    Y = start_y + np.repeat(np.arange(0, height_px, 1), width_px) * pixel_size_H
    X = start_x + np.tile(np.arange(0, width_px, 1), height_px) * pixel_size_W
    return X, Y

def get_oriented_points_and_views(X_base_points, Y_base_points, position, rotation):
    width_points = X_base_points
    height_points = Y_base_points


    points = (np.apply_along_axis(rotation.apply_quat, 1,
                                  np.stack((height_points, np.zeros(height_points.shape), width_points),
                                           axis=1)) + position)

    viewdirs = np.apply_along_axis(rotation.apply_quat, 1, np.stack(
        (-np.ones_like(height_points), np.zeros_like(height_points), np.zeros_like(height_points)), axis=1))

    # points = torch.from_numpy(points.astype(dtype=np.float32)).to(device)
    # viewdirs = torch.from_numpy(viewdirs.astype(dtype=np.float32)).to(device)


    return points, viewdirs

