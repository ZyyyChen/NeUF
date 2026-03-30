import numpy as np
import torch


BOX_OFFSETS = torch.tensor(
    [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
    dtype=torch.int32,
)


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    hash_mask = xor_result.new_tensor((1 << log2_hashmap_size) - 1)
    return hash_mask & xor_result



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
    voxel_max_vertex = voxel_min_vertex + torch.ones_like(grid_size) * grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS.to(
        device=bottom_left_idx.device,
        dtype=bottom_left_idx.dtype,
    )
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

# def get_base_points(width, height, width_px, height_px):
#     pixel_size_W = width / width_px
#     pixel_size_H = height / height_px

#     start_x = - width / 2 + pixel_size_W / 2
#     start_y = 0 + pixel_size_H / 2

#     Y = start_y + np.repeat(np.arange(0, height_px, 1), width_px) * pixel_size_H
#     X = start_x + np.tile(np.arange(0, width_px, 1), height_px) * pixel_size_W
#     return X, Y

def get_base_points(width, height, width_px, height_px, offset_x_mm=0, offset_y_mm=0):
    """
    计算图像每个像素对应的物理坐标
    
    Args:
        width: 裁剪后图像的物理宽度 (mm)
        height: 裁剪后图像的物理高度 (mm)
        width_px: 裁剪后图像的像素宽度
        height_px: 裁剪后图像的像素高度
        offset_x_mm: ROI在原图坐标系中的x偏移 (mm)
        offset_y_mm: ROI在原图坐标系中的y偏移 (mm)
    """
    pixel_size_W = width / width_px
    pixel_size_H = height / height_px

    # 考虑原始图像坐标系的偏移
    start_x = -width / 2 + pixel_size_W / 2 + offset_x_mm
    start_y = 0 + pixel_size_H / 2 + offset_y_mm

    Y = start_y + np.repeat(np.arange(0, height_px, 1), width_px) * pixel_size_H
    X = start_x + np.tile(np.arange(0, width_px, 1), height_px) * pixel_size_W
    return X, Y

def get_oriented_points_and_views(X_base_points, Y_base_points, position, rotation):
    """
    优化版本：使用矩阵乘法替代 apply_along_axis，速度提升 10-100 倍
    """
    # 构建局部坐标点 (N, 3)
    local_points = np.stack((Y_base_points, X_base_points, np.zeros_like(Y_base_points)), axis=1)
    
    # 使用旋转矩阵进行批量变换 (N, 3) @ (3, 3)^T = (N, 3)
    rotmat = rotation.as_rotmat()  # (3, 3)
    points = local_points @ rotmat.T + position  # 广播加法
    
    # 视线方向也是批量计算
    local_viewdirs = np.stack((- np.ones_like(Y_base_points),
                                np.zeros_like(Y_base_points),
                                np.zeros_like(Y_base_points)), axis=1)
    viewdirs = local_viewdirs @ rotmat.T
    
    return points, viewdirs
