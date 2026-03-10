"""
可视化超声探头轨迹
- X: 探头方向（用箭头表示）
- Z: slice的宽度（矩形的长）
- Y: 平面外方向（矩形的宽）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import os


class Arrow3D(FancyArrowPatch):
    """3D箭头绘制类"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        xs = [x1, x1+dx]
        ys = [y1, y1+dy]
        zs = [z1, z1+dz]

        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
    
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        xs = [x1, x1+dx]
        ys = [y1, y1+dy]
        zs = [z1, z1+dz]
        
        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def quaternion_to_rotation_matrix(quat):
    """
    将四元数转换为旋转矩阵
    输入: [w0, w1, w2, w3]
    """
    w0, w1, w2, w3 = quat
    
    # 规范化
    norm = np.sqrt(w0**2 + w1**2 + w2**2 + w3**2)
    w0, w1, w2, w3 = w0/norm, w1/norm, w2/norm, w3/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1 - 2*(w2**2 + w3**2), 2*(w1*w2 - w0*w3), 2*(w1*w3 + w0*w2)],
        [2*(w1*w2 + w0*w3), 1 - 2*(w1**2 + w3**2), 2*(w2*w3 - w0*w1)],
        [2*(w1*w3 - w0*w2), 2*(w2*w3 + w0*w1), 1 - 2*(w1**2 + w2**2)]
    ])
    
    return R


def load_probe_trajectory(json_path):
    """从JSON加载探头位置和旋转信息"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    positions = []
    rotations = []
    
    # 获取扫描参数
    infos = data['infos']
    width_mm = infos['scan_dims_mm']['width']  # Z方向（宽度）
    depth_mm = infos['scan_dims_mm']['depth']  # Y方向（深度/厚度）
    
    # 读取每个slice的位置和旋转
    for key in sorted(data.keys()):
        if key != 'infos':
            slice_data = data[key]
            pos = np.array([slice_data['x'], slice_data['y'], slice_data['z']])
            quat = np.array([slice_data['w0'], slice_data['w1'], slice_data['w2'], slice_data['w3']])
            
            positions.append(pos)
            rotations.append(quat)
    
    return np.array(positions), np.array(rotations), width_mm, depth_mm


def visualize_probe_trajectory_3d(json_path, num_slices=None, arrow_scale=2, 
                                   rect_scale=1.0, save_path=None):
    """
    3D可视化探头轨迹
    
    Args:
        json_path: infos.json 文件路径
        num_slices: 显示的切片数量（None表示全部）
        arrow_scale: 箭头长度缩放因子
        rect_scale: 矩形大小缩放因子
        save_path: 保存图像的路径
    """
    positions, rotations, width_mm, depth_mm = load_probe_trajectory(json_path)
    
    if num_slices is not None:
        positions = positions[::max(1, len(positions)//num_slices)][:num_slices]
        rotations = rotations[::max(1, len(rotations)//num_slices)][:num_slices]
    
    print(f"✓ 加载了 {len(positions)} 个探头位置")
    print(f"  Slice宽度(Z方向): {width_mm:.2f} mm")
    print(f"  Slice厚度(Y方向): {depth_mm:.2f} mm")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.6, label='探头轨迹')
    
    # 绘制起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              color='green', s=100, marker='o', label='起点', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              color='red', s=100, marker='s', label='终点', zorder=5)
    
    # 为每个位置绘制坐标系
    arrow_length = arrow_scale
    
    # 均匀采样：显示最多20个探头位置
    num_to_show = min(20, len(positions))
    indices = np.linspace(0, len(positions)-1, num_to_show, dtype=int)
    
    for idx in indices:
        pos = positions[idx]
        quat = rotations[idx]
        R = quaternion_to_rotation_matrix(quat)
        
        # 坐标轴方向
        x_axis = R[:, 0]  # 探头方向
        y_axis = R[:, 1]  # Y方向（平面外）
        z_axis = R[:, 2]  # Z方向（宽度）
        
        # 绘制探头方向箭头（X轴，红色）
        arrow = Arrow3D(pos[0], pos[1], pos[2],
                       arrow_length*x_axis[0], arrow_length*x_axis[1], arrow_length*x_axis[2],
                       mutation_scale=20, lw=2, arrowstyle="-|>", color="red", alpha=0.7)
        ax.add_artist(arrow)
        
        # 绘制矩形表示probe面
        # 矩形四个角：基于Y和Z轴
        # 调整长宽比：Z方向（宽）是Y方向（厚）的1.5倍
        rect_size_y = depth_mm * rect_scale / 2 * 0.5  # Y方向缩小
        rect_size_z = width_mm * rect_scale / 2 * 1.5  # Z方向放大
        
        corners = np.array([
            pos,  # 中心
            pos + rect_size_y * y_axis + rect_size_z * z_axis,
            pos + rect_size_y * y_axis - rect_size_z * z_axis,
            pos - rect_size_y * y_axis - rect_size_z * z_axis,
            pos - rect_size_y * y_axis + rect_size_z * z_axis,
            pos + rect_size_y * y_axis + rect_size_z * z_axis,  # 闭合
        ])
        
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 
               'b-', linewidth=1.5, alpha=0.5)
        
        # 填充矩形
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        rect_corners = np.array([
            pos + rect_size_y * y_axis + rect_size_z * z_axis,
            pos + rect_size_y * y_axis - rect_size_z * z_axis,
            pos - rect_size_y * y_axis - rect_size_z * z_axis,
            pos - rect_size_y * y_axis + rect_size_z * z_axis,
        ])
        poly = Poly3DCollection([rect_corners], alpha=0.2, facecolor='cyan', edgecolor='blue')
        ax.add_collection3d(poly)
    
    # 设置标签和标题
    ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=10, fontweight='bold')
    ax.set_title('Probe Trajectory\nRed arrow=X(probe direction)  Cyan rect=Z x Y plane', 
                fontsize=12, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 已保存到: {save_path}")
    
    plt.show()


def visualize_probe_trajectory_2d(json_path, num_slices=None, save_path=None):
    """
    2D俯视图和侧视图
    """
    positions, rotations, width_mm, depth_mm = load_probe_trajectory(json_path)
    
    if num_slices is not None:
        step = max(1, len(positions) // num_slices)
        positions = positions[::step]
        rotations = rotations[::step]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # XY平面图（俯视图）
    ax = axes[0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
    
    # 均匀采样显示箭头
    num_to_show = min(15, len(positions))
    indices = np.linspace(0, len(positions)-1, num_to_show, dtype=int)
    
    for idx in indices:
        pos = positions[idx]
        quat = rotations[idx]
        R = quaternion_to_rotation_matrix(quat)
        x_axis = R[:, 0]
        
        # 绘制小箭头表示方向
        ax.arrow(pos[0], pos[1], x_axis[0]*2, x_axis[1]*2,
                head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)
    
    ax.set_xlabel('X (mm)', fontweight='bold')
    ax.set_ylabel('Y (mm)', fontweight='bold')
    ax.set_title('Top View (XY plane)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # XZ平面图（侧视图）
    ax = axes[1]
    ax.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 2], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2], color='red', s=100, marker='s', label='End', zorder=5)
    
    # 均匀采样显示箭头
    num_to_show = min(15, len(positions))
    indices = np.linspace(0, len(positions)-1, num_to_show, dtype=int)
    
    for idx in indices:
        pos = positions[idx]
        quat = rotations[idx]
        R = quaternion_to_rotation_matrix(quat)
        x_axis = R[:, 0]
        
        ax.arrow(pos[0], pos[2], x_axis[0]*2, x_axis[2]*2,
                head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)
    
    ax.set_xlabel('X (mm)', fontweight='bold')
    ax.set_ylabel('Z (mm)', fontweight='bold')
    ax.set_title('Side View (XZ plane)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存到: {save_path}")
    
    plt.show()


def main():
    # 配置
    data_folder = "./data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original"
    json_path = os.path.join(data_folder, "infos.json")
    
    if not os.path.exists(json_path):
        print(f"✗ 错误: 找不到文件 {json_path}")
        return
    
    print("=" * 70)
    print("超声探头轨迹可视化")
    print("=" * 70)
    
    # 3D可视化
    print("\n生成3D轨迹图...")
    visualize_probe_trajectory_3d(
        json_path, 
        num_slices=50,  # 显示所有位置（密度取决于step）
        arrow_scale=4,  # 增加箭头长度
        rect_scale=0.12,  # 减小矩形大小
        save_path="./probe_trajectory_3d.png"
    )
    
    # 2D可视化
    print("\n生成2D投影图...")
    visualize_probe_trajectory_2d(
        json_path,
        num_slices=None,  # 显示全部
        save_path="./probe_trajectory_2d.png"
    )
    
    print("\n" + "=" * 70)
    print("✓ 可视化完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
