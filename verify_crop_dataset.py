"""
验证crop后的数据集是否正确

这个脚本帮助你验证:
1. 像素物理尺寸比例是否保持一致
2. 点云坐标范围是否合理
3. 可视化点云投影
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def verify_dataset(dataset):
    """验证数据集的crop是否正确应用"""
    
    print("\n" + "="*60)
    print("数据集验证报告")
    print("="*60)
    
    # 1. 检查像素物理尺寸比例
    print("\n【1】像素物理尺寸比例")
    pixel_size_w = dataset.width / dataset.px_width
    pixel_size_h = dataset.height / dataset.px_height
    
    print(f"  宽度方向: {pixel_size_w:.6f} mm/px")
    print(f"  高度方向: {pixel_size_h:.6f} mm/px")
    
    ratio_diff = abs(pixel_size_w - pixel_size_h) / pixel_size_w * 100
    if ratio_diff < 1:
        print(f"  ✓ 比例一致 (差异: {ratio_diff:.2f}%)")
    else:
        print(f"  ✗ 比例不一致 (差异: {ratio_diff:.2f}%)")
        print(f"    警告: 宽高方向的像素物理尺寸应该相同!")
    
    # 如果有原始尺寸,检查是否保持一致
    if dataset.orig_px_width > 0 and dataset.orig_px_height > 0:
        print("\n【2】与原始图像的比例对比")
        if hasattr(dataset, 'roi_2d') and dataset.roi_2d:
            # 假设从infos.json读取的是原始物理尺寸
            # 计算原始的像素物理尺寸
            orig_pixel_size_w = dataset.width / dataset.roi_2d['width']
            orig_pixel_size_h = dataset.height / dataset.roi_2d['height']
            
            print(f"  原始像素尺寸: {orig_pixel_size_w:.6f} x {orig_pixel_size_h:.6f} mm/px")
            print(f"  当前像素尺寸: {pixel_size_w:.6f} x {pixel_size_h:.6f} mm/px")
            
            if abs(orig_pixel_size_w - pixel_size_w) < 0.0001:
                print(f"  ✓ 像素物理尺寸保持一致")
            else:
                print(f"  ✗ 像素物理尺寸改变了!")
        else:
            print(f"  未应用ROI裁剪")
    
    # 3. 点云范围检查
    print("\n【3】点云坐标范围")
    print(f"  X轴: {dataset.point_min[0]:.2f} ~ {dataset.point_max[0]:.2f} mm (范围: {dataset.point_max[0]-dataset.point_min[0]:.2f} mm)")
    print(f"  Y轴: {dataset.point_min[1]:.2f} ~ {dataset.point_max[1]:.2f} mm (范围: {dataset.point_max[1]-dataset.point_min[1]:.2f} mm)")
    print(f"  Z轴: {dataset.point_min[2]:.2f} ~ {dataset.point_max[2]:.2f} mm (范围: {dataset.point_max[2]-dataset.point_min[2]:.2f} mm)")
    
    # 4. 数据集统计
    print("\n【4】数据集统计")
    print(f"  训练slices: {len(dataset.slices)}")
    print(f"  验证slices: {len(dataset.slices_valid)}")
    print(f"  总像素数: {dataset.pixels.shape[0]:,}")
    print(f"  总点数: {dataset.points.shape[0]:,}")
    
    # 5. 检查一个slice的数据
    print("\n【5】单个slice检查")
    if len(dataset.slices) > 0:
        slice_0_pixels = dataset.get_slice_pixels(0)
        slice_0_points = dataset.get_slice_points(0)
        
        print(f"  Slice 0:")
        print(f"    像素数: {slice_0_pixels.shape[0]}")
        print(f"    点数: {slice_0_points.shape[0]}")
        print(f"    期望: {dataset.px_width * dataset.px_height}")
        
        if slice_0_pixels.shape[0] == dataset.px_width * dataset.px_height:
            print(f"    ✓ 像素数正确")
        else:
            print(f"    ✗ 像素数不匹配!")
    
    print("\n" + "="*60)
    

def visualize_slice(dataset, slice_idx=0, save_path=None):
    """可视化一个slice的点云和图像"""
    
    if slice_idx >= len(dataset.slices):
        print(f"错误: slice {slice_idx} 不存在 (总共{len(dataset.slices)}个)")
        return
    
    # 获取数据
    pixels = dataset.get_slice_pixels(slice_idx).cpu().numpy()
    points = dataset.get_slice_points(slice_idx).cpu().numpy().squeeze()
    
    # 重塑图像
    image = pixels.reshape(dataset.px_height, dataset.px_width)
    
    # 创建可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 原始图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Slice {slice_idx} - Image\n{dataset.px_width}x{dataset.px_height} px')
    ax1.set_xlabel('Width (px)')
    ax1.set_ylabel('Height (px)')
    
    # 2. XZ投影
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(points[:, 0], points[:, 2], c=pixels.flatten(), 
                         cmap='gray', s=1, vmin=0, vmax=255)
    ax2.set_title(f'Point Cloud - XZ Projection\n{dataset.width:.1f}x{dataset.height:.1f} mm')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Intensity')
    
    # 3. XY投影
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(points[:, 0], points[:, 1], c=pixels.flatten(), 
                         cmap='gray', s=1, vmin=0, vmax=255)
    ax3.set_title('Point Cloud - XY Projection')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Intensity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
    else:
        plt.show()


def compare_datasets(dataset_no_crop, dataset_with_crop):
    """对比裁剪前后的数据集"""
    
    print("\n" + "="*60)
    print("裁剪前后对比")
    print("="*60)
    
    print("\n像素尺寸:")
    print(f"  裁剪前: {dataset_no_crop.px_width} x {dataset_no_crop.px_height} px")
    print(f"  裁剪后: {dataset_with_crop.px_width} x {dataset_with_crop.px_height} px")
    
    print("\n物理尺寸:")
    print(f"  裁剪前: {dataset_no_crop.width:.2f} x {dataset_no_crop.height:.2f} mm")
    print(f"  裁剪后: {dataset_with_crop.width:.2f} x {dataset_with_crop.height:.2f} mm")
    
    print("\n像素物理尺寸:")
    ps_w_before = dataset_no_crop.width / dataset_no_crop.px_width
    ps_h_before = dataset_no_crop.height / dataset_no_crop.px_height
    ps_w_after = dataset_with_crop.width / dataset_with_crop.px_width
    ps_h_after = dataset_with_crop.height / dataset_with_crop.px_height
    
    print(f"  裁剪前: {ps_w_before:.6f} x {ps_h_before:.6f} mm/px")
    print(f"  裁剪后: {ps_w_after:.6f} x {ps_h_after:.6f} mm/px")
    
    if abs(ps_w_before - ps_w_after) < 0.0001 and abs(ps_h_before - ps_h_after) < 0.0001:
        print(f"  ✓ 像素物理尺寸保持一致!")
    else:
        print(f"  ✗ 警告: 像素物理尺寸改变了!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证crop后的数据集")
    parser.add_argument('--dataset-folder', type=str, required=True,
                       help='数据集文件夹路径')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化第一个slice')
    parser.add_argument('--slice-idx', type=int, default=0,
                       help='要可视化的slice索引')
    parser.add_argument('--save-fig', type=str,
                       help='保存可视化图像的路径')
    
    args = parser.parse_args()
    
    # 导入数据集
    print(f"加载数据集: {args.dataset_folder}")
    
    try:
        # 尝试使用修复版本
        from dataset import Dataset
        print("使用 dataset_fixed.py")
    except ImportError:
        # 使用原始版本
        from dataset_1 import Dataset
        print("使用 dataset.py")
    
    dataset = Dataset(args.dataset_folder)
    
    # 验证
    verify_dataset(dataset)
    
    # 可视化
    if args.visualize:
        print(f"\n可视化 slice {args.slice_idx}...")
        visualize_slice(dataset, args.slice_idx, args.save_fig)


# 使用示例:
# python verify_crop_dataset.py --dataset-folder ./data/simu_56 --visualize
# python verify_crop_dataset.py --dataset-folder ./data/simu_56 --visualize --save-fig ./crop_check.png
