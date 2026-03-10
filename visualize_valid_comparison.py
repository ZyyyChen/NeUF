"""
可视化比较验证集的渲染结果和原始图像
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset_1 import Dataset
from slice_renderer import SliceRenderer
from nerf_network import NeRF
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path, model_config):
    """加载训练好的模型"""
    model = NeRF(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def visualize_comparison(dataset, model, slice_renderer, num_slices=4, save_path=None):
    """
    可视化对比验证集的原始图像和渲染结果
    
    Args:
        dataset: 数据集对象
        model: 训练好的模型
        slice_renderer: 切片渲染器
        num_slices: 要可视化的切片数量
        save_path: 保存图像的路径（可选）
    """
    num_slices = min(num_slices, len(dataset.slices_valid))
    
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_slices):
            # 获取原始图像
            original = dataset.get_slice_valid_pixels(i).cpu().numpy().reshape(
                dataset.px_height, dataset.px_width
            )
            
            # 渲染图像
            rendered = slice_renderer.render_slice_from_dataset_valid(
                model, i, reshaped=True
            ).cpu().numpy()
            
            # 计算差异
            diff = np.abs(original - rendered)
            
            # 可视化
            axes[i, 0].imshow(original, cmap='gray')
            axes[i, 0].set_title(f'验证集 #{i} - 原始图像\n(对应数据集后部图像)')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(rendered, cmap='gray')
            axes[i, 1].set_title(f'验证集 #{i} - 渲染结果')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(diff, cmap='hot')
            axes[i, 2].set_title(f'差异图\nMAE: {np.mean(diff):.4f}')
            axes[i, 2].axis('off')
            
            # 打印统计信息
            print(f"\n验证集 #{i} 统计:")
            print(f"  原始图像范围: [{original.min():.3f}, {original.max():.3f}]")
            print(f"  渲染结果范围: [{rendered.min():.3f}, {rendered.max():.3f}]")
            print(f"  MAE (平均绝对误差): {np.mean(diff):.4f}")
            print(f"  RMSE (均方根误差): {np.sqrt(np.mean(diff**2)):.4f}")
            print(f"  最大误差: {np.max(diff):.4f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 保存对比图到: {save_path}")
    
    plt.show()


def main():
    # ===== 配置区域 =====
    # 数据集路径
    data_folder = "./data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original"
    
    # 模型检查点路径 - 修改为你的实际路径
    checkpoint_path = "./logs/25-02-2026/HASH_Patient0_0/checkpoints/checkpoint_3000.pt"
    
    # 模型配置（需要与训练时一致）
    model_config = {
        'D': 4,
        'W': 128,
        'use_encoding': True,
        'encoding_type': 'HASH',
        'nb_levels': 16,
        'nb_feat_per_level': 2,
        'log2_hashmap_size': 19,
        'base_resolution': 16,
        'max_resolution': 2048,
    }
    
    # 要可视化的验证集数量
    num_slices_to_show = 4
    
    # 保存路径（可选）
    save_path = "./validation_comparison.png"
    
    # ===== 加载数据和模型 =====
    print("=" * 70)
    print("加载数据集...")
    dataset = Dataset(data_folder, nb_valid=4, seed=17081998)
    print(f"✓ 数据集加载完成")
    print(f"  训练集: {len(dataset.slices)} 张")
    print(f"  验证集: {len(dataset.slices_valid)} 张")
    print(f"  图像尺寸: {dataset.px_width}x{dataset.px_height} px")
    print(f"  物理尺寸: {dataset.width:.2f}x{dataset.height:.2f} mm")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ 错误: 找不到模型检查点: {checkpoint_path}")
        print("请修改 checkpoint_path 变量为你的实际模型路径")
        return
    
    print(f"\n加载模型: {checkpoint_path}")
    model = load_model(checkpoint_path, model_config)
    print("✓ 模型加载完成")
    
    print("\n初始化渲染器...")
    slice_renderer = SliceRenderer(dataset=dataset)
    print("✓ 渲染器初始化完成")
    
    # ===== 可视化对比 =====
    print("\n" + "=" * 70)
    print("开始生成对比可视化...")
    print("=" * 70)
    
    visualize_comparison(
        dataset, 
        model, 
        slice_renderer, 
        num_slices=num_slices_to_show,
        save_path=save_path
    )
    
    print("\n" + "=" * 70)
    print("可视化完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
