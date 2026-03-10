"""
快速性能测试脚本 - 找出 dataset 加载慢的原因
"""

import os
# 解决 OpenMP 多库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, os.path.dirname(__file__))

from dataset_profiler import DatasetWithProfiling

# 测试路径 - 改为你的实际路径
test_folders = [
    # "./data/simu_56",
    "./data/cerebral_data/Pre_traitement_echo_v2/Recalage/Patient0/us_recal_original",
]

for folder in test_folders:
    if os.path.exists(folder):
        print(f"\n{'='*70}")
        print(f"测试: {folder}")
        print(f"{'='*70}")
        
        try:
            dataset = DatasetWithProfiling(folder, profile=True)
            print(f"\n✓ 成功加载，共 {len(dataset.slices)} 张图像")
        except FileNotFoundError as e:
            print(f"✗ 文件不存在: {e}")
        except Exception as e:
            print(f"✗ 错误: {e}")
        break
    else:
        print(f"⚠ 跳过 (不存在): {folder}")

print("\n" + "="*70)
print("性能分析完成")
print("="*70)
