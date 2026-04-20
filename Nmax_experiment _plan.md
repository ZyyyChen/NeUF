# Hash Grid N_max 实验计划

## 当前配置
- `N_min = 16`, `N_max = 512`, `L = 16 levels`
- 各级分辨率: 16, 20, 25, 32, 40, 50, 63, 79, 99, 124, 156, 195, 245, 307, 385, 512

## 实验目标
找到"speckle 明显减少 + 边缘仍然 sharp"的甜点

---

## 实验 1：逐步降 N_max（保持 L=16）

跑以下 5 组，其他超参不变：

| 实验 | N_max | 效果预期 |
|------|-------|----------|
| A (baseline) | 512 | 当前结果，≈KNN，speckle 全保留 |
| B | 256 | 最高频砍一半，轻微去 speckle |
| C | 128 | 中等去 speckle，边缘可能开始轻微模糊 |
| D | 64  | 较强去 speckle，观察边缘是否还在 |
| E | 32  | 过度平滑 baseline，确认下界 |

### 代码改动（以 tiny-cuda-nn 为例）
```python
# 只需改 config 里的一个数
encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": (N_MAX / 16) ** (1.0 / 15),  # <-- 改 N_MAX
    # 或者直接算:
    # N_max=512 → per_level_scale=1.254
    # N_max=256 → per_level_scale=1.197
    # N_max=128 → per_level_scale=1.142
    # N_max=64  → per_level_scale=1.090
    # N_max=32  → per_level_scale=1.047
}
```

### 评估指标（每组都记录）
- [ ] 选 3-5 张有明确组织边缘的切片，目视对比
- [ ] 计算 edge 区域的梯度幅值（越大越 sharp）
- [ ] 计算均匀组织区域的 intensity 方差（越小 speckle 越少）
- [ ] 如果有 ground truth：PSNR, SSIM
- [ ] 截图存档，做成对比 montage

---

## 实验 2：保持 N_max=512，减少 level 数（可选对照）

保持最高和最低分辨率不变，减少中间级数，看信息是否冗余：

| 实验 | L (levels) | per_level_scale |
|------|-----------|-----------------|
| F | 12 | 1.346 |
| G | 8  | 1.640 |

这组实验回答的问题是：speckle 是被"太多中间级"过度拟合了，还是被"最高几级"拟合了。

---

## 实验 3：保留全部 levels，但 mask 高频级（更精细控制）

不改 hash grid 配置，而是在 forward 时把高频级 feature 乘一个系数：

```python
def forward_with_level_mask(hash_grid, coords, active_levels=16):
    """
    hash_grid 输出 shape: [N, L * F]
    L = n_levels, F = n_features_per_level
    """
    all_features = hash_grid(coords)  # [N, L*F]
    F_per_level = 2  # n_features_per_level
    
    # 把高于 active_levels 的 feature 置零
    mask = torch.ones(16 * F_per_level, device=all_features.device)
    mask[active_levels * F_per_level:] = 0.0
    
    return all_features * mask
```

然后扫 `active_levels` 从 16 到 8：

| active_levels | 最高有效分辨率 |
|--------------|---------------|
| 16 | 512 (baseline) |
| 14 | 307 |
| 12 | 195 |
| 10 | 124 |
| 8  | 79 |

**优点**：不需要重新训练 hash grid，可以在同一个训好的模型上快速扫描。
**注意**：如果是训练时 mask，效果更好（网络会适应）。推理时 mask 只是近似。

---

## 判断标准

### 找甜点的方法
画一条曲线：
- X 轴: N_max (或 active_levels)
- Y 轴: edge sharpness (梯度幅值) 和 speckle level (均匀区方差)

两条曲线的交叉区域就是甜点。通常 speckle 先快速下降，边缘后缓慢下降，中间有一个 plateau。

### Edge sharpness 简易计算
```python
import numpy as np
from scipy.ndimage import sobel

def edge_sharpness(img, edge_mask):
    """edge_mask: 手动标注的边缘区域 binary mask"""
    gx = sobel(img, axis=0)
    gy = sobel(img, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_mag[edge_mask > 0])

def speckle_level(img, uniform_mask):
    """uniform_mask: 手动标注的均匀组织区域"""
    return np.std(img[uniform_mask > 0]) / np.mean(img[uniform_mask > 0])
```

---

## 下一步
拿到甜点之后，在该 N_max 配置上加：
1. Elevational MSP consistency loss（方向 3）
2. 各向异性 TV regularization（方向 4）
3. Structure/speckle 双场分离（方向 6）
