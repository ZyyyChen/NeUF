# Dual-Frequency PE 修改计划

## 总览

在现有 NeUF 代码库上叠加双频率编码分支，**不破坏现有 HASH / FREQ / NONE 三种模式**，
通过新增 `--encoding DUAL_HASH` 和 `--encoding DUAL_FREQ` 两个选项激活。

涉及文件：
- **新建** `dual_freq_encoder.py`
- **修改** `nerf_network.py`
- **修改** `main.py`

切换参数说明：

| `--encoding`   | PE 类型              | 说明                        |
|----------------|----------------------|-----------------------------|
| `Hash`         | 原始 HashEncoder     | 不变                        |
| `Freq`         | 原始 BaseEncoder     | 不变                        |
| `None`         | 无编码               | 不变                        |
| `DUAL_HASH`    | 双 HashEncoder 分支  | 粗粒度Hash + 细粒度Hash      |
| `DUAL_FREQ`    | 双随机傅里叶特征分支 | 低σ RFF + 高σ RFF            |

---

## 文件 1：新建 `dual_freq_encoder.py`

完整内容如下，包含两种 PE 实现及统一接口。

```python
# dual_freq_encoder.py
"""
双频率编码器模块
提供两种实现，通过 pe_type 参数切换：
  - 'hash'  : 双粗细粒度 HashEncoder
  - 'fourier': 双低高σ 随机傅里叶特征 (RFF)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_encoder import HashEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────
# 1. 随机傅里叶特征编码器
# ─────────────────────────────────────────────
class RFFEncoder(nn.Module):
    """Random Fourier Feature encoder，控制频率范围通过 sigma。"""

    def __init__(self, sigma: float, n_freq: int = 64, input_dim: int = 3):
        super().__init__()
        B = torch.randn(input_dim, n_freq) * sigma
        self.register_buffer('B', B)
        self.out_dim = n_freq * 2   # sin + cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B           # (N, n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ─────────────────────────────────────────────
# 2. 空间自适应门控网络
# ─────────────────────────────────────────────
class GateNet(nn.Module):
    """
    输入：低频编码特征
    输出：每个空间点的高频门控权重 ∈ (0, 1)
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32),    nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_low: torch.Tensor) -> torch.Tensor:
        return self.net(feat_low)   # (N, 1)


# ─────────────────────────────────────────────
# 3. 统一双频编码器接口
# ─────────────────────────────────────────────
class DualFreqEncoder(nn.Module):
    """
    统一接口，pe_type 控制底层实现：
      pe_type='hash'    → 双 HashEncoder
      pe_type='fourier' → 双 RFFEncoder
    """

    def __init__(
        self,
        pe_type: str = 'hash',
        bounding_box=None,
        # Hash 参数
        n_levels_low: int = 8,
        n_levels_high: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution_low: int = 16,
        finest_resolution_low: int = 64,
        base_resolution_high: int = 64,
        finest_resolution_high: int = 512,
        # Fourier 参数
        sigma_low: float = 1.0,
        sigma_high: float = 20.0,
        n_freq: int = 64,
        # 门控参数
        use_gate: bool = True,
        gate_hidden: int = 64,
        # 高频渐进激活
        hf_activate_ratio: float = 0.6,   # 训练进度超过此值开始激活高频
        hf_max_weight: float = 1.0,       # 高频最大融合权重
    ):
        super().__init__()

        self.pe_type = pe_type.lower()
        self.use_gate = use_gate
        self.hf_activate_ratio = hf_activate_ratio
        self.hf_max_weight = hf_max_weight

        if self.pe_type == 'hash':
            assert bounding_box is not None, "DUAL_HASH 需要 bounding_box"
            self.enc_low = HashEncoder(
                bounding_box=bounding_box,
                n_levels=n_levels_low,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution_low,
                finest_resolution=finest_resolution_low,
            )
            self.enc_high = HashEncoder(
                bounding_box=bounding_box,
                n_levels=n_levels_high,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution_high,
                finest_resolution=finest_resolution_high,
            )

        elif self.pe_type == 'fourier':
            self.enc_low  = RFFEncoder(sigma=sigma_low,  n_freq=n_freq)
            self.enc_high = RFFEncoder(sigma=sigma_high, n_freq=n_freq)

        else:
            raise ValueError(f"未知 pe_type: {pe_type}，支持 'hash' 或 'fourier'")

        self.out_dim_low  = self.enc_low.out_dim
        self.out_dim_high = self.enc_high.out_dim
        # 对外暴露的总维度（拼接，供 NeRF 初始化 MLP 用）
        self.out_dim = self.out_dim_low + self.out_dim_high

        # 门控网络：以低频特征为输入
        if use_gate:
            self.gate_net = GateNet(self.out_dim_low, hidden=gate_hidden)
        else:
            self.gate_net = None

        self.to(DEVICE)

    # ------------------------------------------------------------------
    def global_hf_weight(self, progress: float) -> torch.Tensor:
        """
        sigmoid 渐进曲线：
          progress < hf_activate_ratio  → 接近 0
          progress > hf_activate_ratio  → 快速趋向 hf_max_weight
        """
        k = 10.0
        raw = torch.sigmoid(torch.tensor(k * (progress - self.hf_activate_ratio)))
        return self.hf_max_weight * raw

    # ------------------------------------------------------------------
    def forward_decomposed(
        self, x: torch.Tensor, progress: float = 1.0
    ) -> dict:
        """
        返回字典，供损失计算和可视化使用：
          feat_low   : 低频特征
          feat_high  : 高频特征
          gate       : 空间门控权重 (N,1)，use_gate=False 时为全1
          hf_weight  : 全局进度权重（标量）
          combined   : 最终合并特征 (N, out_dim_low + out_dim_high)
        """
        feat_low  = self.enc_low(x)
        feat_high = self.enc_high(x)

        gate = self.gate_net(feat_low) if self.use_gate else torch.ones(
            x.shape[0], 1, device=x.device)

        hf_weight = self.global_hf_weight(progress)

        # 加权高频特征
        weighted_high = hf_weight * gate * feat_high

        combined = torch.cat([feat_low, weighted_high], dim=-1)

        return {
            'feat_low':  feat_low,
            'feat_high': feat_high,
            'gate':      gate,
            'hf_weight': hf_weight,
            'combined':  combined,
        }

    def forward(self, x: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return self.forward_decomposed(x, progress)['combined']

    def parameters_low(self):
        params = list(self.enc_low.parameters())
        return params

    def parameters_high(self):
        params = list(self.enc_high.parameters())
        if self.gate_net:
            params += list(self.gate_net.parameters())
        return params
```

---

## 文件 2：修改 `nerf_network.py`

### 2-a  import 区域（顶部添加一行）

```python
# 在 import hash_encoder 后添加：
import dual_freq_encoder
```

### 2-b  `NeRF.__init__` 新增成员变量

在 `self.encoding_initialized = False` 之后添加：

```python
self.dual_encoder: "dual_freq_encoder.DualFreqEncoder | None" = None
self.training_progress: float = 0.0   # 由训练循环更新，0.0→1.0
```

### 2-c  新增方法 `init_dual_encoding`

在 `init_base_encoding` 方法之后插入整块方法：

```python
def init_dual_encoding(
    self,
    pe_type: str = 'hash',
    bounding_box=None,
    # Hash 参数
    n_levels_low: int = 8,
    n_levels_high: int = 8,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 19,
    base_resolution_low: int = 16,
    finest_resolution_low: int = 64,
    base_resolution_high: int = 64,
    finest_resolution_high: int = 512,
    # Fourier 参数
    sigma_low: float = 1.0,
    sigma_high: float = 20.0,
    n_freq: int = 64,
    # 门控
    use_gate: bool = True,
    hf_activate_ratio: float = 0.6,
    hf_max_weight: float = 1.0,
):
    if self.encoding_initialized:
        print("encoding initialized twice"); exit(-1)

    self.dual_encoder = dual_freq_encoder.DualFreqEncoder(
        pe_type=pe_type,
        bounding_box=bounding_box,
        n_levels_low=n_levels_low,
        n_levels_high=n_levels_high,
        n_features_per_level=n_features_per_level,
        log2_hashmap_size=log2_hashmap_size,
        base_resolution_low=base_resolution_low,
        finest_resolution_low=finest_resolution_low,
        base_resolution_high=base_resolution_high,
        finest_resolution_high=finest_resolution_high,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        n_freq=n_freq,
        use_gate=use_gate,
        hf_activate_ratio=hf_activate_ratio,
        hf_max_weight=hf_max_weight,
    )

    # 统一接口：encode 仍然是 callable，query 方法不变
    self.encode    = lambda x: self.dual_encoder(x, self.training_progress)
    self.in_ch     = self.dual_encoder.out_dim
    self.out_ch    = 1
    self.skips     = [4]
    self.dir_ch    = 0
    self.use_encoding  = True
    self.use_direction = False
    self.encoding_type = f"DUAL_{pe_type.upper()}"

    # encoder_params 包含双编码器全部参数
    self.encoder_params = list(self.dual_encoder.parameters())
    self.encoding_initialized = True
```

### 2-d  修改 `_init_from_ckpt`

在 `elif(ckpt["encoding"] == "HASH"):` 块之后，添加：

```python
elif ckpt["encoding"].startswith("DUAL_"):
    pe_type = ckpt["encoding"].split("_", 1)[1].lower()  # 'hash' 或 'fourier'
    self.init_dual_encoding(
        pe_type=pe_type,
        bounding_box=ckpt.get("bounding_box"),
        n_levels_low=ckpt.get("n_levels_low", 8),
        n_levels_high=ckpt.get("n_levels_high", 8),
        n_features_per_level=ckpt.get("n_features_per_level", 2),
        log2_hashmap_size=ckpt.get("log2_hashmap_size", 19),
        base_resolution_low=ckpt.get("base_resolution_low", 16),
        finest_resolution_low=ckpt.get("finest_resolution_low", 64),
        base_resolution_high=ckpt.get("base_resolution_high", 64),
        finest_resolution_high=ckpt.get("finest_resolution_high", 512),
        sigma_low=ckpt.get("sigma_low", 1.0),
        sigma_high=ckpt.get("sigma_high", 20.0),
        n_freq=ckpt.get("n_freq", 64),
        use_gate=ckpt.get("use_gate", True),
        hf_activate_ratio=ckpt.get("hf_activate_ratio", 0.6),
        hf_max_weight=ckpt.get("hf_max_weight", 1.0),
    )
    if self.dual_encoder is not None and "dual_encoder_state" in ckpt:
        self.dual_encoder.load_state_dict(ckpt["dual_encoder_state"])
```

### 2-e  修改 `get_save_dict`

在 `elif self.encoding_type == "HASH":` 块之后添加：

```python
elif self.encoding_type.startswith("DUAL_"):
    pe_type = self.encoding_type.split("_", 1)[1].lower()
    enc = self.dual_encoder
    dic.update({
        "bounding_box":          enc.enc_low.bounding_box if hasattr(enc.enc_low, 'bounding_box') else None,
        "n_levels_low":          enc.enc_low.n_levels     if hasattr(enc.enc_low, 'n_levels') else None,
        "n_levels_high":         enc.enc_high.n_levels    if hasattr(enc.enc_high, 'n_levels') else None,
        "n_features_per_level":  enc.enc_low.n_features_per_level if hasattr(enc.enc_low, 'n_features_per_level') else None,
        "log2_hashmap_size":     enc.enc_low.log2_hashmap_size    if hasattr(enc.enc_low, 'log2_hashmap_size') else None,
        "base_resolution_low":   float(enc.enc_low.base_resolution.cpu())   if hasattr(enc.enc_low, 'base_resolution') else None,
        "finest_resolution_low": float(enc.enc_low.finest_resolution.cpu()) if hasattr(enc.enc_low, 'finest_resolution') else None,
        "base_resolution_high":  float(enc.enc_high.base_resolution.cpu())  if hasattr(enc.enc_high, 'base_resolution') else None,
        "finest_resolution_high":float(enc.enc_high.finest_resolution.cpu())if hasattr(enc.enc_high, 'finest_resolution') else None,
        "sigma_low":             getattr(enc.enc_low,  'sigma', None),
        "sigma_high":            getattr(enc.enc_high, 'sigma', None),
        "use_gate":              enc.use_gate,
        "hf_activate_ratio":     enc.hf_activate_ratio,
        "hf_max_weight":         enc.hf_max_weight,
        "dual_encoder_state":    enc.state_dict(),
    })
```

### 2-f  修改 `get_encode_name` 和 `get_rep_name`

`get_encode_name` 已经返回 `self.encoding_type`，无需改动。

---

## 文件 3：修改 `main.py`

### 3-a  新增实例变量（`NeUF.__init__` 中）

在 `self.hash_finest_resolution = ...` 之后添加：

```python
# Dual-freq 参数
self.dual_pe_type              = kwargs.get("dual_pe_type", "hash")
self.dual_n_levels_low         = int(kwargs.get("dual_n_levels_low", 8))
self.dual_n_levels_high        = int(kwargs.get("dual_n_levels_high", 8))
self.dual_finest_resolution_low  = int(kwargs.get("dual_finest_resolution_low", 64))
self.dual_finest_resolution_high = int(kwargs.get("dual_finest_resolution_high", 512))
self.dual_base_resolution_low    = int(kwargs.get("dual_base_resolution_low", 16))
self.dual_base_resolution_high   = int(kwargs.get("dual_base_resolution_high", 64))
self.dual_sigma_low            = float(kwargs.get("dual_sigma_low", 1.0))
self.dual_sigma_high           = float(kwargs.get("dual_sigma_high", 20.0))
self.dual_n_freq               = int(kwargs.get("dual_n_freq", 64))
self.dual_use_gate             = bool(kwargs.get("dual_use_gate", True))
self.dual_hf_activate_ratio    = float(kwargs.get("dual_hf_activate_ratio", 0.6))
self.dual_hf_max_weight        = float(kwargs.get("dual_hf_max_weight", 1.0))
self.dual_sparsity_weight      = float(kwargs.get("dual_sparsity_weight", 0.01))
self.dual_gate_weight          = float(kwargs.get("dual_gate_weight", 0.5))
```

### 3-b  修改 `_initialize_model_encoding`

在 `if encoding_name == "none":` 块之后，`raise ValueError` 之前插入：

```python
if encoding_name in ("dual_hash", "dual_freq"):
    pe_type = "hash" if encoding_name == "dual_hash" else "fourier"
    self.nerf.init_dual_encoding(
        pe_type=pe_type,
        bounding_box=self.dataset.get_bounding_box(),
        n_levels_low=self.dual_n_levels_low,
        n_levels_high=self.dual_n_levels_high,
        n_features_per_level=self.hash_n_features_per_level,
        log2_hashmap_size=self.hash_log2_hashmap_size,
        base_resolution_low=self.dual_base_resolution_low,
        finest_resolution_low=self.dual_finest_resolution_low,
        base_resolution_high=self.dual_base_resolution_high,
        finest_resolution_high=self.dual_finest_resolution_high,
        sigma_low=self.dual_sigma_low,
        sigma_high=self.dual_sigma_high,
        n_freq=self.dual_n_freq,
        use_gate=self.dual_use_gate,
        hf_activate_ratio=self.dual_hf_activate_ratio,
        hf_max_weight=self.dual_hf_max_weight,
    )
    self.nerf.init_model(8, 256)
    return
```

### 3-c  在训练循环中更新 progress

在 `run` 方法的 `for iteration in progress_bar:` 循环体**顶部**添加：

```python
# 双频模式：更新训练进度，控制高频渐进激活
if self.nerf.dual_encoder is not None:
    self.nerf.training_progress = iteration / max(1, self.N_iters)
```

### 3-d  修改 `_compute_training_loss`，添加双频专属损失

在方法尾部 `return loss, components` 之前插入：

```python
# ── 双频模式追加损失 ──────────────────────────────────
if self.nerf.dual_encoder is not None:
    progress = self.nerf.training_progress

    # 重新前向，获取分解结果（仅在 Random/Patch 模式有 _current_points）
    if self._current_points is not None:
        with torch.no_grad():
            decomp = self.nerf.dual_encoder.forward_decomposed(
                self._current_points.reshape(-1, 3), progress
            )
        feat_high = decomp['feat_high']
        gate      = decomp['gate']

        # 1. 高频稀疏约束（鼓励高频只在边界处激活）
        sparsity = torch.mean(torch.abs(feat_high))
        components['dual_sparsity'] = sparsity
        loss = loss + self.dual_sparsity_weight * sparsity

        # 2. 门控边界引导（梯度图软监督），仅在 Slice 训练模式下可用
        if self.training_mode == "Slice" and hasattr(self, '_current_target_slice'):
            gate_loss = self._compute_gate_boundary_loss(
                gate, self._current_target_slice
            )
            gate_w = self.dual_gate_weight * progress  # 后期加强
            components['dual_gate'] = gate_loss
            loss = loss + gate_w * gate_loss
```

### 3-e  新增辅助方法 `_compute_gate_boundary_loss`

在 `_compute_spatial_smoothness` 之后添加：

```python
def _compute_gate_boundary_loss(
    self,
    gate: torch.Tensor,         # (N, 1)
    target_slice: torch.Tensor, # (H*W,) or (H, W)
) -> torch.Tensor:
    """
    用 Scharr 梯度幅值图作为门控的软目标：
    梯度大的地方期望 gate≈1，均匀区域期望 gate≈0
    """
    H, W = self.dataset.px_height, self.dataset.px_width
    t = target_slice.reshape(1, 1, H, W).float()

    grad_x = F.conv2d(t, self.scharr_x, padding=1)
    grad_y = F.conv2d(t, self.scharr_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)   # (1,1,H,W)

    # 归一化到 [0, 1]
    g_max = grad_mag.max()
    if g_max > 1e-8:
        grad_mag = grad_mag / g_max

    # 展平对齐 gate
    target_gate = grad_mag.reshape(-1, 1)[:gate.shape[0]]

    return F.mse_loss(gate, target_gate.detach())
```

### 3-f  修改 `_run_validation`，输出双频分解预览

在 `self._save_preview_images(iteration, rendered)` 之后添加：

```python
# 双频分解可视化
if self.nerf.dual_encoder is not None:
    self._save_dual_freq_preview(iteration)
```

新增方法：

```python
def _save_dual_freq_preview(self, iteration: int) -> None:
    """保存低频/高频/门控/合并四张图用于调试。"""
    if not self.dataset.slices_valid:
        return
    idx = 0
    with torch.no_grad():
        progress = self.nerf.training_progress
        pts = self.dataset.get_slice_valid_points(idx).reshape(-1, 3)
        decomp = self.nerf.dual_encoder.forward_decomposed(pts, progress)

    H, W = self.dataset.px_height, self.dataset.px_width
    def to_img(t):
        arr = t.reshape(H, W).cpu().float().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    # 低频均值图 / 高频均值图 / 门控图
    low_img  = to_img(decomp['feat_low'].mean(dim=-1))
    high_img = to_img(decomp['feat_high'].mean(dim=-1))
    gate_img = to_img(decomp['gate'].squeeze(-1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(low_img,  cmap='gray');    axes[0].set_title('低频特征均值')
    axes[1].imshow(high_img, cmap='hot');     axes[1].set_title('高频特征均值')
    axes[2].imshow(gate_img, cmap='hot');     axes[2].set_title(f'门控 (progress={progress:.2f})')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(self.run_paths.image_dir / f'dual_freq_{iteration:06d}.png', dpi=100)
    plt.close()
```

### 3-g  新增 CLI 参数（`parse_args` 函数）

在 `hash_group` 之后添加新 group：

```python
dual_group = parser.add_argument_group("Dual-frequency PE parameters")
dual_group.add_argument("--dual-pe-type",
    dest="dual_pe_type", default="hash", choices=["hash", "fourier"],
    help="双频 PE 底层实现：hash 或 fourier")
dual_group.add_argument("--dual-n-levels-low",
    dest="dual_n_levels_low", type=int, default=8)
dual_group.add_argument("--dual-n-levels-high",
    dest="dual_n_levels_high", type=int, default=8)
dual_group.add_argument("--dual-finest-resolution-low",
    dest="dual_finest_resolution_low", type=int, default=64)
dual_group.add_argument("--dual-finest-resolution-high",
    dest="dual_finest_resolution_high", type=int, default=512)
dual_group.add_argument("--dual-base-resolution-low",
    dest="dual_base_resolution_low", type=int, default=16)
dual_group.add_argument("--dual-base-resolution-high",
    dest="dual_base_resolution_high", type=int, default=64)
dual_group.add_argument("--dual-sigma-low",
    dest="dual_sigma_low", type=float, default=1.0)
dual_group.add_argument("--dual-sigma-high",
    dest="dual_sigma_high", type=float, default=20.0)
dual_group.add_argument("--dual-n-freq",
    dest="dual_n_freq", type=int, default=64)
dual_group.add_argument("--dual-no-gate",
    dest="dual_use_gate", action="store_false", default=True,
    help="禁用空间门控网络")
dual_group.add_argument("--dual-hf-activate-ratio",
    dest="dual_hf_activate_ratio", type=float, default=0.6,
    help="训练进度超过此比例时高频开始激活，默认 0.6")
dual_group.add_argument("--dual-hf-max-weight",
    dest="dual_hf_max_weight", type=float, default=1.0)
dual_group.add_argument("--dual-sparsity-weight",
    dest="dual_sparsity_weight", type=float, default=0.01)
dual_group.add_argument("--dual-gate-weight",
    dest="dual_gate_weight", type=float, default=0.5)
```

在 `main()` 的 `NeUF(...)` 调用中补充对应 kwargs（与现有 hash 参数格式一致）：

```python
neuf = NeUF(
    # ... 现有参数不变 ...
    dual_pe_type=args.dual_pe_type,
    dual_n_levels_low=args.dual_n_levels_low,
    dual_n_levels_high=args.dual_n_levels_high,
    dual_finest_resolution_low=args.dual_finest_resolution_low,
    dual_finest_resolution_high=args.dual_finest_resolution_high,
    dual_base_resolution_low=args.dual_base_resolution_low,
    dual_base_resolution_high=args.dual_base_resolution_high,
    dual_sigma_low=args.dual_sigma_low,
    dual_sigma_high=args.dual_sigma_high,
    dual_n_freq=args.dual_n_freq,
    dual_use_gate=args.dual_use_gate,
    dual_hf_activate_ratio=args.dual_hf_activate_ratio,
    dual_hf_max_weight=args.dual_hf_max_weight,
    dual_sparsity_weight=args.dual_sparsity_weight,
    dual_gate_weight=args.dual_gate_weight,
)
```

---

## 使用示例

### DUAL_HASH 模式（默认推荐）

```bash
python main.py \
  --encoding DUAL_HASH \
  --dual-pe-type hash \
  --dual-n-levels-low 8 \
  --dual-n-levels-high 8 \
  --dual-finest-resolution-low 64 \
  --dual-finest-resolution-high 512 \
  --dual-hf-activate-ratio 0.6 \
  --dual-hf-max-weight 1.0 \
  --dual-sparsity-weight 0.01 \
  --tv-weight 5.0 \
  --nb-iters-max 20000
```

### DUAL_FREQ 模式（更可控的频率边界）

```bash
python main.py \
  --encoding DUAL_FREQ \
  --dual-pe-type fourier \
  --dual-sigma-low 1.0 \
  --dual-sigma-high 20.0 \
  --dual-n-freq 64 \
  --dual-no-gate \
  --dual-hf-activate-ratio 0.5 \
  --tv-weight 5.0
```

---

## 变更摘要

| 文件 | 类型 | 变更内容 |
|------|------|---------|
| `dual_freq_encoder.py` | **新建** | RFFEncoder / GateNet / DualFreqEncoder 三个类 |
| `nerf_network.py` | 修改 | import + 2个成员变量 + `init_dual_encoding` + `_init_from_ckpt` 分支 + `get_save_dict` 分支 |
| `main.py` | 修改 | 16个新实例变量 + `_initialize_model_encoding` 分支 + 训练循环 progress 更新 + 损失追加 + 2个新方法 + CLI 参数 |

**不改动文件**：`dataset.py` / `slice_renderer.py` / `hash_encoder.py` / `base_encoder.py` / `utils.py` / `bakeDataset.py`
