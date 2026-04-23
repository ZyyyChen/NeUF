# 两阶段课程训练实现计划（给 Claude Code）

## 目标

新增训练模式 `CurriculumRS`（Curriculum Random→Slice），实现：

```
阶段一 [0, phase_switch_ratio)：Random 模式
  - 快速建立全局密度场
  - 高频分支处于抑制状态（hf_weight ≈ 0）
  - loss：MSE + sparsity（无空间邻域约束）

阶段二 [phase_switch_ratio, 1.0]：Slice 模式
  - 精修空间结构和边界
  - 高频分支开始激活（hf_weight 从 0 爬升到 1）
  - loss：MSE + TV_low + 梯度loss + wavelet_hf
```

切换点 `phase_switch_ratio` 与 `hf_activate_ratio` 保持同一个值，两者联动。

---

## 重要说明

**每次修改前必须先读取文件最新内容，根据实际代码做精确 str_replace。**
不要盲目套用本文档中的代码片段，变量名或结构可能已有差异。

涉及文件：`main.py` 只改这一个文件。
`dual_freq_encoder.py` 和 `nerf_network.py` 本次不需要改动。

---

## Step 1：新增实例变量

### 1-a 在 `NeUF.__init__` 的 kwargs 解析区域

找到 `self.training_mode = kwargs.get("training_mode", "Random")` 这一行，
在其**之后**追加：

```python
self.phase_switch_ratio = float(kwargs.get("phase_switch_ratio", 0.5))
# 当前所处阶段，运行时动态更新，初始为 "Random"
self._active_mode: str = "Random"
```

### 1-b 确认 `_validate_configuration` 接受新模式

找到：
```python
if self.training_mode not in {"Random", "Slice", "Patch"}:
    raise ValueError(f"Unknown training mode: {self.training_mode}")
```

替换为：
```python
if self.training_mode not in {"Random", "Slice", "Patch", "CurriculumRS"}:
    raise ValueError(f"Unknown training mode: {self.training_mode}")
```

### 1-c 在 `_validate_configuration` 末尾追加参数校验

```python
if not (0.0 < self.phase_switch_ratio < 1.0):
    raise ValueError(
        f"phase_switch_ratio must be in (0, 1), got {self.phase_switch_ratio}"
    )
```

---

## Step 2：新增阶段判断辅助方法

在 `_validate_dataset_configuration` 方法**之前**插入：

```python
def _current_phase(self, progress: float) -> str:
    """
    根据训练进度返回当前激活的采样模式。
    仅在 CurriculumRS 模式下有意义，其他模式直接返回 self.training_mode。
    """
    if self.training_mode != "CurriculumRS":
        return self.training_mode
    return "Random" if progress < self.phase_switch_ratio else "Slice"
```

---

## Step 3：修改训练循环 `run`

### 3-a 找到训练循环中更新 `training_progress` 的那行

当前代码（可能如下，以实际为准）：
```python
if self.nerf.dual_encoder is not None:
    self.nerf.training_progress = iteration / max(1, self.N_iters)
```

在这行**之后**追加：

```python
# 更新当前激活模式（CurriculumRS 下动态切换）
self._active_mode = self._current_phase(self.nerf.training_progress)
```

同时在 `run` 方法里，找到阶段切换的位置，加一行 Tensorboard 记录
（放在 `self.tb_writer.add_scalar("train/lr", ...)` 之后）：

```python
# 记录当前阶段（0=Random，1=Slice），方便 Tensorboard 观察切换时机
phase_value = 0.0 if self._active_mode == "Random" else 1.0
self.tb_writer.add_scalar("train/phase", phase_value, iteration)
```

### 3-b 在阶段切换时打印一次提示

在 `_active_mode` 更新之后追加：

```python
# 首次切换到 Slice 阶段时打印提示
if (self.training_mode == "CurriculumRS"
        and self._active_mode == "Slice"
        and getattr(self, '_phase_switched', False) is False):
    self._phase_switched = True
    tqdm.tqdm.write(
        f"\n[CurriculumRS] 切换到 Slice 阶段，iteration={iteration}，"
        f"progress={self.nerf.training_progress:.3f}"
    )
```

在 `__init__` 的实例变量区补充初始化：
```python
self._phase_switched: bool = False
```

---

## Step 4：修改 `_sample_training_batch`

找到该方法，**整体替换**为以下版本：

```python
def _sample_training_batch(self):
    # CurriculumRS 模式：根据当前阶段动态选择采样策略
    active = self._active_mode  # 由训练循环实时更新

    if active == "Slice":
        return self._sample_slice_batch_decomposed()

    if active == "Patch":
        return self._sample_patch_batch()

    if active == "Random":
        return self._sample_random_batch_decomposed()

    raise ValueError(f"Unknown active mode: '{active}'")
```

---

## Step 5：新增 `_sample_random_batch_decomposed` 方法

这是原有 Random 分支的提取，加入 DUAL 模式的分解支持。

在 `_sample_slice_batch` 方法**之后**插入：

```python
def _sample_random_batch_decomposed(self):
    """Random 采样，DUAL 模式下同时获取 d_low / d_high。"""
    indices = self._next_random_indices()
    target = self.dataset.get_indices_pixels(indices)
    self._current_points   = self.dataset.get_indices_points(indices)
    self._current_viewdirs = self.dataset.get_indices_viewdirs(indices)

    if getattr(self.nerf, 'dual_mode', False):
        decomp = self.nerf.query_decomposed(
            self._current_points,
            self._current_viewdirs,
        )
        self._current_d_low  = decomp["d_low"]
        self._current_d_high = decomp["d_high"]
        density = decomp["density"]
    else:
        self._current_d_low  = None
        self._current_d_high = None
        density = self.slice_renderer.query_random_positions(
            self.nerf, indices, reshaped=False
        )

    return target, density
```

---

## Step 6：新增 `_sample_slice_batch_decomposed` 方法

在 `_sample_random_batch_decomposed` **之后**插入：

```python
def _sample_slice_batch_decomposed(self):
    """
    Slice 采样，DUAL 模式下同时获取 d_low / d_high。
    d_low / d_high 通过对整张切片的点调用 query_decomposed 获得。
    """
    slice_index = np.random.randint(len(self.dataset.slices))
    target = self.dataset.get_slice_pixels(slice_index).to(DEVICE)

    if getattr(self.nerf, 'dual_mode', False):
        points   = self.dataset.get_slice_points(slice_index)
        viewdirs = self.dataset.get_slice_viewdirs(slice_index)

        if self.jitter_training:
            points = self.slice_renderer._apply_jitter(
                points, self.slice_renderer.width_px, self.slice_renderer.height_px
            )

        self._current_points   = points
        self._current_viewdirs = viewdirs

        decomp = self.nerf.query_decomposed(points, viewdirs)
        self._current_d_low  = decomp["d_low"]
        self._current_d_high = decomp["d_high"]
        density = decomp["density"]
    else:
        self._current_d_low  = None
        self._current_d_high = None
        density = self.slice_renderer.render_slice_from_dataset(
            self.nerf, slice_index, jitter=self.jitter_training,
        )

    return target, density
```

---

## Step 7：修改 `_compute_training_loss`

找到该方法，**整体替换**为以下版本：

```python
def _compute_training_loss(self, target, density, iteration=0):
    mse_loss = self.mse(density, target)
    loss = mse_loss
    components: dict[str, torch.Tensor] = {"mse": mse_loss}

    active = self._active_mode  # "Random" 或 "Slice"

    # ── 非DUAL模式：原有逻辑按 active 分发 ─────────────────────
    if not getattr(self.nerf, 'dual_mode', False):
        if active == "Patch":
            tv = self._compute_patch_tv_loss(density)
            components["patch_tv"] = tv
            loss = loss + self.tv_weight * tv

        elif active == "Slice":
            grad = self._compute_gradient_loss(target, density)
            tv   = self._compute_tv_loss(density)
            components["gradient"] = grad
            components["tv"] = tv
            loss = loss + self.grad_weight * grad + self.tv_weight * tv

        elif active == "Random":
            if self._current_points is not None:
                smooth = self._compute_spatial_smoothness(
                    self._current_points, self._current_viewdirs
                )
                components["smoothness"] = smooth
                loss = loss + self.tv_weight * smooth

        return loss, components

    # ── DUAL 模式：分支级 loss，按阶段分发 ──────────────────────
    d_low  = self._current_d_low
    d_high = self._current_d_high

    if active == "Random":
        # 阶段一：只有稀疏约束，不需要空间邻域
        if d_high is not None:
            sparsity = torch.mean(torch.abs(d_high))
            components["dual_sparsity"] = sparsity
            loss = loss + self.dual_sparsity_weight * sparsity

    elif active == "Slice":
        # 阶段二：完整空间约束
        if d_low is not None:
            # TV 只加在低频分支（核心：不压制高频）
            tv_low = self._compute_tv_loss(d_low)
            components["tv_low"] = tv_low
            loss = loss + self.tv_weight * tv_low

            # 梯度 loss 加在总输出（边界对齐）
            grad = self._compute_gradient_loss(target, density)
            components["gradient"] = grad
            loss = loss + self.grad_weight * grad

        if d_high is not None:
            # 高频稀疏（Slice 阶段仍保留，但权重可以更小）
            sparsity = torch.mean(torch.abs(d_high))
            components["dual_sparsity"] = sparsity
            loss = loss + self.dual_sparsity_weight * 0.1 * sparsity

        if d_low is not None and d_high is not None:
            # Wavelet 高频增强 loss（只有完整切片才能计算）
            wavelet_loss = self._compute_wavelet_hf_loss(d_low, density)
            components["wavelet_hf"] = wavelet_loss
            loss = loss + 0.1 * wavelet_loss

    return loss, components
```

---

## Step 8：新增 CLI 参数

### 8-a 在 `parse_args` 中找到 `--training-mode` 的定义

找到：
```python
parser.add_argument("--training-mode", default="Random", choices=["Random", "Slice", "Patch"])
```

替换为：
```python
parser.add_argument(
    "--training-mode",
    default="Random",
    choices=["Random", "Slice", "Patch", "CurriculumRS"],
)
```

### 8-b 在 `parse_args` 中追加新参数

在 `--smoothness-delta` 之后追加：

```python
parser.add_argument(
    "--phase-switch-ratio",
    dest="phase_switch_ratio",
    type=float,
    default=0.5,
    help=(
        "CurriculumRS 模式下从 Random 切换到 Slice 的训练进度阈值，"
        "建议与 --dual-hf-activate-ratio 设为同一值，默认 0.5"
    ),
)
```

### 8-c 在 `main()` 的 `NeUF(...)` 调用中追加

```python
phase_switch_ratio=args.phase_switch_ratio,
```

---

## Step 9：验证清单

```
[ ] main.py:
    - _validate_configuration 接受 "CurriculumRS"
    - _current_phase 方法存在
    - self._active_mode 在 run 循环中每次迭代更新
    - self._phase_switched 初始化为 False
    - _sample_training_batch 按 _active_mode 分发
    - _sample_random_batch_decomposed 方法存在
    - _sample_slice_batch_decomposed 方法存在
    - _compute_training_loss 按 active 分发，DUAL 分支 Slice 阶段包含 tv_low

[ ] 运行验证（快速冒烟测试）：
    python main.py \
        --encoding DUAL_HASH \
        --training-mode CurriculumRS \
        --phase-switch-ratio 0.5 \
        --dual-hf-activate-ratio 0.5 \
        --nb-iters-max 200 \
        --plot-freq 100 \
        --save-freq 200
    → 控制台在 iteration≈100 附近打印 "[CurriculumRS] 切换到 Slice 阶段"
    → Tensorboard train/phase 曲线在该点从 0 跳到 1
    → loss/components/tv_low 在切换后出现
    → loss/components/dual_sparsity 在整个训练期间存在
```

---

## 参数推荐配置

```bash
python main.py \
  --encoding DUAL_HASH \
  --training-mode CurriculumRS \
  --phase-switch-ratio 0.5 \
  --dual-hf-activate-ratio 0.5 \
  --nb-iters-max 20000 \
  --tv-weight 5.0 \
  --grad-weight 0.1 \
  --dual-sparsity-weight 0.01 \
  --dual-hf-max-weight 1.0
```

关键联动：`--phase-switch-ratio` 和 `--dual-hf-activate-ratio` **必须设为同一个值**，
确保模式切换和高频激活同步发生。

---

## 训练过程时序图

```
iteration:    0          5000        10000       15000       20000
progress:     0.0         0.25        0.5         0.75        1.0
              │                        │
              ▼                        ▼
阶段:      Random ──────────────────► Slice
              │                        │
高频权重:   ≈0.0 ───────────────────► 0.5 ──────────────────► 1.0
              │                        │
loss:       MSE                      MSE
          + sparsity               + tv_low        (TV不再伤高频)
                                   + gradient
                                   + sparsity×0.1
                                   + wavelet_hf    (高频增强开始)
```
