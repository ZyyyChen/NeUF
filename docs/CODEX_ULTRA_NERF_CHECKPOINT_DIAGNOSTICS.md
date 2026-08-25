# Codex 执行任务：Ultra-NeRF 当前 checkpoint 的三项只读诊断

## 1. 任务目的

当前 Ultra-NeRF 物理渲染结果被高密度随机颗粒主导，主要解剖结构明显丢失。本任务只对**现有 checkpoint**做三项诊断，以判断问题来自：

1. 反射/散射能量分配异常；
2. `rho_b`、`rho_s` 的硬 Bernoulli 路径阻断梯度；
3. 将已经 scan-converted 的凸阵扇形图像错误地按竖直图像列执行 `cumprod`。

在三项诊断之前，必须先建立可靠的扇区有效区域掩膜，完全排除扇区外背景和设备界面文字。

本任务的输出是诊断报告，不是新的重建结果。

---

## 2. 严格范围

### 2.1 允许

- 读取现有数据、配置和 checkpoint；
- 新增一个独立的只读诊断脚本；
- 对一个固定代表性 frame 做前向传播；
- 为检查梯度做**一次** `loss.backward()`；
- 给现有 renderer 增加 `return_intermediates=True` 或 debug hook；
- 保存数组、统计量、PNG、JSON 和 Markdown 报告；
- 生成扇区掩膜，但不得覆盖原始图像。

### 2.2 禁止

- 禁止重新训练；
- 禁止 `optimizer.step()`；
- 禁止修改 checkpoint 权重；
- 禁止参数扫描；
- 禁止启动多个随机种子实验；
- 禁止增加新的损失、网络、PSF 或渲染模型；
- 禁止用平滑、去噪或重新归一化“改善”当前结果；
- 禁止修改 `recons3D.py`、`recons3d_exact_from_saved.py` 和 KNN 导出路径；
- 禁止把这项任务扩展成大量 smoke tests；
- 禁止独立 min-max 拉伸 `R`、`B` 和 `E` 后做视觉比较。

必须在代码中加入硬保护：

```python
assert optimizer_step_count == 0
```

诊断脚本不得创建 optimizer。若复用了训练器，只能调用 `zero_grad()` 和 `backward()`，并 monkey-patch 或断言 `optimizer.step()` 从未被调用。

---

## 3. 开始前的代码定位

不要假设文件名与旧版本完全相同。先在实际项目中定位：

```bash
rg -n "rho_b|rho_s|border|scatter|attenuation|reflection|cumprod|return_intermediates|checkpoint" .
```

确认并记录：

- 当前 checkpoint 路径；
- dataset/config 路径；
- Ultra-NeRF 五个原始输出的真实通道顺序；
- `R`、`B`、`T_att`、`T_ref` 在当前代码中的真实变量；
- 当前 `cumprod` 的维度；
- 当前训练图像的 `[H,W]` 方向；
- 当前 renderer 使用的像素点/射线样本及 reshape 顺序；
- 当前 loss 实现；
- 是否已经存在可靠的二维扇区 mask。

旧设计的预期通道顺序是：

```text
0: alpha_raw
1: beta_raw
2: rho_b_raw
3: rho_s_raw
4: phi_raw
```

但诊断脚本必须从当前实现或 checkpoint 配置核实，不能只复制该顺序。

建议新增入口：

```text
neuf/diagnose_ultra_nerf_checkpoint.py
```

建议命令接口：

```bash
python -m neuf.diagnose_ultra_nerf_checkpoint \
  --checkpoint <CURRENT_CHECKPOINT> \
  --dataset <DATASET_OR_CONFIG> \
  --frame-index <FIXED_FRAME_INDEX> \
  --seed 0 \
  --output-dir <RUN_DIR>/checkpoint_diagnostics
```

若项目已经有统一 CLI，接入现有入口，但仍需提供一个单命令复现方式。

---

## 4. 第零步：扇区有效区域掩膜

掩膜必须先于其他三项诊断生成并核验。后续所有统计、loss、显示范围和路径选择均使用同一掩膜。

### 4.1 掩膜必须排除

- 凸阵扇区外黑色背景；
- 底部 `DROIT`、`GCT` 等文字；
- 右侧日期、深度、增益及其他设备参数；
- 左右边缘的刻度、符号和其他屏幕标记；
- 任何与组织图像无关的 UI 元素。

### 4.2 掩膜来源优先级

按以下优先级执行：

1. **优先使用采集/预处理阶段已有的可靠 sector/FOV mask**；
2. 若存在准确的凸阵 scan-conversion 参数，按探头几何生成扇区 mask；
3. 只有前两项均不存在时，才从全部训练帧自动估计固定 mask。

不要把 `BaseSliceRenderer._points_in_scan_bounds()` 自动当作二维扇区 mask。它若只是三维 bounding-box 检查，不能排除屏幕文字和扇区外背景。

### 4.3 自动估计的最低要求

若必须从图像估计：

1. 使用全部或均匀抽取的多帧图像建立稳定的空间 support，不能根据单张图像的灰度逐帧产生 mask；
2. 对跨帧最大值或高分位图做低阈值二值化；
3. 保留位于图像中央、面积最大的扇区连通分量；
4. 使用 closing 和 hole filling 补全真实组织中的低回声/无回声区域；
5. 去除所有与主扇区不连通的文字和 UI 连通分量；
6. 必要时轻微向内腐蚀 1–3 pixels，避免扇区边缘和黑色背景泄漏；
7. mask 必须是固定 `[H,W]` 布尔数组，并应用到所有 frame。

不能简单使用 `image > 0` 作为最终 mask，因为组织内部可能存在低回声区域，且文字同样非零。

### 4.4 强制核验

必须输出：

```text
mask/sector_mask.npy
mask/sector_mask.png
mask/mask_overlay_frame_<id>.png
mask/mask_overlay_three_frames.png
mask/mask_stats.json
```

`mask_overlay_three_frames.png` 应包含浅部、中部、深部或扫描序列前/中/后的三张代表帧，显示 mask 边界。

`mask_stats.json` 至少记录：

```json
{
  "height": 0,
  "width": 0,
  "valid_pixel_count": 0,
  "valid_fraction": 0.0,
  "connected_components_kept": 1,
  "source": "existing|probe_geometry|estimated",
  "ui_regions_excluded": true
}
```

若自动 mask 仍包含文字、日期或右侧参数，必须停止并报告 `MASK_INVALID`，不能继续后续诊断。

### 4.5 mask 的使用方式

- 物理 renderer 仍按当前 checkpoint 的原始逻辑完成一次完整前向传播；
- 不要在 renderer 前将原始参数图乘 mask，以免改变现有模型的真实行为；
- renderer 完成后，mask 用于统计、loss、显示和路径选择；
- PNG 中 mask 外统一显示为黑色；
- NPY 中可同时保存原始完整数组和 masked 数组，masked 数组的无效位置设为 `NaN`。

---

## 5. 诊断一：导出物理分量和 R-only/B-only

### 5.1 单次前向传播

固定：

- 一个 frame index；
- 当前 checkpoint；
- 当前训练/验证配置；
- `seed=0`；
- 与当前评估一致的模型模式。

对同一次 forward 的返回值导出：

```text
E                  # 当前最终输出
R                  # reflection component, 即 R-only
B                  # backscattering component, 即 B-only
rho_b
rho_s
phi
T_att
T_ref
```

如果已有以下中间量，也一并保存，但不要为此改变公式：

```text
alpha
beta
border_indicator G
scatter_indicator H_s
G_psf
S_psf
transmission I
```

`R-only` 和 `B-only` 必须来自**同一次随机采样、同一次 forward** 中已经计算出的 `R` 和 `B`。不要分别关闭某个分支后重新运行，因为重新采样会造成不可比较。

### 5.2 数值一致性

检查：

```python
reconstruction_error = max_abs(E - (R + B))
```

要求：

```text
reconstruction_error <= 1e-6
```

若当前实现还有确定且已记录的后处理，应同时报告后处理前后的恒等关系，不能静默忽略。

### 5.3 统计量

所有统计仅在 `sector_mask == True` 内计算：

```python
sum_abs_R = sum(abs(R[mask]))
sum_abs_B = sum(abs(B[mask]))
B_fraction = sum_abs_B / (sum_abs_R + sum_abs_B + eps)
R_fraction = 1.0 - B_fraction
```

同时记录每张图在 mask 内的：

- min；
- max；
- mean；
- standard deviation；
- p01、p50、p95、p99、p99.5；
- nonzero fraction；
- saturation fraction（若有显示裁剪）。

对 `rho_b`、`rho_s`、`phi` 额外记录：

- `[0,1]` 直方图；
- mean/std；
- p05、p50、p95；
- 落在 `[0.45,0.55]` 的比例，用来判断是否停留在 `sigmoid(0)≈0.5` 附近。

对 `T_att`、`T_ref` 检查范围、浅部值和随深度变化，但不要在路径几何尚未核验前解释为真实物理衰减。

### 5.4 可视化规则

输出：

```text
maps/frame_<id>_target.png
maps/frame_<id>_E.png
maps/frame_<id>_R_only.png
maps/frame_<id>_B_only.png
maps/frame_<id>_rho_b.png
maps/frame_<id>_rho_s.png
maps/frame_<id>_phi.png
maps/frame_<id>_T_att.png
maps/frame_<id>_T_ref.png
maps/frame_<id>_decomposition_panel.png
arrays/*.npy
map_statistics.json
```

显示范围：

- `rho_b`、`rho_s`、`phi`、`T_att`、`T_ref` 固定为 `[0,1]`；
- `E`、`R`、`B` 必须使用同一个显示范围；
- 共同上限可取 `E[mask]` 的 `p99.5`，并在 JSON 中记录；
- 禁止对 `R`、`B`、`E` 分别 min-max；
- 所有图明确标注变量名、frame index、checkpoint 和 seed；
- mask 外全部置黑。

### 5.5 诊断标志

写入 `diagnostic_flags.json`：

```text
SCATTER_DOMINANT = B_fraction >= 0.80
RHO_B_NEAR_HALF  = fraction(rho_b in [0.45,0.55]) >= 0.50
RHO_S_NEAR_HALF  = fraction(rho_s in [0.45,0.55]) >= 0.50
```

阈值只用于自动标记，不用于修改模型。

---

## 6. 诊断二：rho_b/rho_s 梯度是否被阻断

### 6.1 只允许一次 backward

执行：

```python
model.zero_grad(set_to_none=True)

outputs = renderer(..., return_intermediates=True, return_raw=True)
raw = outputs["raw"]
raw.retain_grad()

pred = outputs["E"]
loss = masked_diagnostic_loss(pred, target, sector_mask)
loss.backward()
```

之后立即停止。严禁调用 `optimizer.step()`。

优先复用当前训练 loss，并确保只在 mask 内计算。若当前 SSIM/MS-SSIM 实现不支持 mask，则本次梯度可达性诊断使用：

```python
loss = ((pred - target)[mask] ** 2).mean()
```

并在报告中明确写出 `diagnostic_loss = masked_mse`。本诊断只判断梯度路径是否存在，不评价训练目标优劣。

### 6.2 必须检查两个层级

#### A. 原始五通道输出的梯度

记录：

```python
raw_grad_mean_abs[ch]
raw_grad_max_abs[ch]
raw_grad_l2[ch]
```

分别对应：

```text
alpha_raw, beta_raw, rho_b_raw, rho_s_raw, phi_raw
```

这是最直接的判断依据。

#### B. 输出头参数的梯度

若五个输出来自一个 `Linear(...,5)`：

```python
weight_row_grad_norm[ch] = head.weight.grad[ch].norm()
bias_grad_abs[ch] = abs(head.bias.grad[ch])
```

若使用五个独立 head，则记录每个 head 全部参数的总 L2 grad norm。

不要只统计整个共享 MLP 的总梯度，因为 alpha/beta/phi 的梯度会掩盖 rho 通道为零的问题。

### 6.3 判定

默认数值阈值：

```text
ZERO_GRAD_THRESHOLD = 1e-12
ACTIVE_GRAD_THRESHOLD = 1e-9
```

标记：

```text
RHO_B_GRAD_ZERO = rho_b_raw_grad_l2 <= 1e-12
RHO_S_GRAD_ZERO = rho_s_raw_grad_l2 <= 1e-12
OTHER_CHANNELS_ACTIVE = max(alpha,beta,phi raw_grad_l2) >= 1e-9
BERNOULLI_GRAD_BLOCKED = RHO_B_GRAD_ZERO and RHO_S_GRAD_ZERO and OTHER_CHANNELS_ACTIVE
```

同时输出：

```text
gradients/gradient_norms.json
gradients/gradient_norms.csv
gradients/gradient_barplot.png
```

报告中必须说明硬 Bernoulli、`.detach()`、`stop_gradient` 或不可重参数化 `.sample()` 的具体代码位置。

不要因为零梯度而在本任务中改成 Relaxed Bernoulli 或概率期望；这里只诊断和报告。

---

## 7. 诊断三：在原图上画出10条真实 cumprod 传播路径

### 7.1 目标

要画的是**当前代码实际用于组织 `cumprod` 的路径**，不是根据理想凸阵几何另外画出的示意图。

必须追踪：

- `raw` 从像素/三维点 reshape 成 A-line 张量的真实索引；
- `cumprod(dim=...)` 的真实维度；
- 每条路径上从浅到深的样本次序；
- 每个样本对应的原图像素坐标 `(row,col)`；
- 如有 world-space 样本，同时保存其 `(x,y,z)`。

### 7.2 选择10条路径

在有效扇区内按横向覆盖均匀选择10条路径：

```python
selected = evenly_spaced_valid_paths(total=10)
```

不得只选中央附近。左右外侧路径对判断凸阵发散几何最重要。

### 7.3 映射回原图

优先使用 renderer 中的原始索引映射。若当前代码只保存了 world points，则使用数据集中同一 frame 的完整像素点网格做 nearest-neighbor 回投影，并记录最大/中位回投影误差。

在原始目标图上：

- 每条路径用不同颜色；
- 用圆点标出浅部起点；
- 用箭头指示 `cumprod` 的深度方向；
- 标记路径编号；
- 叠加 sector mask 边界；
- 保留原始图像宽高比例；
- 不显示扇区外 UI。

输出：

```text
paths/frame_<id>_cumprod_paths_overlay.png
paths/frame_<id>_cumprod_paths_mask_only.png
paths/path_pixel_coordinates.json
paths/path_world_coordinates.npy
paths/path_geometry_metrics.json
```

### 7.4 自动几何指标

对每条路径的像素坐标计算：

```text
horizontal_drift_px = max(col) - min(col)
vertical_extent_px  = max(row) - min(row)
direction_angle_deg
inside_mask_fraction
depth_order_monotonic
```

定义：

```text
NEAR_VERTICAL_PATH:
    horizontal_drift_px <= max(1 px, 0.002 * W)

PARALLEL_VERTICAL_GEOMETRY:
    至少 8/10 条路径为 NEAR_VERTICAL_PATH
```

另记录10条路径方向角的 spread。对于扇形凸阵图像，除中央路径外，左右路径应在图像坐标中向不同方向发散，并应大致指向共同的虚拟中心。如果当前路径几乎都是恒定列或彼此平行，则不能代表 scan-converted 扇形图像的真实 A-line。

### 7.5 硬停止条件

若满足任一条件：

```text
PARALLEL_VERTICAL_GEOMETRY == true
任一主要路径 depth_order_monotonic == false
任一主要路径 inside_mask_fraction < 0.95
cumprod 维度无法与真实路径索引对应
```

则必须：

1. 在报告中设置：

```json
{
  "STOP_PHYSICS_BRANCH": true,
  "reason": "current cumprod paths do not match convex-sector propagation geometry"
}
```

2. 生成：

```text
STOP_PHYSICS_BRANCH.txt
```

3. 不得继续训练、微调、参数扫描或正式 A/B 实验；
4. 不得在本任务中擅自重写凸阵 scan conversion；
5. 只报告需要恢复的探头/scan-conversion 信息，例如虚拟中心、曲率半径、扇区角度、深度采样及 pixel-to-beam 映射。

---

## 8. 输出目录

最终目录必须为：

```text
checkpoint_diagnostics/
├── command.txt
├── checkpoint_info.json
├── sector_mask.npy
├── diagnostic_summary.json
├── diagnostic_report.md
├── diagnostic_flags.json
├── mask/
│   ├── sector_mask.png
│   ├── mask_overlay_frame_<id>.png
│   ├── mask_overlay_three_frames.png
│   └── mask_stats.json
├── maps/
│   ├── frame_<id>_target.png
│   ├── frame_<id>_E.png
│   ├── frame_<id>_R_only.png
│   ├── frame_<id>_B_only.png
│   ├── frame_<id>_rho_b.png
│   ├── frame_<id>_rho_s.png
│   ├── frame_<id>_phi.png
│   ├── frame_<id>_T_att.png
│   ├── frame_<id>_T_ref.png
│   └── frame_<id>_decomposition_panel.png
├── arrays/
│   └── *.npy
├── gradients/
│   ├── gradient_norms.json
│   ├── gradient_norms.csv
│   └── gradient_barplot.png
└── paths/
    ├── frame_<id>_cumprod_paths_overlay.png
    ├── frame_<id>_cumprod_paths_mask_only.png
    ├── path_pixel_coordinates.json
    ├── path_world_coordinates.npy
    └── path_geometry_metrics.json
```

若触发停止条件，根目录额外包含：

```text
STOP_PHYSICS_BRANCH.txt
```

---

## 9. diagnostic_summary.json 最低结构

```json
{
  "checkpoint": "",
  "frame_index": 0,
  "seed": 0,
  "no_retraining": true,
  "optimizer_step_count": 0,
  "mask_valid": false,
  "reconstruction_error_E_minus_R_plus_B": 0.0,
  "B_fraction": 0.0,
  "R_fraction": 0.0,
  "rho_b_near_half_fraction": 0.0,
  "rho_s_near_half_fraction": 0.0,
  "rho_b_raw_grad_l2": 0.0,
  "rho_s_raw_grad_l2": 0.0,
  "BERNOULLI_GRAD_BLOCKED": false,
  "parallel_vertical_path_count": 0,
  "PARALLEL_VERTICAL_GEOMETRY": false,
  "STOP_PHYSICS_BRANCH": false,
  "stop_reason": ""
}
```

JSON 的 key 名可以增加，不能删除上述核心字段。

---

## 10. 最终 Markdown 报告

`diagnostic_report.md` 只回答以下问题：

1. mask 是否完全排除了扇区外背景和所有 UI？
2. 当前输出主要来自 `R` 还是 `B`？给出 `B_fraction/R_fraction`。
3. `rho_b`、`rho_s` 是否集中在 0.5 附近？
4. `rho_b_raw`、`rho_s_raw` 和对应输出头参数的 gradient norm 是多少？
5. 是否确认硬 Bernoulli 路径阻断了这两个通道的梯度？
6. `cumprod` 的实际维度和路径索引是什么？
7. 10条路径是扇形发散，还是竖直/平行？
8. 是否触发 `STOP_PHYSICS_BRANCH`？
9. 在不重新训练的前提下，当前 checkpoint 的失败最直接由哪些证据支持？

报告必须嵌入以下图片的相对链接：

- mask overlay；
- decomposition panel；
- gradient bar plot；
- 10条传播路径 overlay。

不要在本报告中提出或实现新的模型路线。本任务结束于证据和停止判断。

---

## 11. 最小验收标准

只有以下全部满足，任务才算完成：

- [ ] 未发生重新训练；
- [ ] `optimizer_step_count == 0`；
- [ ] 使用了固定 checkpoint、frame 和 seed；
- [ ] sector mask 已核验且排除所有 UI；
- [ ] 导出了 `R`、`B`、`rho_b`、`rho_s`、`phi`、`T_att`、`T_ref`；
- [ ] `R-only`、`B-only` 来自同一次 forward；
- [ ] 验证了 `E = R + B`；
- [ ] 完成一次且仅一次 backward；
- [ ] 分通道报告 raw-output 和 output-head grad norm；
- [ ] 在原图上画出了当前代码实际使用的10条路径；
- [ ] 给出了竖直/平行路径的自动判定；
- [ ] 若路径不符合凸阵扇形传播，已生成 `STOP_PHYSICS_BRANCH.txt`；
- [ ] 生成 `diagnostic_summary.json` 和 `diagnostic_report.md`；
- [ ] 未启动任何额外训练、参数扫描或大批量 smoke test。

完成后只提交诊断脚本、必要的最小 debug hook 和 `checkpoint_diagnostics/` 结果，不得顺手修改主训练方法。
