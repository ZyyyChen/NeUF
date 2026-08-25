# Codex 执行规范：在独立工程中复现经典 Ultra-NeRF 几何对比基线

> **执行对象**：Codex（在 `/home/zchen/Code` 所在主机上执行）
> **原始数据**：`/home/zchen/Code/geometry_aware_3dus_repro/data_raw`
> **新工程目录**：`/home/zchen/Code/ultra_nerf_classic_geometry_baseline`
> **任务性质**：在用户数据上重实现并评估经典 Ultra-NeRF。它是“论文方法在新数据上的基线复现”，不是对原论文数值结果的逐项复刻。
> **核心原则**：固定数据、固定位姿、固定经典模型；不得引入此前 NeUF 或 geometry-aware 工程中的任何模型、渲染器、训练器、损失、位姿优化或 sagittal 监督。

---

## 0. 一句话目标

新建一个完全隔离的 PyTorch 工程，只读访问 `data_raw`，按照 Wysocki 等人的经典 Ultra-NeRF 实现“五参数组织场 + 超声物理渲染”，完成动态序列的留出切片重建、独立 sagittal 几何评估、参数场导出及可重复的对比报告。

最终方法标签固定为：

```text
UltraNeRF-classic-fixed-pose
```

不得把任何改进版本仍命名为该标签。

---

## 1. 绝对约束：先读，再执行

### 1.1 工程隔离

只允许创建：

```text
/home/zchen/Code/ultra_nerf_classic_geometry_baseline
```

数据只允许从以下目录读取：

```text
/home/zchen/Code/geometry_aware_3dus_repro/data_raw
```

执行前：

```bash
export RAW_DATA=/home/zchen/Code/geometry_aware_3dus_repro/data_raw
export NEW_PROJECT=/home/zchen/Code/ultra_nerf_classic_geometry_baseline

test -d "$RAW_DATA" || { echo "Missing RAW_DATA: $RAW_DATA"; exit 1; }
test ! -e "$NEW_PROJECT" || { echo "NEW_PROJECT already exists; stop instead of reusing it"; exit 1; }
mkdir -p "$NEW_PROJECT"
cd "$NEW_PROJECT"
git init
```

如果 `NEW_PROJECT` 已存在，**停止并报告**，不要清空、覆盖或复用。

### 1.2 禁止复用的内容

不得导入、复制、软链接或通过 `sys.path` 访问旧工程的以下内容：

- `neuf` 包及其任何子模块；
- 现有 `nerf_network.py`、`slice_renderer.py`、`slice_render_ray.py`；
- `pose_refinement.py`、sagittal supervision；
- hash、dual-frequency、Kronecker 编码；
- KNN、2.5D 插值、`recons3D.py` 或旧体数据导出代码；
- geometry-aware 项目的模型、训练代码、checkpoint、缓存或结果；
- 任何针对当前数据手工调好的旧参数。

允许共享的只有：

1. `data_raw` 中的原始图像、原始位姿、标定信息和预先冻结的对比 split/protocol；
2. 公开 Ultra-NeRF 论文及官方代码，用于核对经典方法；
3. 所有方法共同使用、且与模型无关的评估协议文件。

完成实现后必须运行：

```bash
rg -n "from neuf|import neuf|pose_refinement|sagittal_supervision|dual_freq|kronecker|hash_encoder|recons3D" src tests scripts || true
```

结果必须为空。数据绝对路径只应出现在 YAML/JSON 配置和运行记录中，不应硬编码进 `src/`。

### 1.3 原始数据不可写

- 不得在 `data_raw` 中生成缓存、split、缩略图或日志；
- 所有派生数据写入新工程的 `artifacts/data_audit/`；
- 开始和结束时保存 `data_raw` 的文件清单、大小、mtime 和 SHA-256；
- 若原始数据哈希在实验期间变化，停止并标记该次运行无效。

---

## 2. 经典 Ultra-NeRF 基线的锁定定义

主要依据：

- 论文：[Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging](https://proceedings.mlr.press/v227/wysocki24a.html)
- 官方实现：[magdalena-wysocki/ultra-nerf](https://github.com/magdalena-wysocki/ultra-nerf)

先将官方仓库仅作为参考克隆到新工程内部，并记录 commit：

```bash
mkdir -p third_party
git clone https://github.com/magdalena-wysocki/ultra-nerf.git third_party/ultra-nerf-official
git -C third_party/ultra-nerf-official rev-parse HEAD > artifacts_official_commit.txt
```

`third_party/ultra-nerf-official` 不参与 Python import。若网络不可用，记录失败原因，按本文给出的锁定定义独立实现，不得转而复用旧 NeUF 代码。

### 2.1 网络

实现一个坐标到五个组织参数的 MLP：

\[
F_\Theta:\mathbf q=(x,y,z)\longmapsto
(\alpha,\beta,\rho_b,\rho_s,\phi).
\]

固定设置：

| 项目 | 锁定值 |
|---|---:|
| 输入 | 仅 3D 世界坐标，不输入 view direction |
| 位置编码 | NeRF Fourier PE，包含原坐标，`L=10`，输入维数 `3 + 2×3×10 = 63` |
| MLP | 8 个全连接隐藏层 |
| 隐藏宽度 | 128（采用官方命令行实现的默认宽度） |
| 激活 | ReLU |
| skip | 第 5 个隐藏层激活后拼接位置编码，使下一层接收 skip 特征 |
| 输出 | 5 个 raw channels，顺序严格为 `[alpha, beta, rho_b, rho_s, phi]` |
| view-direction branch | 无 |
| coarse/fine hierarchical sampling | 无 |
| hash grid / feature grid | 无 |
| per-frame latent code | 无 |

激活约束在渲染器中执行：

\[
\alpha=|a|,\quad
\beta=\sigma(b),\quad
\rho_b=\sigma(p_b),\quad
\rho_s=\sigma(p_s),\quad
\phi=\sigma(s).
\]

不要把 `abs(alpha)` 改为 Softplus，不要添加法向量、反射方向、显式 view MLP、uncertainty head 或其他输出。

### 2.2 线性探头射线几何

每个 B-mode frame 由 `W` 条平行 scan lines 构成，每条 scan line 沿深度采样 `H` 个点。一帧的网络查询张量为：

```text
[W scanlines, H depth samples, 3 world coordinates]
```

最终图像转置回：

```text
[H depth, W lateral]
```

使用物理单位时，原始 mm 坐标统一乘 `1e-3` 转为 m；图像深度和宽度也必须使用同一转换。不得对平移、像素间距和射线深度采用不同尺度。

对当前数据的局部坐标约定，先验证下式，而不是直接猜轴或符号。令 `y_h` 为深度坐标、`x_w` 为横向坐标、`R_i,t_i` 为第 `i` 帧位姿：

\[
\mathbf q_{i,h,w}^{\mathrm{pixel}}
=R_i[y_h,x_w,0]^T+t_i,
\]

则经典射线应写为：

\[
\mathbf o_{i,w}=R_i[0,x_w,0]^T+t_i,\qquad
\mathbf d_i=R_i[1,0,0]^T,
\]

\[
\mathbf q_{i,h,w}^{\mathrm{ray}}
=\mathbf o_{i,w}+y_h\mathbf d_i.
\]

必须满足：

```text
max ||q_pixel - q_ray|| < 1e-6 mm
```

若不满足，优先检查 quaternion 顺序、矩阵转置、局部轴定义和深度方向；不得通过翻转预测图像来掩盖几何错误。

### 2.3 经典物理渲染

对第 `r` 条 scan line 上第 `i` 个深度采样点，定义间距 `Δ_i`。使用 exclusive cumulative product，使当前位置的透射只由它之前的采样点决定。

衰减：

\[
A_{r,i}=\exp(-\alpha_{r,i}\Delta_i),\qquad
T^{\mathrm{att}}_{r,i}=\prod_{j<i}A_{r,j}.
\]

边界采样和反射透射：

\[
G_{r,i}\sim\mathrm{Bernoulli}(\rho_{b,r,i}),
\]

\[
T^{\mathrm{ref}}_{r,i}
=\prod_{j<i}(1-\beta_{r,j}G_{r,j}).
\]

散射模板：

\[
H_{r,i}\sim\mathrm{Bernoulli}(\rho_{s,r,i}),\qquad
S_{r,i}=H_{r,i}\phi_{r,i}.
\]

总传播项、反射和后向散射：

\[
T_{r,i}=T^{\mathrm{att}}_{r,i}T^{\mathrm{ref}}_{r,i},
\]

\[
R_{r,i}=T_{r,i}\beta_{r,i}(K_{\mathrm{PSF}}*G)_{r,i},
\]

\[
B_{r,i}=T_{r,i}(K_{\mathrm{PSF}}*S)_{r,i},
\]

\[
\hat I_{r,i}=R_{r,i}+B_{r,i}.
\]

PSF 严格按官方发布实现：

- `7 × 7` 归一化二维 Gaussian kernel（代码中的 radius `g_size=3`）；
- scanline 轴标准差为 `2 px`；
- depth 轴标准差为 `1 px`；
- zero padding，输出尺寸不变；
- 不学习 PSF。

保持官方发布代码的离散采样行为：`G`、`H` 使用 Bernoulli sample，并对 sample 停止梯度。论文附录明确说明该步骤并非完全可微。因此：

- `rho_b`、`rho_s` 输出行的直接梯度可能为零；
- 这属于经典基线的已知限制，必须记录，不能偷偷换成 Relaxed Bernoulli、straight-through estimator 或概率期望；
- RNG 状态必须进入 checkpoint，保证 resume 后的随机序列可复现。

论文描述了散射幅度的正态采样，但官方发布代码实际使用 `Bernoulli(rho_s) × sigmoid(phi)`。本基线以**公开代码的可执行行为**为准，不额外加入正态噪声，并在 `METHOD_FIDELITY.md` 中明确记录该差异。

不要添加：

- alpha compositing；
- 标准视觉 NeRF 的 density/radiance 积分；
- 额外 log compression；
- 学习式 TGC；
- Fresnel、reverberation 或多次反射；
- fan-beam 路径（除非数据审计证实该数据来自凸阵并提供准确扇形成像几何；当前默认是线阵平行 scan lines）。

### 2.4 损失函数

输入 B-mode 图像先按照第 4 节的固定规则归一化到 `[0,1]`。主损失固定为：

\[
\mathcal L
=0.9\,[1-\mathrm{MS\mbox{-}SSIM}(\hat I,I)]
+0.1\,\mathrm{MSE}(\hat I,I).
\]

设置：

- `MS-SSIM max_val=1`；
- filter size `7`；
- 每次迭代随机选择一张训练 frame，并渲染完整 frame；
- 不允许随机像素训练，因为 2D PSF 和 SSIM 要求完整的 scanline-depth 邻域。

论文附录报告 `lambda=0.9`，官方 CLI 默认是 `0.75`。本对比实验预注册 `0.9` 为唯一主设置；不得分别跑两个值后选择较好的结果。

---

## 3. 新工程结构

建立以下结构，不得在旧工程中补文件：

```text
ultra_nerf_classic_geometry_baseline/
├── README.md
├── METHOD_FIDELITY.md
├── pyproject.toml
├── requirements-lock.txt
├── configs/
│   ├── classic_primary.yaml
│   └── smoke.yaml
├── src/ultra_nerf_classic/
│   ├── __init__.py
│   ├── data.py
│   ├── poses.py
│   ├── rays.py
│   ├── encoding.py
│   ├── model.py
│   ├── renderer.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   └── export.py
├── scripts/
│   ├── inspect_data.py
│   ├── build_protocol.py
│   ├── visualize_geometry.py
│   ├── overfit_one_frame.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_parameter_volume.py
├── tests/
│   ├── test_data_orientation.py
│   ├── test_pose_convention.py
│   ├── test_ray_pixel_equivalence.py
│   ├── test_exclusive_cumprod.py
│   ├── test_psf.py
│   ├── test_renderer_outputs.py
│   ├── test_renderer_gradients.py
│   └── test_split_integrity.py
├── protocol/
├── artifacts/
├── runs/
├── reports/
└── third_party/ultra-nerf-official/
```

用 `src` layout 安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip freeze > requirements-lock.txt
```

至少依赖：PyTorch、NumPy、SciPy、h5py、Pillow/imageio、scikit-image、pytorch-msssim、pandas、PyYAML、matplotlib、pytest。LPIPS 只用于评估，不得进入训练损失。

---

## 4. Phase A：只做数据审计，不开始训练

当前执行说明无法直接读取主机上的 `data_raw`，因此 Codex 必须先生成事实清单，不得按文件名猜格式。

### 4.1 文件与变量清单

运行：

```bash
python scripts/inspect_data.py \
  --data-root "$RAW_DATA" \
  --output artifacts/data_audit
```

脚本要递归识别并报告：

- PNG/JPG/TIFF 等图像文件；
- `.npy/.npz` 的 shape、dtype、数值范围；
- MATLAB v7/v7.3 文件中的变量名、shape、dtype；
- `infos.json` 或其他 pose 文件的顶层键和每帧字段；
- 图像帧数与 pose 数是否一一对应；
- pixel spacing、ROI offset、物理宽度和深度；
- sagittal reference 图像及其几何信息；
- 是否存在有效区域 mask；
- 是否存在预先冻结的 `geometry_comparison_split.json`。

输出：

```text
artifacts/data_audit/
├── file_manifest.json
├── array_manifest.json
├── pose_manifest.json
├── intensity_histograms.png
├── frame_montage_original.png
├── candidate_orientations.png
└── audit_report.md
```

`audit_report.md` 必须区分“已从文件验证”和“仍未知”。缺少以下任一信息时停止，不得进入训练：

1. 动态 B-mode 帧与 pose 的可靠对应；
2. 图像的 depth/lateral 方向；
3. pixel spacing 或物理 probe width/depth；
4. quaternion 的顺序和从局部到世界的变换方向；
5. 可用于训练的超声有效区域。

### 4.2 统一张量方向

训练张量统一为：

```text
images: [N, H_depth, W_lateral]
poses:  [N, 4, 4], local-to-world
mask:   [H_depth, W_lateral] 或 [N, H_depth, W_lateral]
```

若原始 `data_recal` 为 `[frame, lateral, depth]`，明确执行：

```python
images = data_recal.transpose(0, 2, 1)
```

但只有在 montage 和物理尺寸共同证明方向正确时才允许这样做。保存原始到 canonical 的 axis permutation；不得在 loader、renderer 和 evaluator 中重复转置。

### 4.3 强度预处理

固定规则：

- `uint8`：除以 `255`；
- 其他整数：使用数据元信息给出的真实 bit depth；
- float 且已在 `[0,1]`：保持不变；
- float 超出 `[0,1]` 且无明确标尺：停止并要求补充信息；
- 所有帧使用同一映射；
- 禁止逐帧 min-max、CLAHE、去噪、锐化或 histogram matching。

完全排除屏幕黑色背景、文字、日期、设备参数、`DROIT/GCT` 等 overlay。mask 必须由原始图像或元信息产生并冻结，不能根据任何模型预测产生。

生成 `normalization.json` 和 mask 可视化。所有比较方法必须使用同一个 mask 和归一化规则。

### 4.4 quaternion 和坐标门禁

同时检查候选 `wxyz` 与 `xyzw` 顺序，不得仅凭字段名称决定。对每个候选：

- 验证 `R^T R≈I`、`det(R)≈1`；
- 绘制 probe trajectory；
- 绘制局部 depth、lateral、elevational 三个轴；
- 在起始、中间、结束三帧分别绘制 10 条 scan lines；
- 验证第 2.2 节的 `q_pixel == q_ray`；
- 确认射线从图像顶部向解剖深部传播，而不是沿 frame 间扫描轨迹传播。

必须生成：

```text
artifacts/data_audit/trajectory_axes.png
artifacts/data_audit/rays_frame_start.png
artifacts/data_audit/rays_frame_middle.png
artifacts/data_audit/rays_frame_end.png
artifacts/data_audit/pose_convention.json
```

若用于 `cumprod` 的传播路径在图像内不是 depth scan lines，停止，不得继续训练。

### 4.5 Phase A 通过条件

`reports/GATE_A_DATA_AND_GEOMETRY.md` 必须给出：

- 选定的数组变量和帧数；
- canonical shape；
- 强度范围；
- probe width/depth 与 spacing；
- quaternion 顺序；
- local-to-world 公式；
- 最大 ray/pixel 坐标误差；
- 训练 mask 覆盖率；
- sagittal 图像是否有可靠 pose；
- 所有未解决问题。

只有所有 blocking 项清零后才能进入 Phase B。

---

## 5. Phase B：冻结对比 protocol

### 5.1 动态序列 split

优先使用：

```text
$RAW_DATA/protocol/geometry_comparison_split.json
```

若它存在，复制其 SHA-256 到本次报告并只读使用。若不存在，在新工程中生成：

```text
protocol/geometry_comparison_split.json
```

默认单序列插值协议：

- `frame_index % 10 == 4`：validation；
- `frame_index % 10 == 9`：test；
- 其他：train；
- 首尾帧强制保留在 train，以避免把外推混入主插值任务；
- split 按原始 frame index 决定，不进行随机打乱；
- 三个集合的 frame、文件名和 pose 不得重复。

如果数据包含多个独立 sweep，禁止跨 sweep 随机拆帧。应以 sweep 为单位遵循已有 geometry comparison protocol；若没有 protocol，停止并先报告可选的 sweep-level 划分，不得自行挑选最有利的测试 sweep。

### 5.2 sagittal reference 的角色

sagittal 图像固定为：

```text
test-only external geometric reference
```

它不得用于：

- 网络训练；
- loss；
- pose refinement；
- early stopping；
- 超参数选择；
- checkpoint 选择；
- 训练强度映射拟合。

如果 sagittal pose 未由标定/几何流程给出，不允许利用 sagittal 图像内容优化它。本基线应把该项标为“无法进行可靠 cross-plane evaluation”，而不是执行图像配准后当作固定真值。

### 5.3 固定比较表

在开始训练前写入 `protocol/method_card.json`：

| 字段 | 值 |
|---|---|
| method | `UltraNeRF-classic-fixed-pose` |
| training images | dynamic train frames only |
| validation images | dynamic validation frames only |
| sagittal supervision | false |
| pose optimization | false |
| view direction input | false |
| rendering | classic five-parameter Ultra-NeRF |
| initialization from another method | false |
| test-time fitting | false |

---

## 6. Phase C：实现与单元测试

### 6.1 必须通过的测试

1. **位置编码**：shape 为 `[...,63]`，频率严格为 `2^0...2^9`；
2. **网络输出**：raw shape 为 `[...,5]`；
3. **参数范围**：`alpha>=0`，其余四项在 `[0,1]`；
4. **pose**：旋转矩阵正交且 determinant 为 1；
5. **ray/pixel 等价**：误差小于 `1e-6 mm`；
6. **exclusive cumprod**：用手算的小张量逐元素核对；
7. **PSF**：kernel 为 `7×7`、和为 1、scanline/depth 标准差方向无交换；
8. **renderer**：输出 `I,R,B,T_att,T_ref` shape 一致且无 NaN/Inf；
9. **梯度**：`alpha/beta/phi` 有可传播路径；记录 `rho_b/rho_s` 的零梯度行为，不将其当作代码错误；
10. **split**：train/val/test 无交集；
11. **mask**：训练和评估都不包含屏幕 overlay；
12. **可复现性**：同一 seed、同一 checkpoint、同一 RNG state 得到逐元素一致的 render。

运行：

```bash
pytest -q
```

失败时只修复导致测试失败的基础实现，不得用新增模型模块绕过测试。

### 6.2 一帧过拟合门禁

运行：

```bash
python scripts/overfit_one_frame.py \
  --config configs/smoke.yaml \
  --frame-id <中央训练帧> \
  --output runs/overfit_one_frame
```

要求：

- loss 明显持续下降；
- 输出不是全黑、常数或棋盘格；
- `R-only`、`B-only` 与最终 `R+B` 一致；
- 保存五个参数图、两个 transmission 图和 10 条传播路径；
- 保存每个输出 head 和共享 trunk 的 grad norm。

若一帧都无法拟合，禁止开始 200k-step 正式训练。

---

## 7. Phase D：正式训练

`configs/classic_primary.yaml` 至少锁定：

```yaml
method: UltraNeRF-classic-fixed-pose
data_root: /home/zchen/Code/geometry_aware_3dus_repro/data_raw
protocol: protocol/geometry_comparison_split.json

units:
  source: mm
  internal: m
  scale: 0.001

model:
  positional_encoding_levels: 10
  include_input: true
  depth: 8
  width: 128
  skip_after_hidden_layer: 5
  output_channels: 5
  use_viewdirs: false

renderer:
  samples_per_ray: native_image_depth
  sampling: equidistant_pixel_centers
  exclusive_cumprod: true
  psf_radius: 3
  psf_sigma_scanline_px: 2.0
  psf_sigma_depth_px: 1.0
  bernoulli_stop_gradient: true
  log_compression: false

loss:
  ms_ssim_weight: 0.9
  mse_weight: 0.1
  ms_ssim_filter_size: 7

optimizer:
  name: Adam
  learning_rate: 0.0001
  lr_decay_factor: 0.1
  lr_decay_steps: 250000

training:
  iterations: 200000
  frames_per_step: 1
  full_frame: true
  mixed_precision: false
  checkpoint_every: 2000
  validate_every: 2000
  seeds: [0, 1, 2]
```

`native_image_depth` 在 Phase A 后解析为实际整数并写入每次 run 的 resolved config。不得因为显存问题偷偷减少 depth samples。显存不足时对网络查询分 chunk，但渲染结果和损失仍必须来自完整 frame。

运行三次独立 seed：

```bash
for seed in 0 1 2; do
  python scripts/train.py \
    --config configs/classic_primary.yaml \
    --seed "$seed" \
    --run-dir "runs/classic_seed_${seed}"
done
```

checkpoint 选择仅依据 dynamic validation 的预注册指标。测试集和 sagittal reference 不参与选择。

每个 run 必须保存：

```text
runs/classic_seed_X/
├── resolved_config.yaml
├── environment.txt
├── git_state.txt
├── data_manifest_sha256.txt
├── protocol_sha256.txt
├── checkpoints/
├── rng_states/
├── train_metrics.csv
├── val_metrics.csv
├── grad_norms.csv
├── diagnostics/
└── logs/
```

训练中每 2000 steps 至少保存：

- train/val 总 loss、MSE、MS-SSIM；
- `alpha,beta,rho_b,rho_s,phi` 输出 head 的 grad norm；
- `R-only`、`B-only`、`R+B`；
- `T_att`、`T_ref`；
- 参数分布的 min/median/max 和饱和比例；
- 一张固定 validation frame 的相同可视化。

不得在看到正式结果后修改 mask、split、轴方向、loss 权重、PSF 或迭代数。任何修改必须形成新的 method tag，不能覆盖主基线。

---

## 8. Phase E：评估

### 8.1 留出动态切片

对每个 test frame 按其原始固定 pose 完整渲染。只在冻结的有效区域 mask 内计算：

- PSNR；
- SSIM；
- MS-SSIM；
- NRMSE；
- NCC；
- LPIPS（灰度复制为 3 通道，作为补充指标）。

输出 per-frame 数值，并汇总：

- mean；
- median；
- standard deviation；
- frame-level bootstrap 95% confidence interval；
- 三个 seed 的均值与 seed 间标准差。

同时报告运行时间、峰值显存、参数量和训练迭代数。

### 8.2 sagittal 几何评估

由于 sagittal 图像与动态 slices 可能具有不同增益、TGC 或动态范围，不能只给一组强度 PSNR 并据此判断几何。

在 sagittal pose 已被外部几何可靠确定的前提下，渲染 sagittal view，并分两类报告：

**原始强度指标**：

- 使用统一 `[0,1]` 预处理的 PSNR、SSIM、NCC；
- 不做逐图 min-max。

**几何优先指标**：

- gradient-NCC；
- gradient-magnitude SSIM；
- 在同一固定阈值/同一算子下得到的 edge Chamfer distance（换算为 mm）；
- 可选的 contrast-aligned SSIM，必须明确标成 secondary，并对所有比较方法执行同一对齐规则。

如果进行 affine intensity alignment `a·prediction+b`，必须：

- 单独报告对齐前、对齐后结果；
- 不得把对齐后指标冒充原始图像 fidelity；
- 所有方法使用完全相同的拟合区域与程序；
- 保存 `a,b` 和拟合 mask。

sagittal reference 仍不得反向更新网络或 pose。

### 8.3 3D 输出

Ultra-NeRF 没有唯一的 view-independent B-mode intensity volume。不得把某一个网络 channel 直接称为“重建 B-mode volume”。应导出：

```text
alpha.mhd/raw
beta.mhd/raw
rho_b.mhd/raw
rho_s.mhd/raw
phi.mhd/raw
```

另行保存：

- 按 test poses 渲染的 B-mode frames；
- sagittal render；
- 用统一、与模型无关的 compounding protocol 从渲染帧生成的可视化体数据；
- 体数据的 world origin、spacing、axis order 和方向矩阵。

参数体网格必须由训练/评估共同的 world bounding box 和固定 spacing 定义，并保存 `evaluation_grid.json`。不得为某个方法单独裁剪更有利的区域。

### 8.4 必须导出的诊断

对起始、中间、结束三张 test frame，导出：

- target；
- prediction；
- absolute error；
- `R-only`；
- `B-only`；
- `alpha`；
- `beta`；
- `rho_b`；
- `rho_s`；
- `phi`；
- `T_att`；
- `T_ref`；
- 10 条实际参与 `cumprod` 的传播路径。

该诊断用于确认物理分支确实沿每条 scan line 传播，而不是沿 frame 方向或体数据竖直列错误累积。

---

## 9. 结果解释与失败标准

### 9.1 不得把假设不匹配写成代码成功

经典 Ultra-NeRF 原论文使用多个重叠 sweep 和不同观察角度来约束五个组织参数。当前数据若只有一个稀疏旋转序列，而且相邻 slices 几乎没有重复空间采样，则五参数分解高度欠约束。

因此，结果差可能有两种原因，报告中必须分开：

1. **实现失败**：坐标、射线、转置、mask、cumprod、PSF 或训练没有通过门禁；
2. **方法假设不匹配**：实现通过所有门禁，但单 sweep/低重叠数据无法约束经典 Ultra-NeRF。

只有第二种情况才是有效的 negative baseline result。

### 9.2 正式失败条件

满足任一项则停止正式实验并标记失败：

- ray/pixel 坐标误差超过阈值；
- quaternion convention 无法唯一确认；
- 传播路径不是 depth scan lines；
- 一帧过拟合不收敛；
- 输出出现 NaN/Inf；
- train/test 数据泄漏；
- sagittal pose 是通过 test 图像优化得到的；
- `data_raw` 被修改；
- 实现导入了旧 NeUF/geometry-aware 模块；
- 为改善主结果修改了预注册模型或 protocol，却仍使用相同 method tag。

### 9.3 不允许的“补救”

主基线失败后，不得在同一结果中加入：

- sagittal supervision；
- pose refinement；
- 每帧 gain/bias；
- view-dependent MLP；
- hash/dual-frequency/Kronecker encoding；
- deterministic probability renderer；
- learned PSF；
- KNN/插值初始化；
- geometry-aware loss。

这些可以成为后续独立 ablation，但必须新建 method tag、独立 config 和独立报告，不能覆盖经典基线。

---

## 10. 最终交付物

Codex 完成后必须提交：

1. 全新、可运行的 Git 工程；
2. `reports/GATE_A_DATA_AND_GEOMETRY.md`；
3. `METHOD_FIDELITY.md`，逐项说明论文、官方代码和本 PyTorch 实现的对应关系；
4. 全部 `pytest` 结果；
5. 一帧过拟合报告；
6. 三个 seed 的训练配置和 checkpoint；
7. test per-frame 与汇总指标；
8. sagittal 原始强度与几何优先指标；
9. 五个参数体和渲染结果；
10. `R-only/B-only/T_att/T_ref` 与传播路径诊断；
11. `reports/FINAL_ULTRANERF_CLASSIC_BASELINE.md`；
12. 一条从空环境开始可复现的命令清单。

最终报告的第一张表固定为：

| Method | Pose optimized | Sagittal used in training | Test PSNR | Test SSIM | Test NCC | Sag gradient-NCC | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraNeRF-classic-fixed-pose | No | No | … | … | … | … | 3 |

最后给出明确结论，且只能是以下三类之一：

1. `Implementation invalid`：基础门禁未通过；
2. `Implementation valid, baseline underperforms`：实现有效，但经典模型不适合当前低重叠稀疏数据；
3. `Implementation valid, baseline competitive`：实现有效，指标具有竞争力。

不得只展示最好的一张图，也不得只报告三个 seed 中最好的 seed。

---

## 11. 建议执行顺序

```bash
# 1. 创建全新工程并安装环境
# 2. inspect_data.py：只做数据审计
# 3. 冻结 canonical orientation、pose convention、mask 和 split
# 4. 实现 PE、MLP、rays、renderer、loss
pytest -q

# 5. 一帧过拟合
python scripts/overfit_one_frame.py --config configs/smoke.yaml --frame-id <id>

# 6. 三个正式 seed
for seed in 0 1 2; do
  python scripts/train.py --config configs/classic_primary.yaml \
    --seed "$seed" --run-dir "runs/classic_seed_${seed}"
done

# 7. 固定 checkpoint 规则后统一评估
python scripts/evaluate.py \
  --config configs/classic_primary.yaml \
  --runs runs/classic_seed_0 runs/classic_seed_1 runs/classic_seed_2 \
  --output reports/evaluation

# 8. 导出五参数体和最终报告
python scripts/export_parameter_volume.py \
  --config configs/classic_primary.yaml \
  --output reports/volumes
```

每完成一个 gate 再进入下一阶段。不要在数据方向、pose 或射线尚未验证时直接长时间训练。

---

## 12. 参考文献与实现证据

1. Wysocki M, Azampour MF, Eilers C, et al. **Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging.** Medical Imaging with Deep Learning, PMLR 227:382–401, 2024.

   Paper: <https://proceedings.mlr.press/v227/wysocki24a.html>

2. Official Ultra-NeRF implementation:

   <https://github.com/magdalena-wysocki/ultra-nerf>

关键已核实事实：论文将 3D 坐标映射为 `alpha, beta, rho_b, rho_s, phi`，不向网络显式输入 viewing direction；使用八层 MLP、第五层 skip、`L=10` 位置编码，并以 SSIM 类损失和 L2 联合训练。官方实现采用沿 scan line 的等间距采样、exclusive cumulative products、Bernoulli 边界/散射采样和固定二维 Gaussian PSF。
