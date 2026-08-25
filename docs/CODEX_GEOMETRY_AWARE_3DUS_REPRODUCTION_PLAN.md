# Codex 执行说明：在现有数据集上独立复现 Geometry-aware View-dependent 3D Ultrasound

> 目标论文：Bin Liu et al., *Geometry-aware view-dependent 3D ultrasound implicit representation reconstruction*, Neurocomputing 681 (2026), 133356. DOI: <https://doi.org/10.1016/j.neucom.2026.133356>
>
> 方法基础：M. Wysocki et al., *Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging*. 论文与公式：<https://proceedings.mlr.press/v227/wysocki24a.html>

## 0. 任务目标

在用户当前 NeUF 数据集上复现目标论文的几何感知、视角依赖、物理渲染型三维超声隐式表示，并与当前 NeUF、KNN/IDW 插值等方法进行公平比较。

这是一个**全新的独立对比项目**。不得直接修改、重构、覆盖或提交用户当前 NeUF 项目中的任何文件。现有项目和原始数据只允许以只读方式访问。

需要同时实现并严格区分两条实验线：

1. **GA-Paper（严格论文复现）**：仅实现目标论文中经过原文核验的方法，不加入用户当前项目中的 DualFreq gate、Kronecker 编码、厚度射线积分、强度不确定性头或其他自行设计的模块。
2. **GA-Paper+Sag（数据集适配）**：在 GA-Paper 完成并可独立运行后，加入用户最重要的 sagittal 参考图像及其几何/位姿约束。它属于数据集适配实验，不能冒充论文原始方法。

最终回答的问题是：

- 论文的几何感知物理渲染是否优于当前的逐点强度隐式场？
- 在切片间几乎没有重复空间点、主要任务是切片间插值的条件下，几何感知表示是否仍然改善未见切片？
- 反射与散射的分解是否有数据支持，还是仅仅存在多解？
- sagittal 参考图像是否能改善扫描几何、跨平面结构一致性和插值质量？

---

## 1. 不可违反的项目隔离规则

### 1.1 新项目位置

先识别当前源项目根目录：

```bash
git rev-parse --show-toplevel
```

在其**父目录下创建同级目录**，建议名称：

```text
geometry_aware_3dus_repro/
```

如果当前目录不是 Git 仓库，则先找出包含 `neuf/`、训练入口和 `baked_dataset.pkl` 的源项目目录，然后仍在其父目录创建新项目。

创建前必须执行并记录以下检查：

- 新目录的真实路径不位于源项目根目录之内；
- 新目录不是源项目 Git worktree 的子目录；
- 新项目所有 checkpoint、缓存、日志、渲染结果和导出体数据都写入新目录；
- 原数据路径只通过配置文件引用，不复制、不改名、不重新保存到原目录；
- 不对源项目执行格式化、依赖升级、测试修复、Git 清理或任何“顺手修改”。

### 1.2 源项目完整性证据

在开始与结束时分别保存：

```bash
git -C <SOURCE_REPO> status --porcelain=v1
git -C <SOURCE_REPO> diff --stat
```

写入新项目：

```text
provenance/source_status_before.txt
provenance/source_status_after.txt
```

两者必须一致。源项目本来已有的未提交修改属于用户，不能覆盖、暂存或提交。

### 1.3 代码复用原则

- 只读检查旧实现，重新建立新项目的数据适配层和论文渲染器。
- 若必须读取 Python pickle 中的旧 `neuf.dataset` 类，可由新项目的导出脚本临时只读导入源项目，随后将数据导出为中立格式 HDF5/NPZ；训练不得长期依赖旧项目的内部可变状态。
- 复制少量纯几何函数时，在新文件中注明来源文件和复制日期；不要复制整个旧仓库。
- 新项目使用自己的 `.venv`、`pyproject.toml`/`requirements-lock.txt` 和独立 Git 仓库。

---

## 2. 开始编码前必须完成的论文核验

目标论文公开摘要足以确认总体思想，但不足以安全恢复所有公式和超参数。**不允许根据名称或 Ultra-NeRF 自行补齐目标论文的缺失细节，然后把它称为“精确复现”。**

### 2.1 搜索顺序

1. 在源项目、数据目录、用户下载目录和项目资料中查找目标论文 PDF。
2. 查找作者公开代码、补充材料或正式仓库。
3. 使用 DOI 页面核对正式版本。
4. 如果仍无全文，可以继续完成项目骨架、数据导出、几何审计和 Ultra-NeRF 兼容渲染器，但在进入 **GA-Paper** 实现前暂停，并明确向用户索要 PDF。

### 2.2 建立 `docs/paper_spec.md`

逐项抄录并引用论文页码、公式号、图号或表号：

| 项目 | 必须核验的内容 |
|---|---|
| 输入坐标 | 坐标归一化、世界单位、ray/A-line 定义、采样间距 |
| 隐式字段 | 网络实际输出的全部物理量、每个量的范围和激活函数 |
| 声阻抗 | 声阻抗如何预测、如何沿射线取相邻值 |
| 几何法向 | `∇Z` 的定义、归一化、是否 detach、数值稳定项 |
| 反射 | 声阻抗到反射系数的公式、入射角/法向的使用方式、符号约定 |
| 衰减 | 累积公式、频率、步长、TGC 或补偿项 |
| 散射 | `rho_s`、`phi`、Bernoulli/Normal 采样及可微近似 |
| PSF | 核的公式、大小、标准差/频率、卷积次序和边界处理 |
| 多尺度哈希 | coarse/fine 分支的层数、分辨率、特征维度、哈希表大小、字段连接关系 |
| 网络 | MLP 深度、宽度、skip、初始化和输出头 |
| 损失 | 图像损失、SSIM/L1/L2 权重、正则项及其权重 |
| 训练 | optimizer、学习率、scheduler、batch/ray 数、迭代数、随机种子 |
| 评估 | train/test 划分、指标、图像预处理和渲染设置 |

同时生成机器可读的 `configs/paper_spec.yaml`。每个值包含：

```yaml
value: ...
status: verified | inferred | unknown
source: "p. X, Eq. Y / Table Z / official code path"
```

只要核心公式仍为 `unknown`，实验名称必须包含 `fallback`，不得写成 `paper_reproduction`。

### 2.3 已核验的总体方法边界

以下内容可以作为实现骨架，但仍需用全文恢复精确形式：

- 目标论文不再像 Ultra-NeRF 那样直接预测反射率，而是预测空间声阻抗，并由声阻抗的神经梯度估计组织界面法向。
- 反射具有几何和入射方向依赖；散射用于描述微结构造成的细粒度斑点。
- 论文使用可分离的多尺度/多分辨率哈希特征，使大范围界面反射更多依赖粗尺度表示，细粒度散射依赖更高分辨率表示。
- 最终 B-mode 回波由反射项与后向散射项构成，并包含沿扫描线累积的剩余能量以及 PSF 卷积。

可以先按下列抽象接口组织代码，但 `F_paper` 必须由原文替换：

```math
\{\alpha(\mathbf{x}), Z(\mathbf{x}), \rho_s(\mathbf{x}),
\phi(\mathbf{x}), \ldots\}=f_\theta(\mathbf{x}),
```

```math
\mathbf{n}(\mathbf{x})=
\frac{\nabla_{\mathbf{x}} Z(\mathbf{x})}
{\|\nabla_{\mathbf{x}} Z(\mathbf{x})\|_2+\varepsilon},
```

```math
\beta_g(\mathbf{x},\mathbf{d})=
F_{\mathrm{paper}}\!\left(Z,\nabla Z,\mathbf{n},\mathbf{d},\Delta t\right),
```

```math
E(r,t)=R(r,t)+B(r,t).
```

散射骨架与 Ultra-NeRF 类似：

```math
H(r,t)\sim\mathrm{Bernoulli}(\rho_s(r,t)),\qquad
S(r,t)=H(r,t)\,\phi(r,t),
```

随后由 PSF、剩余能量和论文公式生成反射与后向散射。不要用普通 NeRF 的 RGB alpha-compositing 代替超声渲染。

---

## 3. 对用户当前代码的只读审计结论

实现前重新确认真实源文件路径，但按以下角色理解已有代码：

| 现有文件 | 可借鉴内容 | 严禁误用 |
|---|---|---|
| `utils.py` | `get_base_points`、`get_oriented_points_and_views`；用于坐标一致性测试 | 不能仅靠其逐点坐标完成论文 A-line 渲染 |
| `slice_renderer.py` | 当前 NeUF 逐点强度查询基线 | 它不是论文物理渲染器 |
| `slice_render_ray.py` | 可参考张量分块写法 | 它沿每个像素视向做厚度采样/平均，不是沿每条超声 A-line 从探头到深度的能量累计；不得作为论文实现 |
| `nerf_network.py` | 当前 MLP、哈希编码、checkpoint 格式 | 当前输出为图像强度和 `log_sigma`，不是论文物理字段；不得在原文件上改 |
| `dual_freq_encoder.py` | 已有低/高频哈希编码实验 | gate、训练进度激活和字段混合并非已核验的论文方法，只能作为后续自定义消融 |
| `kronecker_encoder.py` | 三平面各向异性编码实验 | 不是目标论文的多尺度哈希复现 |
| `sagittal_supervision.py` | MATLAB 图像加载、SE(3) 可微位姿细化 | 它默认把 sagittal 图像初始化为中央训练 B-scan 的几何，不能不经核验直接代表真实 sagittal 平面 |
| `test_pose_refinement.py` | SE(3) 单元测试思路 | 不要复制旧训练器依赖 |
| `test_sagittal_supervision.py` | MATLAB 转置、归一化、梯度测试 | 不足以证明 sagittal 平面物理位置正确 |
| `visualize_probe_x_axis.py` | 四元数顺序、局部轴可视化 | 必须进一步确认 `+local-x`/`-local-x` 哪个方向从探头指向组织 |
| `recons3D.py` | sagittal 参考建立的曲线几何、`X_transform=[Cx,Ct,vx,vt,delta,omega,theta]`、KNN 插值 | 这是传统重建/几何参考，不是论文神经渲染器 |
| `recons3d_exact_from_saved.py` | 读取 `infos.json`、导出 `frame_positions_mm`、`frame_rotmats`、`depth_axis_mm`、`sag_axis_mm`、`X_transform` 等 | 不要把像素索引空间与毫米世界坐标混用 |
| `export_knn_baseline.py` | 公平 KNN/IDW 基线和公共网格 | 不能把 train+validation 全部混入后再评价 held-out 切片 |
| `recons3D.py` / `recons3d_exact_from_saved.py` | sagittal 图像帮助估计旋转速度与扫描曲线 | 该几何先验必须单独记录，不能偷偷并入严格论文复现 |

特别注意当前坐标约定：`get_oriented_points_and_views` 将局部像素写成 `(depth, lateral, 0)`，再由旋转矩阵映射到世界坐标；现有 `viewdirs` 使用 `-local-x`。论文渲染所需的波传播方向通常应从探头指向组织。必须通过探头轨迹与切片平面可视化确定正负号，不能直接沿用 `viewdirs`。

---

## 4. 新项目推荐结构

```text
geometry_aware_3dus_repro/
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── configs/
│   ├── data.yaml
│   ├── ga_paper.yaml
│   ├── ga_paper_sag.yaml
│   ├── baseline_point_inr.yaml
│   ├── baseline_knn.yaml
│   └── ablations/
├── docs/
│   ├── paper_spec.md
│   ├── data_contract.md
│   ├── coordinate_convention.md
│   └── experiment_protocol.md
├── provenance/
├── src/ga3dus/
│   ├── data/
│   │   ├── adapter.py
│   │   ├── schema.py
│   │   └── splits.py
│   ├── geometry/
│   │   ├── frames.py
│   │   ├── scanlines.py
│   │   └── sagittal.py
│   ├── encoding/
│   │   └── multiscale_hash.py
│   ├── fields/
│   │   └── tissue_field.py
│   ├── rendering/
│   │   ├── energy.py
│   │   ├── reflection.py
│   │   ├── scattering.py
│   │   ├── psf.py
│   │   └── renderer.py
│   ├── training/
│   │   ├── losses.py
│   │   └── trainer.py
│   └── evaluation/
│       ├── image_metrics.py
│       ├── speckle_metrics.py
│       └── coverage.py
├── scripts/
│   ├── export_from_neuf.py
│   ├── audit_dataset.py
│   ├── visualize_geometry.py
│   ├── train.py
│   ├── render_heldout.py
│   ├── export_volume.py
│   └── evaluate.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_coordinate_parity.py
│   ├── test_scanline_direction.py
│   ├── test_impedance_gradient.py
│   ├── test_reflection_phantom.py
│   ├── test_scattering_statistics.py
│   ├── test_psf.py
│   └── test_sagittal_geometry.py
└── outputs/                  # 全部忽略，不提交大文件
```

建立独立 Git 仓库。在骨架、数据适配、渲染器、实验结果四个阶段分别形成清晰提交；不要操作源项目 Git。

---

## 5. 数据导出与中立数据契约

### 5.1 只读导出

编写 `scripts/export_from_neuf.py`，只读加载：

- `baked_dataset.pkl`；
- `infos.json`；
- 原始/裁剪后的 B-mode 帧；
- train/validation 帧编号；
- ROI 尺寸、偏移和像素间距；
- 四元数顺序与 `reverse_quat`；
- sagittal 图像及其 MATLAB 变量名；
- 如果可用，`recons3d_exact_from_saved.py` 产生的 HDF5、`P`、`tr_coord`、`idx_sag`、`X_transform`、`xp` 和 `zp`。

导出到新项目 `data_cache/<case_id>/dataset.h5`，至少包含：

```text
images                  [N,H,W] float32
frame_ids               [N]
poses_world_from_probe  [N,4,4] float32
local_points_mm         [H,W,3] float32
world_points_mm         [N,H,W,3] float32（可选缓存）
beam_dirs_world         [N,W,3] 或 [N,H,W,3]
pixel_spacing_mm        [2]
roi_offset_mm           [2]
aabb_min_mm             [3]
aabb_max_mm             [3]
train_frame_ids
heldout_frame_ids
sagittal/image          [Hs,Ws]（若存在）
sagittal/pose_initial   [4,4]（若几何可恢复）
sagittal/metadata_json
```

同时写出 `manifest.json`，记录源路径、文件大小、修改时间、图像形状、强度范围、单位、坐标轴定义和代码版本。不要把患者身份信息复制到日志或公开仓库。

### 5.2 强度预处理

- 明确输入是原始包络、线性 B-mode、还是已经 log-compressed 的 8-bit 图像。
- 当前代码倾向于把大于 1 的图像除以 255；不要对每张切片单独 min-max 归一化，否则会破坏跨切片的幅度关系。
- 训练和所有基线必须使用同一全局归一化规则。
- 保存 `intensity_transform` 及其逆变换。
- 如果只能使用 log-compressed B-mode，则论文中的声阻抗、衰减和散射字段只能解释为**有效/表观物理参数**，不得声称获得了定量组织声阻抗。

### 5.3 坐标一致性单元测试

随机选择至少 100 个帧-像素组合，比较新适配层与旧 `get_base_points` + `get_oriented_points_and_views` 的世界坐标：

```text
max absolute error < 1e-5 mm
```

另外验证：

- 横向相邻像素的世界距离等于横向像素间距；
- 深度相邻像素的世界距离等于轴向像素间距；
- 旋转矩阵正交且行列式接近 1；
- 随深度增加，采样点远离探头，而不是朝探头外移动；
- 第一帧、中央帧和最后一帧的平面位置与 `visualize_probe_x_axis.py` 一致。

坐标测试不通过时禁止训练。

---

## 6. 先量化“几乎无重复点”的数据条件

论文和 Ultra-NeRF 的视角依赖分解依赖重叠视图。用户的数据主要用于插值，切片之间几乎没有完全重复的空间点。因此必须先做可辨识性审计，而不是直接长时间训练。

### 6.1 空间覆盖统计

在若干尺度下体素化全部训练采样点，例如：

```text
0.25 mm, 0.5 mm, 1.0 mm, 2.0 mm
```

对每个占用体素计算：

- 落入其中的不同帧数；
- 不同扫描线/视角数；
- 波束方向的最大夹角、均值和标准差；
- 到最近训练平面的距离；
- 是否被 sagittal 平面覆盖。

输出：

```text
outputs/audit/coverage_summary.json
outputs/audit/view_count_histogram.png
outputs/audit/angular_span_histogram.png
outputs/audit/coverage_volume.mhd
outputs/audit/geometry_overview.png
```

### 6.2 解释规则

- “空间点不完全相同”不等于完全没有信息，因为连续隐式场可以通过邻域插值学习空间结构。
- 但如果大多数局部邻域只有单一方向，模型无法仅凭图像唯一分辨“界面反射”和“随机散射”；多个参数组合都可能渲染出相似强度。
- 若中位数视角数接近 1 或局部角度跨度很小，继续做插值实验，但在报告中把反射/散射分解标记为**弱可辨识**，不能用漂亮的参数图作为物理正确性的证据。
- sagittal 参考会增加一个重要的正交/交叉平面约束，但一张参考图仍不能代替论文中的多角度重叠 sweep。

将这些判断写入 `docs/data_identifiability.md`。

---

## 7. 正确建立超声扫描线，而不是普通相机射线

### 7.1 线阵 B-mode 几何

每个 B-mode 横向列对应一条 A-line。对帧 `k`、横向列 `u`、深度样本 `v`：

```math
\mathbf{x}_{k,u,v}=\mathbf{o}_{k,u}+t_v\mathbf{d}_{k,u}.
```

对于当前局部坐标约定，`depth` 位于局部第一轴，`lateral` 位于局部第二轴。需要从位姿矩阵构造：

- 每条 A-line 在成像平面顶部的起点 `o[k,u]`；
- 从探头进入组织的单位方向 `d[k,u]`；
- 与真实轴向像素间距一致的 `t_v`。

如果探头并非线阵或数据已进行扇形成像，必须根据真实探头几何重建每条射线，不能强行使用平行 A-line。

### 7.2 与现有 renderer 的本质区别

- `slice_renderer.py`：在每个 B-mode 像素所在的单个三维点直接预测强度；这是当前 NeUF baseline。
- `slice_render_ray.py`：从每个像素点向切片厚度方向采样并求平均/alpha；这不是目标论文的传播模型。
- 新 renderer：一次处理完整 A-line 或完整小块图像，按深度顺序累计能量，并在反射/散射图上执行论文规定的 PSF 卷积。

禁止为了复用旧接口而退化成逐点独立预测。

---

## 8. 论文模型实现要求

### 8.1 物理字段网络

建立 `TissueField`，输入仅为空间坐标以及论文明确要求的量。不要把 viewing direction 直接送入任意强度 MLP，除非原文明确这样做。视角依赖应主要由几何法向、波束方向和物理渲染产生。

输出头、激活函数和 coarse/fine 特征连接必须来自 `paper_spec`。至少应能导出并可视化：

- 有效衰减；
- 声阻抗；
- 声阻抗梯度与单位法向；
- 几何反射相关量；
- 散射密度；
- 散射幅度；
- 最终反射分量、散射分量与合成回波。

所有字段必须进行有限值检查。声阻抗梯度需保持对坐标可微：

```python
x.requires_grad_(True)
z = field.query_impedance(x)
grad_z = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
normal = grad_z / (grad_z.norm(dim=-1, keepdim=True) + eps)
```

训练时不能意外 `detach` 声阻抗或坐标梯度。

### 8.2 多尺度哈希编码

- 严格复现论文的尺度数、每层分辨率、特征数、哈希容量和字段分支。
- 不要把现有 `DualFreqEncoder` 的 gate、`hf_activate_ratio` 或 `hf_max_weight` 默认加入 GA-Paper。
- 如果论文的 exact encoding 无法核验，则先实现接口与单元测试，配置保持 `unknown`，不要擅自用当前参数填充。
- 为公平比较，所有实验记录有效参数量、显存、训练时间和每帧渲染时间。

### 8.3 反射

反射必须显式依赖：

- 相邻位置的声阻抗关系；
- `∇Z` 得到的界面法向；
- 波传播方向；
- 论文定义的界面/边界概率或连续权重；
- 剩余能量和 PSF。

建立以下合成测试：

1. 均匀声阻抗场：内部反射应接近零；
2. 单个平面阶跃界面：反射集中在界面；
3. 改变界面法向而保持声阻抗差不变：反射应按论文的角度规律变化；
4. 翻转射线方向：结果应符合公式的方向约定；
5. 梯度和 loss 对网络参数均有限且非零。

### 8.4 散射与斑点

训练中的 Bernoulli/Normal 采样、重参数化或连续近似必须来自论文或官方代码。评估时：

- 固定随机种子；
- 同时保存随机渲染和期望值/确定性渲染（如果公式允许）；
- 不允许不同方法各使用一次随机斑点图后直接比较；
- 至少用 3 个种子报告均值与标准差；
- 保存 `rho_s`、`phi`、散射模板、PSF 后散射和最终散射回波。

### 8.5 PSF

- 使用论文的 2D PSF，而不是任意高斯模糊。
- 明确轴向/横向核尺寸、像素单位与毫米单位的换算。
- 用 delta impulse 测试确认卷积方向、中心、padding 和核归一化。
- 所有方法评价时不得额外使用不同的后处理平滑。

### 8.6 损失

严格论文配置优先。若论文沿用 Ultra-NeRF 的 SSIM+L2 思路，也必须从目标论文核验其权重，不直接沿用 Ultra-NeRF 的 `lambda=0.9`。

对用户数据可另外建立适配损失，但必须置于 `ga_paper_sag.yaml`：

```math
\mathcal{L}=\mathcal{L}_{\mathrm{dynamic}}
+\lambda_{\mathrm{sag}}\mathcal{L}_{\mathrm{sag}}
+\lambda_{\mathrm{pose}}\mathcal{L}_{\mathrm{pose}}
+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{trajectory}},
```

任何新增正则都要单独消融。

---

## 9. sagittal 参考图像的正确使用

用户的 sagittal 图像是本数据集中最重要的参考，并参与过探头旋转速度/扫描曲线估计。它不能仅作为一张普通“额外切片”随意塞入训练。

### 9.1 先恢复 sagittal 几何

优先使用已有重建信息：

- `P`；
- `tr_coord`；
- `idx_sag`；
- 人工对应点；
- `X_transform=[Cx,Ct,vx,vt,delta,omega,theta]`；
- `xp`、`zp`；
- `frame_positions_mm` 和 `frame_rotmats`。

输出 sagittal 平面、动态序列所有切片、探头轨迹、波束方向和对应点的统一三维可视化。先确认：

- sagittal 平面是否真的与预期扫描方向相交；
- 单位是否统一为毫米；
- 旋转速度 `omega` 的正负号和 frame index 定义；
- sagittal 像素轴与世界坐标轴的对应；
- MATLAB 转置是否只发生一次。

### 9.2 位姿细化

若 sagittal 完整 6-DoF 位姿不确定：

1. 用现有几何结果初始化；
2. 只优化一个全局 sagittal SE(3) 增量，而不是每个像素或每行独立位姿；
3. 对平移、旋转和与扫描曲线的交点加入明确先验；
4. 初始位姿、细化位姿、参数变化和 overlay 全部保存；
5. 对 `optimize_pose=false/true` 做消融；
6. 如果自由优化导致位姿大幅漂移以“解释”图像强度，则判定不可接受。

不要直接沿用 `SagittalSliceSupervisor.from_dataset()` 将其初始化为中央动态 B-scan 的假设，除非通过几何资料证明这就是实际 sagittal 平面。

### 9.3 防止评价泄漏

- **GA-Paper**：不使用 sagittal 图像训练；若其位姿可靠，可作为独立跨平面评价。
- **GA-Paper+Sag**：使用 sagittal 图像训练后，不能再把同一整张图像的 SSIM/PSNR 当作测试成绩。主评价仍使用 held-out 动态切片，或预先固定 sagittal 的训练/测试遮罩区域。
- 两条实验线的表格必须分开。

---

## 10. 面向插值的无泄漏数据划分

不要随机拆分像素，也不要把相邻像素或同一帧部分像素同时放入训练和测试来声称三维插值有效。

至少建立以下协议：

### Protocol A：规则稀疏切片插值

- 从完整动态序列中每隔 `s` 张保留一张训练切片；
- `s ∈ {2, 4, 8}`，由帧数和物理间距决定；
- 其余整张切片作为测试；
- 评价结果按“测试切片到最近训练切片的物理距离”分箱。

### Protocol B：连续缺失块

- 预先固定至少 3 个连续缺失区间；
- 每个区间可取 3、5、9 张或等效物理厚度；
- 测试模型对真实空间空洞的插值，而不是仅补相邻一张。

### Protocol C：论文式新视角（仅数据支持时）

- 若存在真正不同角度且覆盖同一区域的 sweep，按整条 sweep 留出；
- 若不存在，不要把相邻旋转切片错误描述成论文式 novel-view 验证。

每个 split 固化为 JSON，所有方法使用完全相同的 split、AABB、图像预处理和评价 mask。

---

## 11. 必须比较的方法

最小公平比较集合：

| ID | 方法 | 目的 |
|---|---|---|
| B0 | KNN/IDW | 传统局部插值下限，使用与神经方法一致的训练切片和网格 |
| B1 | Current NeUF point-INR | 当前逐点强度模型；冻结原实现和配置，不在旧项目上修改 |
| B2 | Single-scale physics renderer | 去掉多尺度哈希，检验编码贡献 |
| B3 | No-geometry reflection | 去掉声阻抗梯度/法向，检验 geometry-aware 贡献 |
| M0 | GA-Paper | 严格目标论文复现 |
| M1 | GA-Paper+Sag | 加入 sagittal 几何和图像监督的数据集适配 |

可选消融只能在上述最小集合完成后进行：

- 无 PSF；
- reflection-only；
- scattering-only；
- sagittal 固定位姿/细化位姿；
- 现有 DualFreq gate；
- Kronecker 编码。

后两项是用户现有创新探索，不属于论文复现。

---

## 12. 评价指标与可解释性边界

### 12.1 held-out 整张切片

在统一有效 mask 内计算：

- MAE / MSE；
- PSNR；
- SSIM；
- NCC；
- 可选 LPIPS，仅作为补充，不把自然图像感知指标当作主要医学超声证据。

报告每张切片、每个缺失距离分箱、每个病例和 3 个随机种子的结果。不要只报告最佳 checkpoint 的单个数字。

### 12.2 斑点保持

用户目标包含保留/区分 speckle。若存在可靠的均匀组织 ROI，计算：

- speckle SNR：`mean/std`；
- 局部 coefficient of variation；
- 强度直方图距离（Wasserstein/KS）；
- 轴向与横向自相关函数及半高宽；
- 2D 功率谱或径向功率谱差异。

ROI 必须预先定义并对所有方法一致。没有均匀 ROI 时，不能用全图 speckle 统计得出强结论。

### 12.3 结构与反射

- 保存边缘强度、梯度方向与 `∇Z` 法向的一致性图；
- 对可见大界面比较 angle-dependent reflection；
- 保存 reflection/scattering 分量，但明确它们是 latent decomposition；
- 没有物理场 ground truth 或充分多角度观测时，不把分解图解释为真实声阻抗或真实散射密度。

### 12.4 sagittal 一致性

若 sagittal 未参与训练且位姿可靠：

- 渲染相同平面；
- 在重叠有效区域计算 NCC、SSIM、MAE；
- 显示 target、prediction、absolute error、边缘 overlay。

若 sagittal 参与训练，则只用于展示拟合与几何稳定性，主要测试结论来自未见动态切片。

### 12.5 计算成本

报告：

- 总参数量与哈希表大小；
- 峰值 GPU 显存；
- 固定硬件上的训练时间；
- 单帧渲染时间；
- 指定体素网格的导出时间；
- checkpoint 与体数据大小。

---

## 13. 实施阶段与硬门槛

### Phase 0：隔离与资料清单

交付：新仓库骨架、源项目 before 状态、依赖环境、`provenance/manifest.json`。

**门槛**：确认新目录不属于旧仓库。

### Phase 1：论文规格

交付：`docs/paper_spec.md`、`configs/paper_spec.yaml`、公式到代码函数的映射表。

**门槛**：核心反射公式、多尺度连接、PSF 和训练设置均为 `verified`，否则只能进入 fallback 路线。

### Phase 2：数据与几何

交付：中立 HDF5、数据契约、坐标 parity 测试、三维几何图、覆盖/角度审计。

**门槛**：坐标误差、单位、射线方向和 sagittal 位置检查通过。

### Phase 3：解析合成测试

交付：均匀介质、单阶跃界面、散射-only、PSF impulse 等最小 phantom 测试。

**门槛**：反射位置、角度响应、能量累计、随机统计和梯度均符合预期。

### Phase 4：小规模 smoke training

使用少量帧、低分辨率和几百/几千步：

- loss 有下降；
- 无 NaN/Inf；
- 反射与散射均非全零/全一；
- checkpoint 能恢复；
- 固定种子渲染可重复。

**门槛**：不得在 smoke test 失败时启动完整 GPU 任务。

### Phase 5：正式实验

先跑一个固定 split 和一个种子的 B0/B1/B2/B3/M0；确认评价脚本一致后，再扩展到 3 个种子与 M1。

### Phase 6：报告

生成：

```text
outputs/<run_id>/config_resolved.yaml
outputs/<run_id>/environment.txt
outputs/<run_id>/checkpoints/
outputs/<run_id>/renders/
outputs/<run_id>/fields/
outputs/<run_id>/metrics_per_frame.csv
outputs/<run_id>/metrics_summary.json
outputs/<run_id>/timing.json
outputs/comparison/summary.csv
outputs/comparison/figures/
REPORT.md
```

---

## 14. 关键单元测试的预期行为

### 14.1 数据

- MATLAB v7.3 sagittal 图像只转置一次；
- uint8 `[0,255]` 转为 `[0,1]`；float `[0,1]` 不再缩放；
- train/heldout 帧无交集；
- 每个帧编号能追溯到原文件。

### 14.2 几何

- 新旧世界坐标一致；
- 波束方向与深度增大方向点积为正；
- A-line 上的采样点按物理深度单调排列；
- sagittal 平面与已知对应点误差在预设容差内。

### 14.3 物理渲染

- 常量声阻抗：`||∇Z||≈0`，界面反射≈0；
- 阶跃声阻抗：反射峰位置误差不超过一个深度采样；
- 只有散射：反射分量接近零；
- 只有反射：散射分量接近零；
- PSF delta 测试恢复预期核；
- 减小步长时结果收敛，而不是明显漂移；
- `loss.backward()` 后所有预期网络分支具有有限梯度。

### 14.4 可复现性

- 同一 config、checkpoint、seed 的评价渲染逐像素一致或在明确容差内；
- resolved config 包含所有默认值；
- checkpoint 包含 optimizer、scheduler、scaler、迭代数、随机状态与数据 split 哈希。

---

## 15. 常见失败模式与禁止的“假成功”

1. **把逐点强度网络叫作论文复现**：不接受。
2. **使用 `slice_render_ray.py` 的 thickness mean 作为物理 A-line 累积**：不接受。
3. **随意用高斯模糊代替论文 PSF**：不接受。
4. **直接把 `-local-x` 当传播方向，不检查符号**：不接受。
5. **每张切片单独归一化后比较**：不接受。
6. **随机拆像素造成 train/test 泄漏**：不接受。
7. **用参与训练的 sagittal 整图报告测试 SSIM**：不接受。
8. **数据只有单一局部视角，却声称物理上成功分离反射与散射**：不接受。
9. **用 latent 声阻抗图宣称定量声阻抗**：不接受，尤其输入为 log-compressed B-mode 时。
10. **没有 PDF/官方代码仍猜测公式并标注 exact reproduction**：不接受。
11. **只展示最好的一次随机 speckle 渲染**：不接受。
12. **在当前 NeUF 项目中直接改 renderer/network 以省事**：不接受。

当模型能重建 held-out 图像但字段分解随随机种子变化很大时，应报告为“图像重建可行、物理解耦不可辨识”，而不是继续调图直到看起来合理。

---

## 16. 最终验收标准

任务只有同时满足以下条件才算完成：

- [ ] 新项目为独立目录和独立 Git 仓库；
- [ ] 源 NeUF 项目 before/after 状态一致，无新增修改；
- [ ] 论文核心公式和超参数有页码/公式号/代码来源；
- [ ] 数据中立导出和坐标 parity 测试通过；
- [ ] 波传播方向、A-line 和 sagittal 几何经过可视化确认；
- [ ] 合成物理测试全部通过；
- [ ] GA-Paper 与 GA-Paper+Sag 结果完全分开；
- [ ] 所有方法使用相同的整帧 held-out split、预处理、mask 和网格；
- [ ] 至少完成 KNN、Current NeUF、No-geometry、GA-Paper 的最小比较；
- [ ] 至少 3 个随机种子或明确说明计算限制；
- [ ] 报告插值距离分箱结果，而不仅是总体平均；
- [ ] 报告 speckle/结构指标及其适用 ROI；
- [ ] 明确区分“图像拟合”“插值能力”“novel-view 能力”和“物理字段可辨识性”；
- [ ] 保存完整配置、环境、checkpoint、逐帧指标、运行时间和失败日志；
- [ ] `REPORT.md` 能让另一名研究者按命令复现实验。

---

## 17. Codex 每个阶段的汇报格式

每完成一个 Phase，只汇报以下内容，不要用模糊表述：

```text
Completed:
- 实际创建/修改的文件
- 已通过的测试和数值容差

Verified:
- 来自论文或数据的已核验事实

Uncertain/Blocked:
- 尚未核验的公式、路径、数据或几何

Next command:
- 下一条最小、可验证的命令
```

如果长训练尚未运行，必须写“implemented but not experimentally validated”；不要将代码能启动描述为复现成功。

---

## 18. 第一轮立即执行清单

Codex 接到本文件后，先执行以下工作，不要直接启动训练：

1. 找到源 NeUF 根目录和数据路径，保存源仓库状态；
2. 在源项目外创建 `geometry_aware_3dus_repro` 独立项目；
3. 搜索目标论文 PDF/官方代码并填写 `paper_spec`；
4. 只读检查上述已有文件，建立代码角色清单；
5. 导出一个病例到中立 HDF5；
6. 完成坐标 parity、传播方向和 sagittal 几何可视化；
7. 完成空间重叠和角度覆盖审计；
8. 向用户报告已核验事实、真正缺失的信息和是否具备进入 GA-Paper 编码的条件；
9. 只有上述硬门槛通过后，才实现并测试物理渲染器。

