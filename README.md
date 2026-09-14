# Neural Ultrasound Field (NeUF)

新增的 `neuf.edge_field` 是独立的 V0–V2 实验路径：原始 B-mode 监督灰度，
固定的传统 NLSTV λ=0.009 response 监督边缘，并在 V2 中矫正训练帧位姿。
它不使用 Neural STV，也不采用 anatomy/residual 分解。下文原有固定几何
模型及其 checkpoint 接口保持不变。

## 仅用 teacher response 的三维 edge 与位姿修正

`--variants EdgeFixed EdgePose` 使用单输出坐标场，直接学习传统 NLSTV
response；不创建灰度头，原始 B-mode 仅用于核验 teacher 的帧编号和方向。
`EdgeFixed` 固定初始位姿，`EdgePose` 优化训练帧的 SE(3) 修正。两者具有相同
网络初始化、response 像素批次和场更新预算；位姿更新单独计数。

`--plane-resolutions 32 64 128 256 512`（qsub 环境变量 `PLANE_RESOLUTIONS`）
在 Fourier 坐标特征之外加入五级 XY/XZ/YZ 可学习特征平面，每张 4 通道，
经双线性查询拼接后输入 MLP；高分辨率平面逐步开放，位姿可通过其坐标导数更新。
不设置时保留纯 MLP。纯 MLP 的首轮中期结果仍把细边缘拟合成宽响应块，
因此停止该轮并增加局部空间特征；记录保留在 `logs/20260910_train02/`。

场损失包含前景/背景均衡的原生 response 保真、局部归一化相关及 response
梯度保真。它们均不使用灰度。前 35% 固定位姿；35%–70% 期间，当最近 100 步
局部相关损失均值小于 0.8 时，每 5 步冻结场并更新一次位姿；最后 30% 冻结
位姿，直接拟合未平滑 teacher。位姿目标仅含平滑 response 匹配、局部相关和
修正先验。就绪阈值是训练启发式，不证明几何正确；独立位姿采样批次也不等同
留帧配准。首个训练帧固定，验证/测试帧位姿始终固定。

复用 `qsub/neuf/edge_field.sh`，以 `VARIANTS=EdgeFixed EdgePose`、`MODE=train`
选择此路径；`MODE=smoke,STEPS=10,WIDTH=64,PATCH_SIZE=48,POSE_EVERY=1`
使用同一入口的轻量检查。新 checkpoint schema 为 `nlstv_response_field_v1`，
独立世界坐标查询返回 `[...,1]`；旧 V0–V2 checkpoint 仍返回 `[...,2]`。

```
neuf/edge_field/
  model.py                单/双通道场与 SE(3)，旧权重兼容
  losses.py               分开的 response 与旧灰度损失
  workflow.py             训练、轻量检查、独立推理
  evaluation.py           共用查询和 NPY/MHD 体导出
  response_evaluation.py  原生分辨率 response 对照与三维切面/投影
```

结果保存在工作区 `logs/<RUN_ID>/cerebral/index_all/{EdgeFixed,EdgePose,comparison}/`：
`predictions/edge_volume.npy` 与 `.mhd/.raw` 为三维 response，数组轴顺序 Z/Y/X，
MHD 坐标为毫米；`volume_support.npy` 根据本模型修正后的训练位置近似标记
观测覆盖，覆盖外预测未经观测约束。体素间距是导出采样间距，不是已验证的
空间分辨率。模型间三维可视化使用共同覆盖区域。
`poses.npz` 保存输入及修正后位姿；固定帧 NPZ 只含 `edge/teacher/mask`。
`comparison/plots/response_*.png` 对比 teacher 与重建；
`edge_volume_comparison.png` 展示固定世界坐标三切面及最大投影。

所有 response 指标使用原生像素网格。0.15/0.3/0.5 阈值下的支持区域距离、
precision、recall 用于识别缺失和扩宽，不能当作人工解剖边界精度。
无预测或无 teacher 支持时标为无效，并报告有效计数；不沿用灰度阶跃宽度拟合。
response 的跨帧差异和结构场/位姿相互适应仍可能影响几何，目前无独立真实位姿。

项目目标与测试原则：优先检查三维结构连续性、原生 response 细节与位置一致性。
仅扩展已有轻量检查，验证灰度独立性、单通道、SE(3) 梯度、固定参考帧及权重重载；
测试通过不能证明重建质量改善。效果必须由固定图像和固定/修正位姿对照评价。

## NLSTV teacher response：V0–V2

项目目标与测试原则：以重建结构清晰度、对比保持和三维几何一致性为目标。
只在已有 `phase1_smoke` 入口增加一个可选路径，确认数据方向、毫米坐标、
SE(3) 零点梯度、训练与 checkpoint 重载；smoke 通过不能证明图像改善。
正式对照固定单 seed、划分、采样和网络更新数，需审查固定原生分辨率切片、
边界剖面和世界坐标重切片。尚未验证真实位姿精度，没有干净图像或解剖标签真值。

```text
neuf/edge_field/
  data.py         原图/teacher 对齐核验、物理坐标、共享扇区与冻结划分
  model.py        渐进 Fourier MLP、灰度/边缘输出、SE(3) 修正及独立加载
  losses.py       原图鲁棒灰度、多尺度 response、边界一致性
  workflow.py     V0–V2 等网络更新预算训练、已有 smoke 的新路径
  evaluation.py   固定切片、剖面、位姿变化与 float32 体数据导出
```

所有版本从同一初始化开始，使用同一采样序列；默认每版 12000 次网络更新，
每次 4 个 64×64 patch。共享 MLP 为 4×128，10 个渐进 Fourier 频带。
V0 只使用原图灰度损失；V1 加入 teacher response 与边界一致性，位姿固定；
V2 在 20%–70% 的训练区间每 5 步追加一次位姿更新，最后 30% 固定位姿拟合。
位姿更新次数与额外耗时独立记录，因此这不是严格等墙钟预算比较。

teacher 的 MATLAB v7.3 `[W,H,N]` 明确转为 `[N,H,W]`。逐帧核验 `images.npy`
与 baked 灰度在共享扇区内完全一致（浮点容差 1e-6）；源转换会清零扇区外内容，
因此不把外部显示内容作为不一致。所有帧的 response/hash 被记录，但只采样
训练 split 做损失；验证/test 仅用于评价，位姿始终保持输入值。
灰度固定为 uint8/255，response 使用已有逐帧 p99.5 归一化值；不将后者
解释为物理梯度幅值或标定置信概率。有效域向内腐蚀 10px，排除扇区轮廓。

局部坐标沿用 NeUF 的 `(Y axial, X lateral, 0)`，单位 mm；原始位姿为
`world = R @ local + t`。修正是在固定场景中心处左乘 SE(3)，输出转换回原世界坐标。
固定第一张训练帧消除整体坐标自由度，保留位姿幅度和平滑先验；平移生成元
每轴有界 ±2mm，旋转生成元每轴有界 ±2°，这不是最终探头原点位移的硬上限。
固定图像外参和尺度不参与估计。

通过工作区脚本执行，提交前创建对应 qsub 日志目录，且使用未占用的 RUN_ID：

```bash
qsub -v RUN_ID=YYYYMMDD_trainNN,MODE=smoke \
  -o /misc/raid/zchen/Code/qsub/logs/neuf/YYYYMMDD_trainNN/stdout.log \
  -e /misc/raid/zchen/Code/qsub/logs/neuf/YYYYMMDD_trainNN/stderr.log \
  /misc/raid/zchen/Code/qsub/neuf/edge_field.sh
```

正式训练将 `MODE=smoke` 改为 `MODE=train`；可传 `STEPS`、`PATCHES`、
`PATCH_SIZE`、`SEED`、`WIDTH`、`LR`。结果位于工作区 `logs/RUN_ID/cerebral/index_all/`，
其中 `V0/`、`V1/`、`V2/` 保存 checkpoint、metrics、plots、predictions；
`comparison/` 保存横向比较。训练中点与终点保存固定验证图和指标，test 仅在终点评价。
原图是含噪观测，不称为干净 GT。剖面候选仅依据原图选择；无可靠阶跃时宽度为空。
全部 validation/test 的主指标使用每 4px 采样网格；固定四帧及剖面另在原生
分辨率评价，不能将两种分辨率的 SSIM/梯度指标直接混比。V0 的边缘头不受
监督，单模型图中该面板只保留显示布局，不用于判断边缘重建质量。
固定比较沿用 236/226/216/206 和 xywh=[272,407,128,128] ROI；另输出一张
训练集按帧编号排序后的中央帧，标为 training diagnostic，用于区分欠拟合与泛化。
首轮 `20260909_train05` 使用默认 128 通道、12000 步；验证图明显偏平滑。
第二轮 `20260909_train06` 使用 `WIDTH=256,STEPS=40000,LR=0.001`，其余设置
相同。两轮都保留，第二轮的最终质量需以根 notebook 中实际图像与指标判定。

2026-09-09 实测：第二轮三个模型均完成 40000 步，V2 额外完成 4000 次位姿
更新。24 张验证帧的每 4px 网格平均 MSE 为 V0=0.001169817、V1=0.000927616、
V2=0.000979851，平均 SSIM 分别为 0.72237、0.76219、0.75703。V1 在该固定
单 seed 对照中改善了观测拟合和部分结构保留；V2 暂未超过 V1。V2 的训练帧
平均修正为 0.27767mm、0.25643°，没有独立真实位姿，不能据此声称精度提高。
ROI 仍明显缺少细节，尚未达到清晰、窄边界目标。

剖面复核排除了 10–90% 过渡不能完整落入两侧平台之间 [-8,8]px 的拟合。
本轮原始 `comparison/metrics/edge_profiles.csv` 的旧 valid 标记没有该条件，
仅作为原始诊断保留；应使用 `logs/20260909_train08/cerebral/index_all/
edge_field_report/metrics/edge_width_summary.json` 的复核有效数。12 个原图候选中，
V0/V1/V2 分别只有 0/3/3 个满足完整过渡条件，没有全模型共同有效的候选，
因此不报告配对平均宽度或宣称边缘收窄。后续运行已在生产测量函数中加入该条件。

`predictions/poses.npz` 保存原始帧编号、初始/修正后的 4×4 世界位姿和训练索引。
`pose_changes.csv` 是修正幅度，不是对真实姿态的误差。`volume_float.npy` 为
float32 `[z,y,x]`，`.mhd/.raw` 的 origin/spacing 按 x/y/z，默认 0.75mm。
`volume_support.npy` 为距离下采样原始训练点不超过 1.5mm 的近似支持掩膜，
体数据外推区域不能当作可靠组织。所谓 sagittal 比较为固定世界 x 平面，
没有解剖方向标签时不声称它等于标准解剖矢状面。

独立推理入口为 `python -m neuf.edge_field.workflow --checkpoint MODEL.pt
--points world_points.npy --output prediction.npy`（仍通过 qsub 执行）。
输入最后一维为毫米世界坐标，输出最后一维为 `[gray, response]`，无需 teacher。
新 schema 为 `nlstv_edge_field_v1`，不与旧 `NeRF` checkpoint 混用。
checkpoint 保存网络、优化器、修正位姿、物理网格和数据签名；当前入口不提供断点续训。

NeUF 原有主线保留固定几何下的 HashGrid 重建：一个最基础的单头
`HASH` 基线，以及当前 Phase 1 的 `DUAL_HASH` 图像质量方案。旧的
Kronecker、Fourier、Loupas、不确定性头、pose/sagittal 优化、
Ultra-NeRF、KNN/Recons3D 和 GUI 路线均不再属于当前代码。

## 项目目标与测试原则

本项目以提高最终医学超声三维重建质量为最高目标，重点关注解剖结构
清晰度、几何一致性、散斑保留与区分能力、对比度、插值连续性和伪影
抑制。测试只用于确认代码能够运行并防止数据方向、固定几何、梯度传播
和关键渲染路径回归；不创建大量重复或一次性的测试，整个项目最多保留
一个轻量级端到端 smoke test。测试通过不代表重建质量提高。任何方法改动
必须在固定数据、训练配置和重建范围下提供前后图像及定量指标对比；未
观察到明确改善时必须标记为“尚未验证”或“未观察到改善”。

## 保留的模型

| 配置 | Encoder | Decoder | 用途 |
|---|---|---|---|
| `legacy_fixed_geometry` | 单个普通 `HASH` | 单输出 MLP | Phase 1 基线 E0，无渐进高频，masked MSE |
| `dual_single_head_matched` | raw low/high `DUAL_HASH`，无 gate | 与 E0 相同的单输出 MLP | Phase 1 编码器对照 E1 |
| `anatomy_speckle_v1` | 无 gate 的 low/high `DUAL_HASH` | `low→A`，`concat(A, high)→S` | 当前方法 E2，联合 masked MSE |
| `frozen_stv_components_v1` | E1 同款无 gate `DUAL_HASH` | E1 的 8×256 主干，三通道输出 | 冻结 Neural STV 监督三维结构、边界和残差 |

E2 从第一个 step 开始同时训练 low/high encoder 和两个 decoder head：

```text
A = anatomy_head(feat_low)
S = speckle_head(concat(A, feat_high))
I(alpha) = anatomy + alpha * speckle,  alpha in [0, 1]
```

`alpha=1` 是完整重建，`alpha=0` 只输出 anatomy。几何和探头位姿在整个
Phase 1 中保持不变。

## 目录

```text
neuf/
  dataset.py                     数据读取、物理标定与 sector mask
  hash_encoder.py                基础 HashGrid
  dual_freq_encoder.py           当前 low/high 双 HashGrid
  nerf_network.py                四个固定几何 field head
  frozen_stv.py                  训练专用的冻结 teacher 与三分量监督
  main.py                        固定几何训练入口
  slice_renderer.py              逐点切片查询
  export_full_grid_from_ckpt.py  Cartesian 三维导出
  phase1_*.py                    Phase 1 split、loss、评价与 smoke
jobs/pbs/
  run_basic_hash_pbs.sh
  run_phase1_image_quality_eval_pbs.sh
  run_export_full_grid_pbs.sh
jobs/phase_1/
  run_e0.sh
  run_e1.sh
  run_e2.sh
docs/PHASE1_IMAGE_QUALITY.md
```

数据、checkpoint、实验输出和导出体积不属于源码，均由 `.gitignore` 管理。

## 安装

```bash
python -m pip install -e .
```

现有 `requirements.txt` 是当前环境快照；在新机器上建议按 CUDA/PyTorch
版本先安装 PyTorch，再安装其余依赖。

## 准备数据

原始目录需要包含超声图像和记录探头位置的 `infos.dat`：

```bash
python -m neuf.bakeDataset \
  -i path/to/input \
  -o path/to/baked_dataset.pkl
```

## 训练基础单头 HASH

```bash
python -m neuf.main \
  --dataset path/to/baked_dataset.pkl \
  --encoding HASH \
  --field-head legacy_fixed_geometry \
  --training-mode Random \
  --points-per-iter 50000 \
  --nb-iters-max 20000 \
  --root runs/basic_hash_seed3407
```

集群任务：

```bash
qsub jobs/pbs/run_basic_hash_pbs.sh
```

## 训练与评价当前 Phase 1

冻结配置由三个独立 PBS 任务分别训练 E0、E1、E2：

其中 E0 使用普通 `HASH`；E1 使用无 gate 的 `DUAL_HASH` 原始 low/high 特征
拼接。三个模型都在 sector mask 内使用 masked MSE，所有 HashGrid level 从
训练开始共同训练。E2 没有 gate、渐进高频权重、分阶段冻结或额外的末段学习率
缩放；其最终训练输出固定为 `A + S`。

后续实验统一使用“日期 + 试验次数 / 内容”的目录命名：

```text
experiments/YYYYMMDD_trialNN/content/E0/seed3407/
```

例如 `experiments/20260901_trial01/phase1_e1_plain_dual_vs_e2_hf02/E0/seed3407/`。
三个训练脚本未传 `TRIAL_ID` 时，会按各自实验目录自动选择第一个可用编号；
也可以显式传入编号。评价必须使用训练最终选择的 `RUN_DATE`、`TRIAL_ID`
和 `RUN_CONTENT`。当前已在运行的任务保留原目录，不中途迁移。

```bash
qsub jobs/phase_1/run_e0.sh
qsub jobs/phase_1/run_e1.sh
qsub jobs/phase_1/run_e2.sh
qsub -v RUN_DATE=20260901,TRIAL_ID=02,RUN_CONTENT=phase1_e1_plain_dual_vs_e2_hf02 \
  jobs/pbs/run_phase1_image_quality_eval_pbs.sh
```

三个训练任务共享冻结 manifest。为避免共享文件系统锁竞争，建议按
E0、E1、E2 的顺序逐个提交，前一个完成后再提交下一个；每个任务会在日志
中打印自动选择的 trial，三个 checkpoint 均完成后再用该编号提交评价任务。

正式矩阵使用 `seed=[3407,3408,3409]`。当前脚本默认的单 seed 运行只适合
诊断；缺少三个 seed 时最终判定必须为 `INCONCLUSIVE`，不能写成方法通过。
完整的冻结 split、loss、指标和验收规则见
[Phase 1 说明](docs/PHASE1_IMAGE_QUALITY.md)。

唯一保留的轻量 smoke 路径为：

```bash
python -m neuf.phase1_smoke \
  --dataset path/to/baked_dataset.pkl \
  --output-dir /tmp/neuf_phase1_smoke
```

smoke 只检查 train → checkpoint → load → slice → volume query 是否连通，
不构成图像质量结论。

## 三维导出

```bash
python -m neuf.export_full_grid_from_ckpt \
  --ckpt runs/basic_hash_seed3407/latest/ckpt.pkl \
  --output exports/current_hash \
  --alpha 1.0 \
  --component intensity \
  --save-float-output
```

只有 `anatomy_speckle_v1` 支持 `component=anatomy|speckle`。MHD 是固定显示
窗的 `uint8` 文件；定量分析必须使用 `volume_float.npy`，不得对每个模型或
体积单独 min-max 后比较。

## 冻结 Neural STV 的三维分解

项目目标与测试原则：本方案面向 3D U-Net 体分割，目标是在训练一次后用
alpha 连续调节散斑外观：0 端抑制散斑并保留清晰边界，1 端高保真重建含散斑
观测。最小 smoke 只检查梯度、checkpoint 和导出链路，不能证明边界或分割质量
提高；新损失的正式训练效果尚未验证。分割收益需要人工标注和固定评价协议。

`frozen_stv_components_v1` 是独立的新场头。训练时在真实训练切片上运行冻结的
Neural STV，采用 standalone 定义：

```text
S* = A_raw - B_raw
B* = B_raw
R* = I - A_raw
I = S* + B* + R*
loss = masked_MSE(S+B+R, I)
     + λS masked_MSE(S, S*) + λB masked_MSE(B, B*) + λR masked_MSE(R, R*)
     + λA masked_MSE(A, A_raw) + λedge Ledge(A, A_raw, confidence)
     + λ3D Lspatial(A)
A = S+B
I(alpha) = A + alpha*R,  alpha in [0,1]
```

三个分量权重默认均为 1，可用 `--stv-structure-weight`、
`--stv-boundary-weight`、`--stv-residual-weight` 调节。`R` 表示包含散斑的残差，
不是经过物理标定的纯 speckle；上述定义也不同于 E3 adapter 的四分量定义。

新实验默认 `--stv-anatomy-weight 1`、`--stv-edge-weight 0.1`、
`--stv-spatial-weight 0.01`。保边项比较 teacher anatomy 在 1/2 像素跨度上的
置信度加权梯度，避开无效像素对；三维项是在 x/y/z 毫米坐标中对 anatomy
施加中心二阶差分先验，默认步长 `--stv-spatial-step-mm 0.5`、每步最多
`--stv-spatial-points 256` 个中心。teacher 置信度与 anatomy 梯度共同减弱
边界附近的平滑权重，`--stv-edge-scale 0.03` 使用 display 强度/像素单位。
这些是待验证的初始超参数，三维邻域先验不是三维真值监督。R 不受上述额外
平滑约束，但共享主干仍由所有损失共同更新。新损失权重全部设 0 可恢复原目标。

`S*+B*=A_raw`，因此 B 通道不会自动恢复 teacher 已丢失的边界；应检查 R 中
的结构泄漏，不能仅凭原图 MSE 或梯度幅度宣称保边成功。

teacher 使用 `eval()`、冻结参数和 `inference_mode()`，按完整训练切片的规则网格
做带 halo 的 tiled 推理；目标缓存于 CPU，只抽取与 NeUF 训练像素相同的索引。
validation/test 图像不参与目标缓存。teacher 只支持当前 `geometry=none` checkpoint，
保持其像素域配置；NeUF 坐标与位姿继续使用原毫米标定。训练环境通过 `PYTHONPATH`
引用工作区的 `NLSTV/Code/STV`，不复制 STV 实现。

推理时仅加载 NeUF，直接查询 `structure`、`boundary`、`residual`、`anatomy=S+B`
和 `intensity=S+B+alpha*R`；不加载 Neural STV 或 teacher 文件。alpha 仅改变
合成强度，S/B/R/A 不随它改变；`alpha=0` 保留 S+B，`alpha=1` 与原完整输出一致。
现有 v1 checkpoint 的结构元数据保持兼容，无需为开放 alpha 重新训练。
checkpoint 保存 teacher hash、分解规则和训练数据内容/几何签名，不保存 teacher 权重。
断点训练需提供同一 teacher 和相同损失设置；未指定新损失参数时，旧断点沿用
原三分量目标，新断点沿用记录的设置。训练和验证原图 MSE 显式采用 alpha=1，
不受 checkpoint 默认推理 alpha 影响。`--init-e1-checkpoint` 则只迁移 E1
编码器/解码主干，不恢复其优化器，初始化时三分量总和等于原 E1 输出。

运行脚本位于工作区 `qsub/neuf/frozen_stv.sh`，默认只执行最小 smoke：

```bash
qsub -v RUN_ID=YYYYMMDD_trainNN,MODE=smoke \
  -o /misc/raid/zchen/Code/qsub/logs/neuf/YYYYMMDD_trainNN/stdout.log \
  -e /misc/raid/zchen/Code/qsub/logs/neuf/YYYYMMDD_trainNN/stderr.log \
  /misc/raid/zchen/Code/qsub/neuf/frozen_stv.sh
```

提交前创建对应 qsub 日志目录，并选择未使用的运行编号。正式训练用 `MODE=train`，
按预算设置 walltime，并可传 `INIT_E1_CHECKPOINT`。结果统一位于
`/misc/raid/zchen/Code/logs/RUN_ID/cerebral/index_all/neuf_frozen_stv_MODE/`，
其下包含 `checkpoints/`、`metrics/`、`plots/` 和 `run_config/`。
现有三维导出脚本可通过 `COMPONENT=structure|boundary|residual|anatomy|intensity`
分别导出新头；各分量保留原始浮点值，边界/残差显示固定为 `[-0.5,0.5]`。

同一脚本支持以下模式，均需先创建 qsub 日志目录并传入 RUN_ID：

| MODE | 必要参数 | 行为 |
|---|---|---|
| `verify` | `CHECKPOINT` | 唯一 smoke、新损失两步更新、已有权重的固定验证帧 alpha 对比，以及 0/1 两端 1 mm 体数据导出 |
| `evaluate` | `CHECKPOINT` | 无 teacher 的验证集评价和四张固定切片的五档 alpha 图、原始分量和 CSV |
| `export` | `CHECKPOINT`，可选 `ALPHA`、`SPACING_MM` | 导出指定 alpha，默认 alpha=0；不运行 teacher |

例如通过 qsub 传 `MODE=export,CHECKPOINT=/path/to/neuf.pkl,ALPHA=0,SPACING_MM=0.5`。
底层入口是 `python -m neuf.export_full_grid_from_ckpt --ckpt ... --alpha 0 --output ...`，
仍须在 qsub 作业中执行。冻结 STV 场始终保存 `volume_float.npy` 和
`volume_float.mhd/.raw`：float32、NPY 顺序为 z/y/x、MHD spacing/origin 为 x/y/z。
它们保留原始强度供三维分析或分割，不裁值、不逐体归一化；`volume.mhd/.raw`
仅用于 uint8 显示。坐标覆盖仍是现有笛卡尔包围盒，未观测区域不能当作可靠组织。

训练验证保留原 GT/完整重建/分量图，额外保存五档 alpha 及固定 ROI 对比；CSV
中的含噪观测 MSE 和相邻梯度 RMS 只是描述指标，不能替代干净真值或 Dice/HD95。

## 既有 checkpoint 兼容性

当前加载器只接受固定几何、无方向分支的 `HASH` 和 `DUAL_HASH` checkpoint。
属于已移除路线的 checkpoint 会明确报错，不会用 `strict=False` 静默吞掉
新模型的缺失权重。E2 checkpoint 会严格检查 schema 和两个 decoder head。
