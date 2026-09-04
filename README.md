# Neural Ultrasound Field (NeUF)

NeUF 目前只保留固定几何下的 HashGrid 重建主线：一个最基础的单头
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
  nerf_network.py                三个保留的 field head
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

## Checkpoint 边界

当前加载器只接受固定几何、无方向分支的 `HASH` 和 `DUAL_HASH` checkpoint。
属于已移除路线的 checkpoint 会明确报错，不会用 `strict=False` 静默吞掉
新模型的缺失权重。E2 checkpoint 会严格检查 schema 和两个 decoder head。
