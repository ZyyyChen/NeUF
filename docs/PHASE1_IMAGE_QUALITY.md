# Phase 1：固定几何下的简单双分支重建

## 项目目标与测试原则

项目最高优先级是最终三维超声重建图像质量，包括解剖结构清晰度、几何
一致性、散斑、对比度、连续性和伪影抑制。测试只验证运行能力与关键路径
不回归；测试通过不能证明图像质量提高。所有方法结论必须来自固定数据、
固定几何、固定训练预算下的图像与定量对比；否则标为“尚未验证”。

## 范围

Phase 1 完全固定 baked dataset 中的探头位姿、像素物理尺寸、ROI、
`point_min/point_max`、sector mask 和 slice 顺序，只比较图像表示能力：

- E0：`legacy_fixed_geometry` + 普通单 `HASH`，无渐进高频，使用 masked MSE；
- E1：`dual_single_head_matched`，无 gate 的双 HashGrid，使用与 E0 相同的单头；
- E2：`anatomy_speckle_v1`，双 HashGrid、anatomy/speckle 双头；
- C0：只在评价阶段对 E1 做 validation-matched Gaussian smoothing。

本阶段不包含任何 pose、sagittal、view-dependent、uncertainty、Fourier、
Kronecker、ray/A-line physics、KNN 或外部预训练模型。

## E2 模型

`DualFreqEncoder.forward_decomposed()` 返回 low/high 特征。E1 直接使用
`concat(feat_low, feat_high)`；E2 不使用 gate 或渐进高频权重，使用：

```text
A = anatomy_head(feat_low)
S = speckle_head(concat(A, feat_high))
I(alpha) = A + alpha * S
```

默认 `alpha=1.0`。`alpha` 只控制推理和导出时的散斑分量，不进入 encoder，
也不改变坐标或几何。E1 不再以匹配 E2 容量为目标，而是保持 E0 的单头解码器
结构，只替换 HashGrid 编码器。

## 数据冻结

优先使用 baked dataset 原有 held-out slices 作为 test；训练池按原始顺序把
`index % 10 == 5` 划为 validation，其余为 training。若没有 held-out pool，
则完整序列中 `index % 10 == 0` 为 test、`index % 10 == 5` 为 validation。
少于 50 张 transverse slices 的运行只能是 smoke。

训练前冻结并校验：

- `split_manifest.json`；
- 数据文件、像素、pose、point bounds 和 mask hash；
- 腐蚀 3 px 的 metric mask；
- test 中 25%、50%、75% 分位的固定展示切片；
- 只由 ground truth 生成的固定 ROI manifest。

## 训练

正式预算为 20,000 steps，`patch_size=64`，固定 seed 为
`3407, 3408, 3409`。E0/E1/E2 使用相同 split、patch/点预算、优化器、学习率
规则和总 iteration。E0/E1/E2 的所有 HashGrid 和 decoder 参数都从第一个
step 开始联合训练。

E0/E1/E2 都在 sector mask 内优化相同形式的纯 L2：

```text
loss = masked_mean((A + S - target)^2)
```

E2 不再使用分阶段冻结、复合 loss 或额外的末段学习率缩放。由于 A/S 没有
独立监督，二者是模型内部的可观察分量，不预设严格的频率分解保证。

## 评价

完整重建报告 masked MAE、PSNR、SSIM、LP-SSIM、GMS 和输出越界比例。
散斑报告 high-frequency energy、固定 homogeneous ROI 中的 SC/ENL，以及
E2 speckle 的低频泄漏。`alpha=[0,0.25,0.5,0.75,1]` 固定评价单调性、结构
保持和亮度漂移。统计以 slice 配对，并使用固定 seed 的 bootstrap 95% CI。

验收沿用 G0–G7：数据/几何完整性、`alpha=1` 不显著退化、散斑降低、结构
优于匹配平滑、控制稳定、亮度稳定、低频泄漏和三 seed 可复现性。任一可
计算 gate 失败则总结果为 `FAIL`；seed、test 或 ROI 不足则为
`INCONCLUSIVE`；缺少必要输入才是 `BLOCKED`。

## 运行

新实验目录采用“日期 + 试验次数 / 内容”格式：

```text
experiments/YYYYMMDD_trialNN/content/E0/seed3407/
```

`RUN_DATE` 使用 `YYYYMMDD`，`RUN_CONTENT` 用简短的小写英文描述模型和关键
配置。训练脚本未传 `TRIAL_ID` 时自动选择第一个可用编号，也可以显式指定；
评价必须使用训练最终选择的相同三个值。

```bash
qsub jobs/phase_1/run_e0.sh
qsub jobs/phase_1/run_e1.sh
qsub jobs/phase_1/run_e2.sh
qsub -v RUN_DATE=20260901,TRIAL_ID=02,RUN_CONTENT=phase1_e1_plain_dual_vs_e2_hf02 \
  jobs/pbs/run_phase1_image_quality_eval_pbs.sh
```

三个训练脚本是彼此独立的 PBS 任务，但共享冻结 manifest。建议按
E0、E1、E2 顺序逐个提交，前一个完成后再提交下一个，以避免 NFS 文件锁
竞争；从日志确认自动选择的 trial，三个 checkpoint 均完成后再运行评价。

或直接评价已完成 checkpoint：

```bash
python -m neuf.phase1_experiment \
  --dataset path/to/baked_dataset.pkl \
  --output-dir phase1_image_quality \
  --e0 3407=path/to/e0.pkl \
  --e1 3407=path/to/e1.pkl \
  --e2 3407=path/to/e2.pkl \
  --allow-incomplete-seeds
```

`--allow-incomplete-seeds` 仅用于诊断，输出仍必须是 `INCONCLUSIVE`。
