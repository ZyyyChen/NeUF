# Ultra-NeRF checkpoint 只读诊断报告

1. **mask 是否完全排除了扇区外背景和所有 UI？** 结构核验结果为 `True`；固定 mask 仅保留一个中央、逐行与逐列连续且不接触图像边界的连通区域，`ui_regions_excluded=True`。底部文字与两侧设备参数位于 mask 外。见 [mask overlay](mask/mask_overlay_frame_121.png) 和 [three-frame overlay](mask/mask_overlay_three_frames.png)。
2. **当前输出主要来自 R 还是 B？** 主要来自 B（backscattering）；`B_fraction=0.99999998`，`R_fraction=2.1357921e-08`。同一次随机 forward 的分解见 [decomposition panel](maps/frame_121_decomposition_panel.png)。
3. **rho_b、rho_s 是否集中在 0.5 附近？** 否。`rho_b` 的 p50 为 `0.0049958494`，在 `[0.45,0.55]` 内的比例为 `0`；`rho_s` 的 p50 为 `0.19992919`，对应比例为 `0`。自动标志分别为 `RHO_B_NEAR_HALF=False`、`RHO_S_NEAR_HALF=False`。
4. **rho 原始通道和输出头 gradient norm 是多少？** `rho_b_raw`: raw L2 `0`，head weight-row L2 `0`，head bias abs `0`；`rho_s_raw`: raw L2 `0`，head weight-row L2 `0`，head bias abs `0`。见 [gradient bar plot](gradients/gradient_barplot.png)。
5. **是否确认硬 Bernoulli 阻断两个 rho 通道？** `True`。判定使用一次 masked-MSE backward、零梯度阈值 `1e-12` 和其他通道活跃阈值 `1e-09`。阻断位置是 `neuf/ultra_nerf_renderer.py:194-197` 的 border `torch.bernoulli(...).detach()` 和 `:209-212` 的 scatter `torch.bernoulli(...).detach()`；机器可读位置也记录于 `checkpoint_info.json`。
6. **cumprod 的实际维度和路径索引是什么？** 当前 raw layout 为 `[W,H,5]`，`neuf/ultra_nerf_renderer.py:191,206` 的两个 `exclusive_cumprod(..., dim=1)` 都沿 `raw_wh5[原图列, 原图行]` 的第二维推进；`UltraNeRFSliceRenderer._query_raw()` 将原始 `[H,W]` row-major 像素 reshape 后 permute 为该布局。
7. **10 条路径是扇形发散还是竖直/平行？** `10/10` 条被判为近竖直，`PARALLEL_VERTICAL_GEOMETRY=True`，方向角 spread 为 `0°`。它们是当前代码真实的恒定列路径，不是另画的理想探头射线。见 [cumprod path overlay](paths/frame_121_cumprod_paths_overlay.png)。
8. **是否触发 STOP_PHYSICS_BRANCH？** `True`。原因：`current cumprod paths do not match convex-sector propagation geometry`；触发条件为 `PARALLEL_VERTICAL_GEOMETRY, PATH_INSIDE_MASK_FRACTION_BELOW_0.95`。
9. **当前 checkpoint 失败最直接由哪些证据支持？** 同次 forward 中 `B_fraction=0.99999998`、`R_fraction=2.1357921e-08`，E 几乎完全是随机 backscattering；唯一一次 backward 中两个 rho raw/head 梯度均严格为 0，而 alpha/phi 通道仍活跃；实际传播路径为 `10/10` 条恒定列，无法表示凸阵扇形发散。上述证据只支持诊断与停止判断，不构成重建质量改善结果。

## 项目目标与测试原则

项目最高优先级是提高最终重建图像的解剖清晰度、几何一致性、散斑保留与区分能力、对比度和伪影抑制。本任务仅做只读诊断：测试只用于确认代码可运行并防止关键回归，不扩展重复或一次性 smoke-test 框架；测试通过不代表重建质量提高。未进行重新训练、参数扫描或图像增强，本报告结论不声称图像质量改善，方法效果为“尚未验证”。

诊断 checkpoint：`/misc/raid/zchen/Code/NeUF/experiments/ultra_nerf_renderer_cerebral_stable/latest/ckpt.pkl`；frame：`121`；seed：`0`；optimizer step：`0`。
