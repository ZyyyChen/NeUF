# NeUF 接入原始 Ultra-NeRF 超声渲染器：Codex 执行说明

## 1. 任务目标

在**保留当前 NeUF 数据、坐标、编码器、位姿优化与训练/测试划分**的前提下，新增一个与原始 Ultra-NeRF 官方实现一致的超声物理渲染分支，用于与当前 NeUF 点强度渲染进行可重复的 A/B 对比。

本任务不是把 `slice_render_ray.py` 的 `integration` 改成另一个选项，也不是加入普通 NeRF 的 alpha compositing。必须实现 Ultra-NeRF 的完整链路：

1. 网络在每个三维位置预测 5 个原始参数；
2. 将一幅二维超声图的每一列组织为一条轴向 A-line；
3. 沿深度做**排除当前采样点的累积衰减**；
4. 沿深度做**排除当前采样点的界面反射传输损失**；
5. 分别对随机边界图和随机散射图做二维高斯 PSF 卷积；
6. 计算反射回波 `R`、后向散射回波 `B`，输出 `E = R + B`；
7. 保存全部中间物理参数图，确保实现可审计。

参考实现固定为：

- 原始官方仓库：<https://github.com/magdalena-wysocki/ultra-nerf>
- 原始 TensorFlow 渲染代码：<https://github.com/magdalena-wysocki/ultra-nerf/blob/main/run_ultra_nerf.py>
- 论文：<https://proceedings.mlr.press/v227/wysocki24a.html>

开始编码前，将参考仓库的 commit SHA 写入实验配置或运行清单。若论文公式与官方代码不完全一致，**本实验以官方代码 `render_method_convolutional_ultrasound()` 的实际运算为准**。特别是界面传输项必须使用官方代码中的 `1 - beta * G`，不要照排版容易歧义的论文公式改成 `(1-beta)*G`。

---

## 2. 当前代码审查结论

已提供代码中的当前路径是：

- `nerf_network.py`
  - 网络当前输出 `i_clean` 和 `log_sigma`，不是 Ultra-NeRF 的 5 个组织参数；
  - 部分编码模式会输入观察方向；
  - 当前输出头不能直接用于 Ultra-NeRF 物理渲染。
- `slice_renderer.py`
  - 每个图像像素只查询一个三维点，直接得到强度；
  - 这是当前 NeUF 点采样基线，应完整保留。
- `slice_render_ray.py`
  - 以每个像素为中心，沿 `viewdir` 在 `near` 到 `far` 之间取若干样本；
  - 使用 `mean`、`sum` 或普通 alpha compositing 聚合成一个像素；
  - 这不是 Ultra-NeRF：它没有轴向 A-line 的累计衰减、界面传输、边界/散射 Bernoulli 采样和二维 PSF 卷积。
- `sagittal_supervision.py`
  - 当前可随机抽取互不相邻的像素监督；
  - Ultra-NeRF 渲染依赖完整深度前缀和二维邻域，不能对离散随机像素独立渲染。
- `utils.py`
  - 当前局部像素坐标使用 `(depth, lateral, 0)`，然后变换到世界坐标；
  - 接入时应复用数据集中已生成的整幅像素点，避免重新猜测轴方向和位姿约定。
- `recons3D.py`、`recons3d_exact_from_saved.py`、`export_knn_baseline.py`
  - 属于下游重建/导出，不是训练时的 Ultra-NeRF 渲染器；
  - 本任务不得为了接入渲染器而改写这些文件。

---

## 3. 不可违反的范围约束

### 3.1 必须保留

- 现有 `slice_renderer.py`，作为 `point`/`legacy` 基线；
- 现有 `slice_render_ray.py`，但不要把它重命名成 Ultra-NeRF；
- 当前数据读取、世界坐标、ROI 偏移、训练/验证/测试划分；
- FREQ、HASH、DUAL、KRONECKER 等现有空间编码器；
- 当前位姿优化框架和坐标变换；
- 旧 checkpoint 的加载能力。

### 3.2 严禁

- 用普通 NeRF 的 `alpha = 1-exp(-sigma*delta)` 代替 Ultra-NeRF 公式；
- 沿图像法向或切片厚度方向做 mean/sum/alpha 聚合，并称之为 Ultra-NeRF；
- 把随机独立像素拼起来后直接做 `cumprod` 或 PSF 卷积；
- 在横向 chunk 内各自做卷积，产生 chunk 边界接缝；
- 对每张预测图单独 min-max 归一化后再计算指标；
- 删除或覆盖现有基线；
- 为了“改善结果”擅自加入 log compression、TGC、可学习 PSF、法向量反射、Relaxed Bernoulli 或其他论文外模块；
- 仅用能跑通作为验收，必须完成数值、形状、坐标和对比实验测试。

---

## 4. 新增运行模式

统一增加显式参数：

```text
--renderer point
--renderer ultra_nerf
```

建议同时增加：

```text
--ultra-psf-half-size 3
--ultra-psf-lateral-std 2.0
--ultra-psf-axial-std 1.0
--ultra-distance-unit m
--ultra-bernoulli-seed 0
--ultra-eval-mc-samples 1
--ultra-save-parameter-maps
```

默认值必须复现原始实现。`renderer`、PSF 参数、距离单位、随机种子、网络输出模式和图像尺寸都必须写入 checkpoint。加载 checkpoint 时若命令行模式与 checkpoint 不一致，应明确报错，不能静默切换。

---

## 5. 网络输出契约

### 5.1 Ultra-NeRF 分支

对每个查询位置输出：

```text
raw[..., 0] = alpha_raw   # attenuation
raw[..., 1] = beta_raw    # reflection coefficient
raw[..., 2] = rho_b_raw   # border probability
raw[..., 3] = rho_s_raw   # scattering density
raw[..., 4] = phi_raw     # scattering amplitude
```

激活函数必须与官方实现一致：

```python
alpha = torch.abs(alpha_raw)
beta  = torch.sigmoid(beta_raw)
rho_b = torch.sigmoid(rho_b_raw)
rho_s = torch.sigmoid(rho_s_raw)
phi   = torch.sigmoid(phi_raw)
```

Ultra-NeRF 分支不输出 `i_clean`，也不把 `log_sigma` 当作体密度。不得把现有两个输出强行映射成五个参数。

### 5.2 方向输入

原始 Ultra-NeRF 的组织参数是各向同性空间属性，网络不输入观察方向；观察方向依赖由传播和遮挡渲染产生。因此严格分支应设置：

```text
use_directions = false
```

现有空间编码器可以保留，但 Ultra-NeRF 分支只编码三维位置。若需要研究“保留方向输入是否更好”，必须作为另一个明确命名的非严格消融，不能混入主结果。

### 5.3 推荐实现方式

不要破坏旧输出头。可采用以下任一安全方案：

1. 在 `NeRF` 中增加 `output_mode="intensity" | "ultra_nerf"`，按模式创建独立输出头；或
2. 新增 `UltraNeRFField`，复用同一编码器和 MLP 主干。

旧 checkpoint 默认解释为 `output_mode="intensity"`。新分支的旧权重不能伪装成已训练的五参数头；A/B 两个分支必须从相同随机种子分别重新训练。

---

## 6. 图像与 A-line 张量布局

Ultra-NeRF 不是“每个像素一条短射线”。对于线阵二维图像：

- 每个横向列是一条 A-line；
- A-line 上包含从浅到深的所有轴向像素；
- 累积传输只沿轴向深度进行；
- PSF 同时跨横向和轴向进行二维卷积。

复用：

```python
points = dataset.get_slice_points(frame_index).reshape(H, W, 3)
```

其中 `H` 必须是从浅到深的轴向维度，`W` 是横向维度。先通过测试确认，不得凭变量名猜测。为贴合官方实现，网络查询后将结果组织为：

```text
raw_wh5.shape == [W, H, 5]
```

即第 0 维是 A-line/横向索引，第 1 维是深度。所有 `cumprod` 都在第 1 维完成。最终输出再转回项目统一使用的 `[H, W]`。

必须添加一个坐标测试：取任意一列，证明相邻采样点的物理位置沿探头轴向单调排列，第一点靠近 ROI 顶部，最后一点靠近 ROI 底部。若当前数据存储方向相反，只允许在一个集中位置显式翻转，并记录到配置；不能让渲染器在不同接口中各自翻转。

---

## 7. 与官方代码一致的物理渲染

建议新增：

```text
neuf/ultra_nerf_renderer.py
```

输入：

```text
raw_wh5: [W, H, 5]
z_vals_wh: [W, H]，单位为米，沿 H 从浅到深
```

输出字典至少包含：

```text
intensity_map
attenuation_coeff
reflection_coeff
border_probability
border_indicator
attenuation_transmission
reflection_transmission
scatterers_density_coeff
scatterers_density
scatter_amplitude
psf_scatter
b
r
transmission
```

### 7.1 深度采样间隔

```python
dists = torch.abs(z_vals_wh[:, 1:] - z_vals_wh[:, :-1])
dists = torch.cat([dists, dists[:, -1:]], dim=1)
```

若项目内部空间单位为毫米，在进入渲染器前统一乘 `1e-3` 转换为米。不要直接把毫米距离放入官方指数衰减公式，否则指数尺度会相差 1000 倍。

### 7.2 衰减传输

```python
attenuation_step = torch.exp(-alpha * dists)
T_att = exclusive_cumprod(attenuation_step, dim=1)
```

`exclusive_cumprod` 的第一个深度位置必须严格为 1：

```python
def exclusive_cumprod(x, dim):
    inclusive = torch.cumprod(x, dim=dim)
    head_shape = list(x.shape)
    head_shape[dim] = 1
    head = torch.ones(head_shape, dtype=x.dtype, device=x.device)
    return torch.cat([head, inclusive.narrow(dim, 0, x.shape[dim] - 1)], dim=dim)
```

### 7.3 边界采样与反射传输

```python
G = torch.bernoulli(rho_b, generator=border_generator).detach()
reflection_step = 1.0 - beta * G
T_ref = exclusive_cumprod(reflection_step, dim=1)
```

硬 Bernoulli 采样及其不可微行为是原始实现的一部分。主分支不得换成阈值、概率期望或 Relaxed Bernoulli。

### 7.4 原始二维高斯 PSF

原始代码使用 `7 x 7` 的归一化二维高斯核：

- half-size = 3；
- 横向标准差 = 2；
- 轴向标准差 = 1；
- 均值 = 0；
- 核元素总和 = 1。

由于 PyTorch `conv2d` 输入为 `[N,C,H,W]`，建议把 `[W,H]` 映射为 `[1,1,W,H]`，这样卷积核的第一空间轴对应横向、第二空间轴对应轴向。使用零填充 `padding=3`，等价于 TensorFlow 的 `padding="SAME"`。

```python
G_psf = F.conv2d(G[None, None], kernel, padding=3)[0, 0]
```

必须用冲激响应单元测试证明：横向扩展比轴向更宽，不能因为转置错误把两个标准差交换。

### 7.5 后向散射

```python
H_s = torch.bernoulli(rho_s, generator=scatter_generator).detach()
scatterers_map = H_s * phi
S_psf = F.conv2d(scatterers_map[None, None], kernel, padding=3)[0, 0]
```

严格官方代码直接使用 `sigmoid(phi_raw)` 作为散射幅度，没有实际执行论文文字所述的单位方差正态采样。本任务必须跟随代码，不增加正态噪声。

### 7.6 最终回波

```python
I = T_att * T_ref
B = I * S_psf
R = I * beta * G_psf
E = B + R
```

最终将所有 `[W,H]` 图转置回 `[H,W]`。严格主分支不做 log compression、TGC 或逐图归一化。

---

## 8. 查询分块与卷积边界

为控制显存，可以把三维点展平后分块调用网络：

```text
[H,W,3] -> [H*W,3] -> chunked model query -> [H,W,5]
```

但是物理渲染必须在重新拼成完整图以后进行：

- `cumprod` 必须看到每条 A-line 的完整深度；
- PSF 卷积必须看到整幅图；
- 不允许在 network chunk 内独立计算传输或卷积；
- 不允许在横向 block 边界无 halo 地卷积。

如果完整图仍超显存，可使用连续横向块加 3 像素 PSF halo，但每个块仍需包含完整深度，且 loss 只作用于去掉 halo 的中心列。严格复现实验优先使用整幅图。

---

## 9. 训练循环必须同步修改

原始 Ultra-NeRF 每次选择一幅训练图并渲染完整帧。严格分支必须采用同样的结构：

1. 选择一个 frame；
2. 获取该 frame 的完整 `[H,W,3]` 采样点；
3. 查询五参数场；
4. 在完整 `[W,H]` 上渲染；
5. 与对应完整 target frame 计算图像损失。

不能继续使用当前的随机独立像素训练，因为：

- 第 `i` 个深度像素依赖此前所有深度的衰减与反射；
- 每个输出像素依赖二维 PSF 邻域；
- 离散像素没有定义完整的 A-line 前缀或卷积邻域。

若启用 `sagittal_supervision.py`，Ultra-NeRF 分支应调用 `refined_geometry()` 获取完整平面并整帧渲染，不能调用当前 `sample()` 随机抽点后独立渲染。为避免混杂，第一轮 A/B 实验建议先关闭 sagittal auxiliary supervision；主比较通过后再在两个分支中以等权重重新加入。

### 损失的公平性

“渲染器对比”要求两个分支使用相同的目标函数。建议主 A/B 使用原论文配置：

```text
loss = 0.9 * (1 - MS-SSIM(pred, target)) + 0.1 * MSE(pred, target)
```

两个分支均在 `[0,1]` target 上计算同一损失。预测值不做逐图归一化。若为了数值安全需要显示裁剪，只能生成单独的 `display_image = clamp(pred,0,1)`，不得偷偷替换训练张量。

如果要保留当前 NeUF 的 uncertainty/NLL loss，应另报一组“各自最佳配置”的系统级比较，不能把它称为只比较渲染器。

---

## 10. 位姿优化兼容性

位姿优化必须继续通过三维查询点传递梯度：

```text
base full-frame points
    -> refined pose transform
    -> encoded 3D points
    -> five raw parameters
    -> Ultra-NeRF renderer
    -> image loss
```

Bernoulli 样本对 `rho_b` 和 `rho_s` 不可微是官方行为，不应误判为位姿梯度中断。验收时应分别检查：

- loss 对 refined 3D points 有有限、非零梯度；
- loss 对位姿旋转/平移参数有有限梯度；
- `alpha`、`beta`、`phi` 分支有梯度；
- 硬 Bernoulli 路径下 `rho_b`、`rho_s` 不具有重参数化梯度是预期现象。

---

## 11. 随机性与评估规则

必须分别管理 border 和 scatter 两个随机生成器，并把种子写入运行清单。为避免结果依赖 network query chunk 大小：

- 先得到完整 `rho_b`、`rho_s`；
- 再对完整图一次性采样；
- 不要在 network chunk 循环内采样。

训练可随迭代推进随机状态。验证和测试至少提供：

1. `strict_seeded`：固定种子的一次官方式随机渲染，作为主可重复结果；
2. 可选 `mc_mean`：多个随机样本的平均，仅作为稳定性补充，不能替代主结果。

保存预测图时同时保存 seed。相同 checkpoint、frame、seed 的输出必须逐元素一致。

---

## 12. 建议文件改动

### 必须新增

```text
neuf/ultra_nerf_renderer.py
neuf/slice_renderer_ultra_nerf.py
tests/test_ultra_nerf_renderer.py
tests/test_ultra_nerf_slice_integration.py
```

### 必须最小修改

```text
neuf/nerf_network.py
neuf/main.py                 # 实际训练入口；__main__.py 只转发 main()
neuf/sagittal_supervision.py # 仅在启用该监督时适配整帧路径
checkpoint/config 保存逻辑
evaluation/export 脚本
```

### 不应修改

```text
neuf/slice_renderer.py       # 保留当前点采样基线
neuf/slice_render_ray.py     # 保留现有通用 ray integration 实验
recons3D.py
recons3d_exact_from_saved.py
export_knn_baseline.py
```

若实际仓库路径不同，按职责定位文件，不要只根据此处文件名机械新建重复模块。

---

## 13. 单元测试与数值验收

所有测试必须在 CPU 上可运行；CUDA 额外运行集成测试。

### 13.1 形状与范围

- 输入 `[W,H,5]`，所有物理图输出 `[W,H]`；
- 返回项目接口后为 `[H,W]`；
- `alpha >= 0`；
- `beta,rho_b,rho_s,phi` 均在 `[0,1]`；
- 所有输出为有限值。

### 13.2 exclusive cumprod

- 第一深度位置严格为 1；
- 与独立 NumPy 参考实现一致；
- 深度维与横向维没有交换。

### 13.3 解析极端情况

- `rho_b=0, rho_s=0` 时，`R=0`、`B=0`、`E=0`；
- `alpha=0, beta=0` 时，`T_att=T_ref=1`；
- 单一确定边界冲激时，`G_psf` 等于高斯核的正确裁剪/平移；
- 单一散射冲激时，`S_psf` 等于高斯核乘以 `phi`；
- 常量正衰减时，`T_att` 沿深度按解析指数规律单调下降。

### 13.4 PSF

- kernel shape 为 `[1,1,7,7]`；
- kernel sum 在容差内等于 1；
- 横向标准差为 2，轴向标准差为 1；
- 使用 `SAME` 等价零填充；
- 完整渲染与改变 network query chunk 大小的结果一致。

### 13.5 坐标与布局

- `reshape(H,W,3)` 后一列对应同一 A-line；
- 轴向索引从浅到深；
- target 与 prediction 的横纵方向一致；
- 用不对称的人工测试图防止“转置两次后看起来仍正常”。

### 13.6 随机性

- 同 seed 完全一致；
- 不同 seed 的 Bernoulli 图通常不同；
- 结果不依赖模型查询 chunk 大小；
- 验证/测试不会因为数据加载顺序改变 seed。

### 13.7 官方代码对齐

构造一个很小的固定 `raw` 和 `z_vals` 张量，使用独立 NumPy 参考实现逐项比较：

```text
alpha, beta, rho_b, rho_s, phi
T_att, T_ref, G_psf, S_psf, I, R, B, E
```

连续量建议 `rtol <= 1e-5, atol <= 1e-6`。Bernoulli 图应直接作为固定输入注入参考测试，避免 TensorFlow 与 PyTorch 随机数算法不同造成伪差异。

---

## 14. A/B 对比实验设计

### 14.1 主实验：渲染器控制变量

至少运行 3 个独立种子：

| 项目 | A：NeUF point baseline | B：NeUF + strict Ultra-NeRF renderer |
|---|---|---|
| 数据与划分 | 相同 | 相同 |
| ROI、分辨率、位姿 | 相同 | 相同 |
| 空间编码器 | 相同 | 相同 |
| MLP 深度/宽度 | 相同 | 相同 |
| 方向输入 | 均关闭，用于公平渲染器对比 | 关闭 |
| loss | 相同 | 相同 |
| optimizer、LR、迭代数 | 相同 | 相同 |
| 随机种子 | 配对 | 配对 |
| 输出头 | 单强度 | 五物理参数，差异必须注明 |

这组结果称为“渲染模型对比”。由于输出头的语义和维数必然不同，不要声称两组使用同一训练权重。

### 14.2 系统级补充实验

可另比较：

- 当前 NeUF 的现有最佳配置；
- NeUF + strict Ultra-NeRF renderer + Ultra-NeRF 官方损失配置。

这组可回答哪套完整系统更好，但不能单独归因于渲染器。

### 14.3 指标

只在固定 hold-out test frames 上报告：

- PSNR；
- SSIM；
- MAE 或 MSE；
- NCC；
- 每帧推理时间与峰值显存；
- 所有 test frames 的 mean、standard deviation、median；
- 三个训练种子的均值和离散程度。

若已有适合该数据的解剖结构或分割标注，可额外报告边界/结构指标；不要临时从测试图人工挑选“好看的区域”作为主指标。

### 14.4 必须导出的可视化

对相同 test frame、相同显示窗导出：

```text
target
point baseline prediction
Ultra-NeRF E
absolute error of baseline
absolute error of Ultra-NeRF
alpha
beta
rho_b
sampled G
rho_s
sampled H_s
phi
T_att
T_ref
R
B
```

禁止对每一张图独立拉伸动态范围。参数图可使用各自有物理意义的固定范围，预测/目标/误差图必须在不同方法间使用统一范围。

---

## 15. 运行清单与结果目录

每次实验保存：

```text
run_manifest.json
config.yaml or args.json
git_commit.txt
reference_ultra_nerf_commit.txt
train_history.csv
test_metrics_per_frame.csv
test_metrics_summary.json
parameter_maps/
predictions/
checkpoints/
```

`run_manifest.json` 至少包括：数据标识、train/val/test 划分、ROI、H/W、空间单位、renderer、network output mode、方向输入、PSF 参数、Bernoulli seed、loss、optimizer、学习率、迭代数、模型查询 chunk、训练 seed、软件版本和 GPU 型号。

---

## 16. 实施顺序

1. 冻结并运行当前 baseline，记录可复现命令、checkpoint 和 test 指标。
2. 添加 `output_mode` 与 checkpoint 兼容逻辑，不改变旧模式输出。
3. 独立实现 `ultra_nerf_renderer.py`，先通过纯张量单元测试。
4. 实现完整切片/A-line 组织和坐标方向测试。
5. 接入训练循环的整帧渲染；网络查询允许分块，物理渲染不分块。
6. 适配验证、测试、checkpoint 和参数图导出。
7. 如需 sagittal supervision，再把它适配到完整平面渲染。
8. 先在极小数据和低分辨率上做 overfit smoke test。
9. 运行配对种子的正式 A/B 实验。
10. 生成统一指标表和无独立归一化的对比图。

---

## 17. 完成标准

只有同时满足以下条件才可报告任务完成：

- `--renderer point` 的旧训练和旧 checkpoint 加载不受影响；
- `--renderer ultra_nerf` 输出 5 个正确激活的参数；
- 完整 A-line 沿轴向做两个 exclusive cumprod；
- 边界图和散射图都使用硬 Bernoulli；
- 使用原始 7×7、横向 std=2、轴向 std=1 的归一化 PSF；
- 最终严格按 `E=B+R` 输出，未混入 alpha compositing 或 log compression；
- query chunk 不改变结果，图像方向无转置错误；
- 单元测试、CPU smoke test 和至少一个 CUDA integration test 通过；
- 相同 seed 的评估可复现；
- 完成至少 3 个配对种子的 hold-out A/B 对比；
- 输出逐帧指标、汇总指标、运行时间、显存和中间参数图；
- 在最终报告中清楚区分“渲染器控制变量比较”和“各自最佳系统比较”。

若其中任一项未完成，应明确列为未完成项，不得用“基本按照 Ultra-NeRF”代替严格验收。
