# NeUF Related Work：作者审阅与引用核查

核查日期：2026-09-22。正文为英文 Related Work 初稿，中文文件逐段翻译；本文档不属于投稿正文。

## 项目目标与测试原则

项目目标是改善最终三维超声重建的结构清晰度、几何一致性、散斑表现、对比度和伪影。测试仅确认程序运行及防止关键回归，不能证明图像质量改善。本次仅撰写、审读和编译论文，不运行重建实验、不增加研究代码测试。LaTeX 编译成功只说明文稿可构建；文稿不把尚未完成的公平对照或独立病例验证写成结果。

## 文件和阅读方式

- `main_en.tex`：英文独立编译入口。
- `main_zh.tex`：中文独立编译入口，共享同一份英文参考文献。
- `sections/related_work_en.tex`、`sections/related_work_zh.tex`：正文，各含三个小节、十个一一对应的段落；源码注释 P1–P10 用于反馈定位。
- `references.bib`：18 条实际引用；正式出版记录优先，预印本明确标注。
- `build/<RUN_ID>/`：PDF、编译日志和临时文件，已在本目录 `.gitignore` 中排除。

采用适合审阅的单栏格式和按首次出现顺序编号的参考文献；选定目标期刊后，将章节文件和 BibTeX 并入其正式模板。当前文件不是 T-USON 或 Ultrasonics 官方版式。

英文、中文均使用 XeLaTeX → BibTeX → XeLaTeX → XeLaTeX。中文入口优先使用 xeCJK/Fandol；当前集群缺少中文宏包时，使用已有 Droid Sans Fallback 字体和 XeTeX 原生断行，无需安装依赖。本文档末尾记录实际编译状态。

## 论证路线与段落对照

| 段落 | 作用 | 中文审阅重点 |
|---|---|---|
| P1 | 自由手体重建、校准、插值与传统函数近似 | 是否符合实际采集背景；并未宣称 NeUF 属于 sensorless 方法 |
| P2 | 空间错配与方向相关外观的不同影响 | 并未把多帧模糊全部归因于位姿错误 |
| P3 | 神经场与多分辨率哈希表示 | 连续查询与可恢复空间细节没有混为一谈 |
| P4 | 将几何关系纳入联合估计的研究 | 未把固定几何本身写成创新 |
| P5 | 超声物理渲染和方向建模 | 参数场只输入坐标，也可以经渲染产生方向相关图像 |
| P6 | 几何感知多尺度表示和声学模型改进 | 正面讨论与 NeUF 特征设计接近的 Liu 等人的工作 |
| P7 | 散斑抑制中的结构引导 | 传统方法已有结构保持，不使用“现有方法都忽略边缘”的表述 |
| P8 | STV 与非局部结构张量泛函 | 引用的是结构先验理论，不将其等同于解剖标签 |
| P9 | NLSTV 已用于超声逆波束形成 | 区分 RF 数据上的波束形成与已成像 B-mode 帧的体重建 |
| P10 | NeUF 当前机制和研究范围 | 结构响应门控 logit 残差；没有声称已提高几何精度或优于基线 |

## 引用与证据范围

只用论文原文、出版社、作者机构及 PubMed 记录核查。不是系统综述，也不由“本次未找到完全相同方法”推断全球首创。下表区分全文、摘要和仅元数据证据；摘要级证据只支持概括性描述。

| BibTeX key | 核查来源 | 正文所用事实与证据范围 |
|---|---|---|
| `prager2010three` | [出版社](https://journals.sagepub.com/doi/10.1243/09544119JEIM586) | 2010 年综述；用摘要和正式记录支持采集背景。网页迁移/上线日期不作发表年份。 |
| `solberg2007freehand` | [出版社全文页](https://www.sciencedirect.com/science/article/pii/S0301562907001081) | 自由手重建分类与跟踪、校准等误差因素；综述用于背景，不作算法首创来源。 |
| `rohling1999interpolation` | [出版社摘要](https://www.sciencedirect.com/science/article/pii/S1361841599800280) | 实现/研究 RBF 近似并比较三种标准方法；正文用 investigated，避免 developed 的首创暗示。 |
| `mildenhall2020nerf` | [Springer 正式论文](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_24) | ECCV 2020 连续表示与渲染；仅作通用表示背景。 |
| `mueller2022instant` | [作者机构](https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding)、[论文记录](https://arxiv.org/abs/2201.05989) | 多分辨率哈希特征和图形任务中的效率；不将其速度直接写成 NeUF 的速度。 |
| `song2022carotid` | [IEEE 论文记录](https://ieeexplore.ieee.org/document/9958448/)、[IUS 官方程序](https://2022.ieee-ius.org/wp-content/uploads/sites/64/2022/10/IUS-2022-Final-Program-Updated.pdf) | 元数据/官方会议资料确认早期颈动脉 INR 工作。全文受访问限制；正文不描述其网络、监督或性能。 |
| `yeung2024sensorless` | [MedIA 正式刊版](https://www.sciencedirect.com/science/article/pii/S1361841524000720) | 摘要支持无传感器胎脑重建、位置与隐式体联合优化；采用 2024 最终题名和完整作者顺序。 |
| `wysocki2024ultranerf` | [PMLR 记录](https://proceedings.mlr.press/v227/wysocki24a.html)、[全文](https://proceedings.mlr.press/v227/wysocki24a/wysocki24a.pdf) | Sec. 3.2–3.3：坐标参数场与超声渲染；Appendix E：参数解释的局限。按官方 BibTeX 记 2024。 |
| `song2025implicitcell` | [arXiv 原文记录](https://arxiv.org/abs/2503.06686) | 摘要支持分辨单元模型和联合位姿修正。本次未核实正式期刊出版，按预印本引用。 |
| `dou2026gau` | [论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13321781/) | 多扫查错配、联合优化、早期梯度重加权/平滑；不引用抽象中的夸张百分比。 |
| `guo2024ulre` | [arXiv 原文](https://arxiv.org/abs/2408.00860) | 反射方向参数化与方向编码；本次未核实正式期刊出版，按预印本引用。 |
| `liu2026geometry` | [出版社](https://www.sciencedirect.com/science/article/pii/S0925231226007538)、项目根目录同名 PDF | 本地全文 Fig. 2/方法：声阻抗梯度与粗尺度反射、细尺度散射 Hash 特征。卷681，文章133356。 |
| `wysocki2026usnerf` | [出版社](https://www.sciencedirect.com/science/article/pii/S136184152600263X)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/42570451/) | 修订前向模型减少参数冗余，物理启发正则改善声学图解释性；2026-08-03 已在线出版。 |
| `yu2002srad` | [原论文摘要及元数据](https://pubmed.ncbi.nlm.nih.gov/18249696/) | 局部统计引导的散斑抑制扩散；不宣称其解决多帧神经场拟合。 |
| `krissian2007osrad` | [原论文摘要及元数据](https://pubmed.ncbi.nlm.nih.gov/17491469/) | 跨轮廓/主曲率方向的不同扩散强度，含三维超声应用。 |
| `lefkimmiatis2015stv` | [SIAM](https://epubs.siam.org/doi/abs/10.1137/14098154X)、[作者机构](https://bigwww.epfl.ch/publications/lefkimmiatis1501.html) | 张量特征值定义 STV 正则族；通用逆成像方法。 |
| `lefkimmiatis2015nonlocal` | [作者论文全文](https://back.skoltech.ru/storage/app/media/archive/sites/19/2016/07/ticJ2015.pdf)、[作者机构目录](https://ww3.math.ucla.edu/cam-reports-2001-2020/) | 非局部结构张量泛函，结合方向结构与相似邻域；IEEE TCI 1(1):16–29。 |
| `li2026ipbnlstv` | [出版社摘要](https://www.sciencedirect.com/science/article/pii/S0041624X25002252)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/40839915/) | NLSTV 正则化逆波束形成，RF→复数图像；不把该任务的性能数值移用于 NeUF。 |

### 容易混淆的出版信息

- Ultra-NeRF：会议名称含 MIDL 2023，PMLR 正式条目/BibTeX 为 **2024**，卷227，382–401页；采用该记录，不与会议年份混用。
- ImplicitVol：采用 **2024 年 MedIA 最终刊版**，不把 2021 年预印本旧题名重复计作另一项工作。作者中的 INTERGROWTH-21st Consortium 保留。
- IPB-NLSTV：在线日期为2025-08-18，正式卷期为 **2026年1月，157:107788**，DOI 含2025属于正常情况。
- UltraSoundNeRF：卷期标为2026年11月，但已于 **2026-08-03** 在线发表，早于本次核查日期；采用已分配的卷114与文章号104194。
- ImplicitCell 与 UlRe-NeRF：正文和文献表明确标预印本，arXiv DOI 不是正式期刊 DOI。

## 三个 agent 的实际审读与修改

1. **文献与论点 agent（paper_ideas）**：核查传统重建和早期 INR；读了初稿 P1–P3。采纳其 RBF 归属修正、为校准背景补充 Solberg 综述、为网格查询论点补充 Yeung 刊版引用。
2. **科学审稿 agent（paper_reviewer）**：核对最近邻论文及本地 Liu 论文全文，审读初稿 P4–P6/P10。采纳其区分已知几何与可靠几何、拆开 UltraSoundNeRF 重参数化和正则化作用、明确 NLSTV 来源及 logit 门控的建议。
3. **语言与逻辑 agent（language_editor）**：实际逐段审读 P1–P10 的英中稿。采纳其重写 P2 衔接、压缩 P3/P6/P8、删去“直接先例”“已成既有组件”等像审稿回复的措辞，并将 P10 缩为方法定位；同时核对中文限定范围。

语言修改保留科学审稿人的技术约束：P6 不把正则化与参数删减混为同一机制；P10 明确 residual 在最终激活前门控。正文没有“首次”“全面优于”“恢复真实解剖”“纯散斑分解”或未经验证的临床收益。

语言 agent 对修改后的英中稿再次复核，结论为“复核通过，无必改项”；其关于 P10 中文语序的可选建议也已采纳。

## 需要作者确认的内容

- P10 的范围依据当前 `neuf/edge_field/model.py` 和 README 的 NLSTV response 路径。后续若主方法改为方向建模、位姿联合优化或另一 teacher，需同步调整这一段。
- 本轮未越出 NeUF 边界读取 NLSTV teacher 实现。STV/NLSTV 论文支持先验家族的归属；Methods 仍应准确记录实际 response 的定义、生成参数和与原始正则化模型的关系。
- Song 2022 的技术描述受全文可访问性限制，现稿只将其作为早期应用引用；深入比较其监督方式前，应取得全文。
- 目前优先检查中文版 P2、P6、P9、P10：它们分别决定问题边界、最近邻定位、任务区分和本文主张。

## 编译记录

使用项目 `qsub/neuf/compile_paper.sh`，通过已核查的 CPU 路由队列提交，1 CPU、2 GB、最长10分钟，不申请 GPU。

- 最终 Run ID：`20260922_train04`。
- Job ID：`5092001.linux1.dg.creatis.insa-lyon.fr`。
- 结果及编译日志：`paper/build/20260922_train04/`。
- qsub 原生日志：`qsub/logs/neuf/20260922_train04/`。
- 状态：已完成，`exit_code=0`。
- 英文输出：`paper/build/20260922_train04/main_en.pdf`，含参考文献共4页。
- 中文输出：`paper/build/20260922_train04/main_zh.pdf`，含参考文献共4页。
- 核查：两份文献表各有18条实际引用，中英文正文引用顺序一致；最终日志无未定义引用、缺失字形或 overfull 警告；已检查英中首页预览和中文文本提取结果。
- 前三次编译分别用于修正 BibTeX 输出路径限制和集群中文字体兼容问题；最终成功版本以上述 train04 为准。
