# docs/84:SpGEMM 领域技术发展路线全景 + 周三汇报底稿(10-15)

> **用途**:10-21(周三)近期工作汇报的领域综述底稿。
> **方法**:外部检索(多轮)+ 论文一手摘要/贡献清单(Wang25/IA-SpGEMM/ocean 已核原文)
> + 仓库一手源码互证(external_sota 有 bhSparse/HSMU/TileSpGEMM/MMSpGEMM 四套)。
> **纪律**:所有数字带【日期+口径】;未核到一手的标注(二手)。

---

## 〇、一页总览(汇报开场用)

**领域一句话**:SpGEMM 的优化史 = 把"决策权"从编译期不断搬到运行期的历史
——先解决"怎么算"(累积器/负载均衡,2008-2020),
再解决"什么时候算/怎么分配"(估计与调度,2019-2026)。

**两条轴**:
```
Kernel 轴(怎么算):  dense SPA → smem hash → ESC 排序 → 归并 → 混合分域 + tiling/均衡四板斧
调度轴(怎么选):    库级启发式 → 静态阈值 → 离线学习(2019) → 管线内 ML(2025) → 估计式工作流(2026)
                                                                       └→ 我们:形态级分发 + 三层裁定 + 证书
```

**我们的位置一句话**:kernel 轴上我们持七武器生态(部分区间 0.4-0.9×碾压);
调度轴上我们做的是前人没做的最后一层——**管线形态级**的分发,配判废证书兜底。

**四根场外支柱**(§五,"整个领域"视角):估计理论血统 / 分布式·多核·多 GPU /
应用拉动 / 重排局部性与标准化。已有系统综述 = Gao 2023(CSUR 55(12))——
本文定位 = GPU 主线深挖 + 2023-26 更新(估计范式/形态调度)+ 我方视角。

---

## 一、问题定义与两个根本矛盾(一切工作的坐标系)

SpGEMM: C = A·B,输入输出全稀疏。基础算法 Gustavson(1978):**按输出行组织**,
第 i 行 = A 的第 i 行各非零对应的 B 行做"稀疏向量累积"(找同列→累加)。
GPU 上的自然并行单位 = 输出行;**累积器**(去重+累加的行内数据结构)是每行的心脏。

**矛盾 ①:输出未知**。C 的每行 nnz 计算前不可知 → 内存分配和负载均衡都要"预知未来"
→ 催生**两遍范式**(symbolic 数 nnz + numeric 算数),这是 2014 年以来的标准结构。
symbolic 相位平均占 **~28%** 运行时间(ocean ICS'26 实测)——这就是 2026 年
"估计范式"(用 HLL 估计替代精确 symbolic)的靶子,也是我们 DCEST 的哲学同源。

**矛盾 ②:行间失衡 × 行内去重**。行长跨 4-6 个数量级(巨鲸行 vs 微行);
行内去重的代价决定累积器选型,而行内分布(重复度/列跨度)矩阵间差异极大。

**硬件不变量**(H100 实例,我们的台账):smem 每 SM 仅 ~228KB(全形态 dense 梯子
227KB/1024T 的几何就是被它钉死的);smem **无 double 原子加法**,hash 累加重行
只能 CAS 自旋(我们 17 种绕法全败,判为硬件墙);launch 间隙 ~2-5µs/发,
小阵 30-50% 时间是"仪式税"(我们 1c 融合的靶)。

---

## 二、Kernel 算法轴:累积器六谱系 + 均衡四板斧

### 2.1 累积器谱系(核心 taxonomy,每个附"我们的对应武器")

| # | 谱系 | 机制 | 代表工作 | 优劣 | 我们的对应 |
|---|---|---|---|---|---|
| 1 | **dense SPA / 位图** | 整段列区间开数组/位图,直写下标 | Gustavson 1978 原生;现代=bitmap 变体 | 定位 O(1) 零原子;空间 O(n) 只配超宽行 | dense_struct 梯子(巨鲸行 0.46-0.78× 赢 ocean【09-30 同行集口径,docs/79】) |
| 2 | **smem hash 表** | 行内开哈希表,同列碰撞累加 | Nagasaka 2017(nsparse);**spECK 2020 可变长表**;**Winter 2019 行块自适应表** | 空间省、通用;碰撞/原子代价、表尺寸难定 | hash_spa(256-906 带大胜 0.52×)+ hash_heavy/SPEC99(重行 0.726×,env 门待翻默认) |
| 3 | **ESC:expand-sort-compress** | 展开全部中间积→排序→相邻归并去重 | **Dalton 2015(bhSparse)**;nsparse 的 numeric 段 | 无表无原子、写友好;排序 O(k log k) 贵 | 排序管线(后处理共用);MMSpGEMM 计算核同族 |
| 4 | **多路归并** | 行内 m 条有序链直接 k 路归并(免排序) | MMSpGEMM(PACT'25,分区阶段 tournament tree/Varman rank 断点);Gustavson 行归并思想 | 输入天然有序时免 sort;跨行切块复杂 | **merge3**(列域分桶 warp-merge,3 相位形态,first100 主场 0.614×【09-03 复测】) |
| 5 | **cursor/游标直排** | 行短列窄时游标一次滑过 B 区间 | 散见于短行优化 | 极简极快;出生态位即崩 | cursor(c-64 0.879×/crankseg 0.832×【09-30】)+ 微行族(msc 0.406×) |
| 6 | **混合分域 + smem/gmem 协作** | 短行/长行不同累积器 + 两级显存 | **opSparse 2022**(binning 优化+hash);**HSMU 2025**(专治 hash 的 smem 利用率不足,**338 阵评测**);**ocean 2026**(三类累积器 hybrid) | 承认"没有万能累积器"→ 分域 | 七武器 = 分域的极限形态(行级路由,17 相位) |

### 2.2 负载均衡谱系(行间失衡的解法)

1. **朴素映射**:一行一块,靠硬件调度器兜底(2014 前后)——巨鲸行拖死全场;
2. **行长 binning**:按行长/工作量分桶,桶内同构 → opSparse 的 binning 优化、
   我们的 binning(1c 第一刀把双计数器合一,17a7e00);
3. **等大工作窗(merge-path 系)**:把全部中间积排成全局序列按 rank 切等大块
   → **MMSpGEMM**(全局 prefix scan + tournament tree 精确 rank-p 断点,块可跨行,
   2048 项/块;与我们 merge3 同轴不同机制,详见 inno/mmspgemm_comparison.md);
4. **tiling**:把 dense GEMM 分块思想搬到稀疏 → ASpT(PPoPP'19,稀疏/稠密区域
   分开铺 tile);**TileSpGEMM 2022**(Niu et al.,101+ 引,仓库有源码);
5. **分层分裂**:不规则巨阵专项 → **gMAGNUS**(arXiv 2607.22866,Jul 2026,
   hierarchical multisplit;Wolfson-Pou & Helal,承其 ICS'25 局部性工作)。

### 2.3 四代叙事(kernel 轴的分期)

| 代 | 时段 | 主题 | 代表 |
|---|---|---|---|
| 一代 | 2008-2014 | "能跑":CUDA 稀疏时代开启(Bell&Garland 开山为 SpMV;Liu&Vinter IPDPS'14 首 GPU SpGEMM 之一,两遍范式定型) | Liu&Vinter'14 |
| 二代 | 2015-2020 | "算得匀":累积器谱系成形 + 均衡精细化 | Dalton'15(ESC)、nsparse'17、Winter'19(自适应表)、spECK'20、ASpT'19 |
| 三代 | 2022-2025 | "榨干 smem/局部性":混合分域成共识 | opSparse'22、TileSpGEMM'22、HSMU'25、Wolfson-Pou ICS'25、MMSpGEMM'25 |
| 四代 | 2025-2026 | "换范式":估计替代精确、不规则专项 | ocean ICS'26、gMAGNUS'26 |

旁支:硬件轴(tSparse 2020 tensor core 混合精度;分布式轴 Trident 2026
通信避免——单卡外,一句带过即可)。

---

## 三、调度方法轴:五级阶梯(决策权的迁移史)

| 级 | 名称 | 机制 | 代表 | 局限 |
|---|---|---|---|---|
| 0 | 库级启发式 | 库内部按输入规模切粗档 | cuSPARSE(2020 起 SpGEMM 支持) | 黑盒、粒度粗 |
| 1 | **静态解析阈值** | O(1) 输入统计 → 公式/阈值选路径 | ocean `decideMatrixType`(统计+采样,选 estimation vs symbolic 工作流);我们 8 月的 O(1) 公式(flop_proxy、四变量 logit,95% 准确率【08 口径】) | 单维阈值对边界阵脆(ocean 也承认要有 analysis step 兜) |
| 2 | **离线学习逐阵选择** | 语料训练模型,运行期特征→预测最优 | **IA-SpGEMM(ICS'19)**:MatNet 深度模型,2700+ SuiteSparse 阵,sparse features+density representations→直接预测 format+算法(一手摘要已核)——**逐阵学习分发的祖师** | 单层黑盒;选的是同构两阶段管线的变体;无安全网 |
| 3 | **管线内 ML** | ML 嵌进管线做子决策 | Wang 2025(TACO):ML 预测"结果规模估计方法"的选择(一手摘要已核),单形态 hash 内 | 只有一个决策点 |
| 4 | **运行时自纠** | 估计错了现场补救 | ocean:溢出 retry(§3.2 Overflow Handling) | retry 不带回退目标,慢不是错也可能错 |
| 5 | **形态级分发 + 三层裁定 + 证书** | 整条管线形态三选一;解析可解不学习/证书反馈走状态机/物理不可约才学习;判废证书→形态回退 | **我们(查新背书:无人做过,见 inno/novelty-sweep-dispatch-1015.md)** | 复杂度=论文负担(Phase 2 消融矩阵专为它设计) |

**叙事线(汇报用)**:决策点的迁移 = 编译期(选库)→ 调优期(MatNet 2019)→
运行期逐阵(形式选择)→ 运行期逐相位(估计+证书)。每一级迁移都把更多的
"最坏情况保险"换成"运行时信息",而**正确性从"算法保证"变成"运行时机制保证"
(估计→retry→证书)——这是我们判断的领域走向,ocean 是这个走向的最新公开证据。

---

## 四、2025-2026 前沿定谳(谁在打什么)

| 工作 | 出处 | 核心主张 | 与我们关系 |
|---|---|---|---|
| **ocean** | ICS'26(Cornell,Li&Guidi) | HLL 估计替代 symbolic(28% 税)+ 分析步选工作流 + 三类混合累积器;vs spECK 1.4×/opSparse 2.6×/TileSpGEMM 3.5×/HSMU 2.0×(A100,337 方阵) | 头号对手;哲学同源(估计+自纠),但 retry≠证书,单层门≠三层,单形态≠家族 |
| **MMSpGEMM** | PACT'25(GT HPArch) | 多路归并分区:全局 rank 等大窗 + tournament 断点,离线 split+在线 block-ESC | 与 merge3 思想同轴、先发表、必须引(inno/ 比对档:机制差异大,不构成实现撞车) |
| **HSMU** | 2025(仓库有源码) | 专治 hash 累积器的 smem 利用率不足;新累积器+分立 symbolic;**338 阵评测** | 累积器轴近期代表;注意:**ocean 的 337 与 HSMU 的 338 高度疑似传承**——"337 主场"并非 ocean 独创选择 |
| **Wang 2025** | ACM TACO(BIT) | 行分组 + kernel 配置 + **ML 选估计方法** | 估计器选择的最近前作,必引划界 |
| **Wolfson-Pou ICS'25 → gMAGNUS 2607** | Jul 2026 | 数据局部性生成 → 分层 multisplit 攻不规则巨阵 | 巨阵人群;与我们错位 |
| **Trident** | arXiv 2026 | 通信避免型分区(层级网络/分布式) | 单卡外,引用即可 |

**三条观察(汇报的"领域判断"页)**:
1. **估计思想已经扩散**:ocean 用 HLL 砍 symbolic,Wang 用 ML 选估计器——
   "估计+运行时补救"成为新共识;**但补救停留在 retry,判废证书+形态回退是空白**;
2. **混合分域成为共识**(opSparse/HSMU/ocean 全在混),但都停在**累积器级**;
   **管线形态级**的分发(相位结构整体切换)还没人做;
3. **基准集传承现象**:HSMU 338 → ocean 337(几乎同集)——领域在自己选的
   中大阵舒适区互比;我们 437 并集战场(first100 ∪ ocean337,重叠仅 1 阵)
   是对这种评测惯性的直接回应【10-14 定谳】。

---

## 五、GPU 单机之外:领域四支柱("整个 SpGEMM"的完整视角)

> 前面两轴深挖的是我们的战场(GPU 单机 kernel+调度);要"讲整个领域",
> 还有四根支柱。每根一页深度,细节可按需再扩。

### 支柱 A:估计理论血统(ocean 范式的家谱——估计不是 2026 冒出来的)

```
Cohen 1996/1997(理论起点:稀疏乘积 size-estimation 框架,传递闭包/可达性)
 → Amossen/Campagna/Pagh 2014(Algorithmica:SpGEMM 乘积规模更优估计)
 → Flajolet 2007(HLL 近优基数估计)→ Heule 2013(Google:HLL 工程化)
 → Du 2022b(采样压缩比预测输出结构,Xie 组)→ Wang 2025(ML 选估计器)
 → ocean 2026(HLL 进 SpGEMM 主键;置信界引 Saw 1984 Chebyshev)
```

- **我们在此支柱的位置**:minhash+est 双估计器 + margin 状态机(高斯尾
  1+√(2lnN)·1.04/√K);ocean 用 Chebyshev 类界——**两边都在给估计配统计保证**,
  我们多一层"证书判废→形态回退"的硬保证(retry 只保正确不保性能边界);
- **汇报金句**:估计范式是"90 年代理论 → 2010s 大数据工程 → 2026 进 SpGEMM
  主线"的三级跳,不是单点发明;我们和 ocean 站在同一历史进程的两端。

### 支柱 B:分布式 / 多核 / 多 GPU(单机之外的扩展线)

| 工作 | 层 | 机制 |
|---|---|---|
| Buluç & Gilbert 2012(CombBLAS/CPSpGEMM) | 分布式 | 2D 网格分布 + SUMMA 框架稀疏化,领域奠基 |
| Nagasaka et al. 2019(Parallel Computing 90) | 多核/众核 | 跨架构(x86/KNL/Power)性能建模与优化 |
| KokkosKernels SpGEMM(Deveci et al.,Sandia) | 多核/众核 | hash 累积器族,可移植性能层 |
| Brock, Buluç, Yelick ICS'24 | 多 GPU | RDMA 直连算法,避免中间物化 |
| Bellavita et al. ICS'26(Trident,Guidi 组) | 多 GPU/层级互联 | 三叉分区通信避免(ocean 同组) |

- 与我们关系:单卡内的形态调度与多机扩展**正交可叠加**;Buluç 2025 综述视角
  (arXiv:2508.04077 "The ubiquitous sparse matrix-matrix products")值得一读。

### 支柱 C:应用拉动(这个 kernel 为什么值这么多论文)

- **软件起点**:MATLAB 稀疏矩阵引擎(Gilbert/Moler/Schreiber 1992);
- **AMG 代数多重网格**:粗化 = SpGEMM 自乘(Bell/Dalton/Olson 2012 细粒度并行;
  **我们的 A·A 挑战赛题即同谱系**);
- **图算法**:Markov 聚类 HipMCL(Azad 2018)、三角形计数、传递闭包;
- **基因组学**:序列比对/组装 BELLA、蛋白比对(Selvitopi SC20)——
  **ocean 作者 Guidi 的血统就是应用拉动**(做基因组发现 SpGEMM 是瓶颈→下场做 kernel);
- **GNN/ML**:SpMM 亲戚(通信优化 Tripathy SC20;DGL 生态);稀疏注意力 = 下一个潜在拉动。

### 支柱 D:重排/局部性 + 标准化 + 基准方法论(三条支线)

- **重排轴**("把矩阵变乖再算"):Islam et al. 2025(Buluç 组,重排+聚类分块
  计算,ocean 引文)、Wolfson-Pou ICS'25(数据局部性生成)——与我们正交可叠加;
- **标准化**:GraphBLAS 半环标准(SuiteSparse:GraphBLAS,Davis)把 SpGEMM 定为
  核心算子 mxm;半环泛化(布尔/tropical)让图算法共用同一 kernel 族;
- **基准方法论**:SuiteSparse 语料 = 领域通用货币;HSMU 338 → ocean 337 的传承
  (§四观察③)即发生于此;Gao 2023(CSUR)系统综述 = 现有领域地图。

---

## 六、我们的位置(近期工作,全部盖章)

### 6.1 架构一页图

```
L0 矩阵级:形态三选一 ── merge3(3 相位)/ 轻 hash(4-5,待建)/ 全形态(17→8→5)
              │ 三层裁定:解析门(α·Σflop vs β·Σnnz+γ·A_rows)→ 状态机(margin 证书
              │ 直方图)→ 学习型(只管不可约);配置级 DCEST 开关/margin/估计器
L1 行级:七武器(ESC/批量/hash_spa/hash_heavy/cursor/dense 梯子/k7)按行路由
L2 证书:dense 溢出证书(判废→merge3 回退)/ k7 count 证书 / hash retry
        → 敢判死刑,因为 merge3 永远活着
```

### 6.2 记分牌(唯一权威口径,详见 compare/README.md)

| 战场 | 数字 | 口径/日期 |
|---|---|---|
| ocean337(其主场) | 我 C 1.2714 / A 1.2301 / **oracle 1.1846** | 10-14 同日配对 ocean_337.csv;WARMUP_N=1,含 ~1% 满载税 |
| 三臂内部(净可信) | A/C **−3.45%**,oracle/C **−7.0%**,CD 69-75 阵全场最优 | 10-13/14 批内同条件 |
| first100(挑战赛集) | **0.6224**(赢 81/19) | 七月陈数下界;其后 DCEST/merge−43%/修雷全未计入 |
| **437 并集** | GM ≈ **1.02**(保守) | 10-14 外推;权威批待跑 |
| kernel 级同行集 | 去 dense 路由阵后 **0.927 ✅** | 10-05 定谳层(更新于 docs/79 的 09-30 层) |

### 6.3 近两周工程战果

- **DCEST V3** 落地(fc49855):dense_count 相位消灭,est 配额+溢出证书+双 buffer
  紧凑拷贝;屏 18/20 净赢(wb-edu −45.3%);337 门控定谳=niche(69 阵主场)【10-12/13】
- **mh_merge −43%**:指令台账取证(578 指令/行)→ PPT×EST_CONT 模板化重写【10-13,study/41】
- **修雷四连 + 审计 25 条**(study/43):tmp_val 未清零/FORCE_DITER_EST 写序/
  mempool 溢出改回退/#endif 吞行;P0 修后 WRITE_MTX 雷退役
- **1c 两刀落地**:est_collapse_tail 三处合一 + binning 双计数器合一
  (17a7e00/823ed00),守门值级+memcheck 全绿;计时验证等 GPU【10-14】
- **查新定谳**(inno/novelty-sweep-dispatch-1015.md):可主张 novelty 四条
  (形态级分发/三层裁定/证书链/否定结果);IA-SpGEMM 与 Wang 2025 划界完成

### 6.4 独立发现(论文素材,成败都有)

1. **否定结果**:逐行 kernel 时间 ≠ 系统成本(集体效应/launch 税/占用交互),
   337 数据背书【10-10,study/37】;
2. **SiO2 翻面首证**:修管线后最优调度会变(超全 hash A 的解被 on 配置翻面)——
   "先修管线再跑 L0"的方法论依据【10-12】;
3. **并集战场**:两大基准几乎不相交,领域在自己选的战场互比【10-14】。

---

## 七、周三汇报 slide 大纲建议(13 页)

1. **封面+一句话**:SpGEMM on GPU——从"怎么算"到"怎么选"
2. **问题**:两个根本矛盾(输出未知→symbolic 28% 税;行间失衡×行内去重)+ 硬件不变量
3. **Kernel 轴 I**:累积器六谱系(2.1 表精简版,配一行"我们的对应")
4. **Kernel 轴 II**:均衡四板斧 + 四代分期(2.2/2.3 表)
5. **调度轴**:五级阶梯(§三表)+ "决策权迁移"箭头图
6. **前沿 25-26**:ocean/MMSpGEMM/HSMU/gMAGNUS 四格 + 三条观察
7. **场外四支柱**(一页四格):估计理论家谱(Cohen→Amossen→HLL→ocean)/分布式·多核
   (CombBLAS→RDMA/Trident)/应用拉动(MATLAB→AMG→图→GNN;Guidi 血统故事)/重排+GraphBLAS
8. **我们的架构**:三形态×三层×七武器×证书一页图
9. **记分牌**:§6.2 表(强调口径纪律:同日同口径配对)
10. **近期工程**:DCEST/mh_merge −43%/修雷四连/1c(§6.3 精简)
11. **查新定谳**:四条 novelty + 三件前作划界(体现文献掌握)
12. **独立发现**:否定结果/SiO2 翻面/并集战场(§6.4)
13. **下一步**:权威 437 批(GPU 释放即跑)→ L0/L1 训练 → 论文四节

**讲述主线建议**:用"决策权迁移"串全场(领域史→我们的三层=迁移的终点站),
用"并集战场"收尾(评测立场无懈可击),用"否定结果"体现深度。

---

## 八、参考文献(★=已核一手摘要/贡献清单/源码;☆=自 ocean 引文表核得完整书目)

**GPU 单机主线(§二/§三/§四)**
1. ★ Gustavson 1978, *Two fast algorithms for sparse matrices*, TOMS 4(3)
2. Bell & Garland 2008-09, *Efficient SpMV on CUDA*(NVIDIA TR;稀疏 CUDA 开山)
3. ☆ Gilbert, Moler, Schreiber 1992, *Sparse matrices in MATLAB*, SIMAX(软件起点)
4. ★ Liu & Vinter, IPDPS 2014(首 GPU SpGEMM 之一,两遍范式)
5. ☆ Anh, Fan, Wen, ICS 2016, *Balanced hashing and efficient GPU SpGEMM*
6. ☆ Nagasaka, Nukada, Matsuoka, ICPP 2017, nsparse(Pascal;行并行+排序归约)
7. ★ Dalton, Baxter, Merrill, Olson, Garland, IPDPS 2015, *Optimizing sparse matrix operations on GPUs using merge path*(bhSparse;源码在 external_sota/)
8. ★ Winter, Mlakar, Zayer, Seidel, Steinberger, PPoPP 2019(product/reduction 两段+行块自适应 hash 表;TU Graz)
9. Hong et al., PPoPP 2019, ASpT(自适应稀疏 tiling,SpMM 系)
10. ★ Xie, Tan, Liu, Sun, ICS 2019, *IA-SpGEMM*(MatNet 深度模型,2700+ 阵,逐阵预测 format+算法;一手摘要已核)
11. ★ Parger, Winter, Mlakar, Steinberger, PPoPP 2020, spECK(*through lightweight analysis*;可变长 hash 累积器)
12. tSparse 2020(ETH;tensor core 混合精度,旁支)
13. ☆ Du, Guan, Guan, Niu, Huang, Zheng, Xie, IEEE Access 10, 2022, *OpSparse*(binning smem 优化+hash;**与 IA-SpGEMM 同 Xie 组**,ssslab 同源)
14. ★ Niu, Lu, Ji, Song, Jin, Liu, PPoPP 2022, TileSpGEMM(稀疏 tiling;源码在 external_sota/)
15. ★ Wu, Luo, Li, Zhang, Tang, Li, Zhang, Liu, **HPCA 2025**, *HSMU-SpGEMM*(高 smem 利用率累积器+分立 symbolic;338 阵集;README 一手,源码在 external_sota/)
16. ★ Wang, Lin, Wei, Gao, Ji, ACM TACO 2025, DOI 10.1145/3774654(ML 选估计方法;一手摘要已核)
17. Wolfson-Pou, Laukemann, Petrini, ICS 2025(数据局部性生成)
18. ★ MMSpGEMM, PACT 2025(GT HPArch;多路归并分区;源码+比对档 inno/)
19. ★ ocean: Li & Guidi, ICS 2026, arXiv:2604.19004(HLL 估计范式;一手贡献清单已核)
20. ★ gMAGNUS: Wolfson-Pou & Helal, arXiv:2607.22866(2026-07;分层 multisplit)
21. ☆ Islam, Xu, Dai, Buluç, 2025, arXiv:2507.21253(重排+聚类分块计算)

**估计理论血统(§五支柱 A)**
22. ☆ Cohen 1996(IPCO)/1997(JCSS),稀疏乘积 size-estimation 框架
23. ☆ Amossen, Campagna, Pagh, Algorithmica 69(3), 2014, *Better Size Estimation for Sparse Matrix Products*
24. ☆ Flajolet et al. 2007(HLL 理论);Heule et al. 2013(Google,HLL 工程化)
25. ☆ Du et al. 2022b, arXiv:2207.13848, *Predicting the Output Structure of SpGEMM with Sampled Compression Ratio*(Xie 组)
26. ☆ Saw, Yang, Mo 1984(Chebyshev 不等式;ocean 置信界引)

**分布式/多核(§五支柱 B)**
27. Buluç & Gilbert 2012, CombBLAS/CPSpGEMM(JPDC;2D 分布+SUMMA 稀疏化)
28. ☆ Nagasaka, Matsuoka, Azad, Buluç, Parallel Computing 90, 2019(多核/众核建模)
29. KokkosKernels SpGEMM(Deveci et al., Sandia;可移植 hash 累积器族)
30. ☆ Brock, Buluç, Yelick, ICS 2024(RDMA 多 GPU)
31. ☆ Bellavita, Pichetti, Pasquali, Vella, Guidi, ICS 2026, *Trident*(通信避免三叉分区;ocean 同组)

**应用/标准化/综述(§五支柱 C/D)**
32. ☆ Bell, Dalton, Olson, SISC 2012(AMG 细粒度并行);☆ Azad et al. 2018 HipMCL
33. ☆ Guidi et al. 2021a/b(BELLA/基因组);Selvitopi et al. SC20(蛋白比对);Tripathy et al. SC20(GNN 通信)
34. GraphBLAS / SuiteSparse:GraphBLAS(Davis;半环标准)
35. ☆ Gao, Ji, Chang, Han, Wei, Liu, Wang, ACM Comput. Surv. 55(12), 2023, *A Systematic Survey of General Sparse Matrix-matrix Multiplication*(领域现有地图)
36. ☆ Buluç 2025, arXiv:2508.04077, *The ubiquitous sparse matrix-matrix products*(视角综述)

**调度轴理论挂靠**:Rice 1976(算法选择问题);SATzilla;OSKI(Vuduc 2005)

> 与本文配套的划界档:inno/novelty-sweep-dispatch-1015.md(查新)、
> inno/mmspgemm_comparison.md(MMSpGEMM 比对)、docs/79-kernel-landscape-explained.md
> (我方 kernel 生态)、compare/README.md(数据口径 manifest)。
