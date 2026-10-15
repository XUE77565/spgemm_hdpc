# SpGEMM 技术发展全脉络(1978-2026)

> **这是什么**:面向 10-21 汇报的领域史独立文档——只讲领域,不讲我们
> (我们的位置见同目录 02-field-survey.md §六)。
> **写法**:每个时代回答三个问题——解决了什么 / 靠什么解决的 / 给下一个时代留下了什么问题。
> 链式叙事:**上一代的遗留问题 = 下一代的发动机**。
> **证据分级**:★=一手已核(摘要/贡献清单/源码);☆=自 ocean 引文表核得书目;其余=多源二手交叉。
> 日期:2026-10-15。

---

## 〇、一页年表(总览)

| 年 | 工作 | 轴 | 一句话 |
|---|---|---|---|
| 1978 | Gustavson ★ | 算法 | 行式 SpGEMM 母体:输出行=稀疏向量累积 |
| 1992 | Gilbert/Moler/Schreiber ☆ | 软件 | MATLAB 稀疏引擎,SpGEMM 进生产力工具 |
| 1996-97 | Cohen ☆ | 理论 | 稀疏乘积规模估计框架(传递闭包/可达性)——估计范式理论种子 |
| 2007 | Flajolet et al. ☆ | 理论 | HyperLogLog 近优基数估计 |
| 2008-09 | Bell & Garland | GPU | 稀疏 CUDA 开山(SpMV/格式谱系) |
| 2012 | Buluç & Gilbert | 分布式 | CombBLAS:2D 分布+SUMMA 稀疏化,分布式 SpGEMM 奠基 |
| 2012 | Bell/Dalton/Olson ☆ | 应用 | AMG 细粒度并行(粗化=SpGEMM 自乘) |
| 2013 | Heule et al.(Google)☆ | 工程 | HLL 生产工程化 |
| 2014 | Liu & Vinter ★ | GPU | 首 GPU SpGEMM 之一;**两遍范式(symbolic+numeric)定型** |
| 2014 | Amossen/Campagna/Pagh ☆ | 理论 | SpGEMM 乘积规模更优估计算法(Algorithmica) |
| 2015 | Dalton et al. ★ | GPU | bhSparse:merge-path 均衡 + ESC(expand-sort-compress)谱系 |
| 2016 | Anh/Fan/Wen ☆ | GPU | 平衡 hash GPU SpGEMM |
| 2017 | Nagasaka et al. ☆ | GPU | nsparse:行并行+排序归约,Pascal 优化 |
| 2018 | Azad et al. ☆ | 应用 | HipMCL:Markov 聚类大规模化(SpGEMM 为心) |
| 2019 | Winter et al. ★ | GPU | product/reduction 两段+**行块自适应 hash 表** |
| 2019 | Hong et al. | GPU | ASpT 自适应稀疏 tiling(稀疏/稠密分区铺瓦) |
| 2019 | **Xie et al.(IA-SpGEMM)★** | **调度** | **MatNet 深度模型,2700+ 阵,逐阵预测 format+算法——逐阵学习分发祖师** |
| 2019 | Nagasaka et al. ☆ | 多核 | 跨架构(x86/KNL/Power)性能建模(Parallel Computing 90) |
| 2020 | Parger et al.(spECK)★ | GPU | 可变长 hash 累积器+"lightweight analysis" |
| 2020 | tSparse | 硬件 | tensor core 混合精度(旁支) |
| 2020 | Selvitopi/Tripathy SC20 ☆ | 应用 | 蛋白比对/GNN 通信——HPC 与 ML 两端拉动 |
| 2022 | Du et al.(opSparse)☆ | GPU | binning 的 smem 利用率优化+hash 优化(IEEE Access;Xie 组) |
| 2022 | Niu et al.(TileSpGEMM)★ | GPU | 稀疏 tiling 成熟(PPoPP;101+ 引) |
| 2022 | Du et al. b ☆ | 估计 | 采样压缩比预测输出结构(arXiv:2207.13848;Xie 组) |
| 2023 | Gao et al. ☆ | 综述 | CSUR 系统综述——领域已有官方地图 |
| 2024 | Brock/Buluç/Yelick ☆ | 多GPU | RDMA 直连算法(ICS) |
| 2025 | Wu et al.(HSMU)★ | GPU | 高 smem 利用率累积器(**HPCA**;338 阵集) |
| 2025 | Wang et al. ★ | 调度 | **ML 选估计方法**(TACO;单形态内子决策) |
| 2025 | Wolfson-Pou et al. | GPU | 数据局部性生成(ICS) |
| 2025 | MMSpGEMM ★ | GPU | 多路归并分区:tournament 精确 rank 断点(PACT) |
| 2025 | Islam et al. ☆ | GPU | 重排+聚类分块计算(Buluç 组) |
| 2025 | Buluç ☆ | 综述 | "The ubiquitous sparse matrix-matrix products"视角文 |
| **2026** | **ocean(Li & Guidi)★** | **范式** | **HLL 估计替代 symbolic(28% 税)+分析步选工作流+混合累积器(ICS)** |
| 2026 | Bellavita et al.(Trident)☆ | 多GPU | 三叉分区通信避免(ocean 同组,ICS) |
| 2026 | gMAGNUS ★ | GPU | 分层 multisplit 攻不规则巨阵 |

---

## 一、奠基期(1978-2007):一个算法、两颗种子

**解决了什么**:定义问题并给出母体算法。

- **Gustavson 1978**(TOMS)★:C 的第 i 行 = A_i 的非零所对应 B 行的"稀疏向量累积"。
  按输出行组织 → 天然行并行;行内"找同列→累加"= **累积器(accumulator)抽象**诞生。
  此后 48 年所有工作的本质 = 换不同的累积器 + 换不同的行分配方式;
- **MATLAB 1992** ☆(Gilbert/Moler/Schreiber):稀疏矩阵进生产力工具,SpGEMM 从论文题变成工程题;
- **两颗理论种子**:Cohen 1996/97 ☆ 的稀疏乘积规模估计框架(为图算法传递闭包而生);
  Flajolet 2007 ☆ 的 HyperLogLog(为大数据基数估计而生)。
  **两颗种子当时互不相识,2026 年在 ocean 里合流**——这是全脉络最长的一条伏笔线。

**遗留问题**:单机 CPU 性能到顶;真实矩阵的稀疏结构千奇百怪,"一种数据结构走天下"开始吃力。

## 二、CUDA 开荒期(2008-2014):两遍范式的定型

**解决了什么**:把 Gustavson 搬上 GPU,并定型了此后十年的标准结构。

- **Bell & Garland 2008-09**:稀疏 CUDA 时代开启(CSR/ELL/COO/HYB 格式谱系;SpMV 开山);
- **Liu & Vinter IPDPS 2014** ★:GPU SpGEMM 代表作之一——负载均衡+结果预分配+并行插入,
  **symbolic(数每行 nnz)+numeric(算数)两遍范式从此定型**;
- 分布式侧:**Buluç & Gilbert 2012**(CombBLAS)把 dense 的 2D 分布+SUMMA 框架稀疏化,
  为多机世界立了地基(与 GPU 单机线并行发展至今)。

**遗留问题(下一代的发动机)**:
① symbolic 那一遍"只数不算"却平均吃掉 **~28%** 时间(ocean 2026 实测)——精确的代价被记录在案;
② 行长跨数量级,朴素"一行一块"负载分配在巨鲸行上崩盘;
③ 累积器放哪(smem/gmem)、什么形态(hash/数组/排序)——三岔路口出现。

## 三、谱系爆发期(2015-2020):五条路线各自成军

**解决了什么**:把 ②③ 两个遗留问题拆开各攻——累积器谱系和均衡技术同时成形。

**累积器路线**:
- **ESC 谱系**(Dalton 2015 ★,bhSparse):expand 全部中间积→块内排序→相邻归并去重。
  免表免原子,代价是排序;merge-path 同时被引入做等大切块;
- **hash 谱系**:Anh 2016 ☆(平衡 hash)→ nsparse 2017 ☆(行并行+排序归约,Pascal)→
  **Winter 2019 ★**(product/reduction 两段,行块自适应定表)→ **spECK 2020 ★**
  (可变长 hash 表,"lightweight analysis" 定容量)。TU Graz 的 Winter/Mlakar/Steinberger
  组连出两作,hash 路线的主干在此期立起;
- **tiling 谱系**:ASpT 2019(稀疏/稠密区域分开铺瓦)→ TileSpGEMM 2022 ★ 成熟。

**调度路线的第一枪**:
- **IA-SpGEMM(Xie et al., ICS 2019)★**:MatNet 深度模型,SuiteSparse 2700+ 阵训练,
  sparse features + density representations → **逐阵预测最优 format+算法**。
  "input-aware"的旗帜立起来了——但它是**单层黑盒**,且选择粒度是同构管线的变体。

**遗留问题**:
① 每条路线都有自己的甜区,没有万能累积器成为共识——**但大家默认"选一次,全程用"**;
② hash 表容量怎么定依然靠经验/轻分析——估计理论(Cohen/Amossen)还在图论界沉睡;
③ 中小阵上"仪式开销"(launch/预处理)占比越来越大——被大阵论文系统性忽略。

## 四、榨取与混合期(2022-2025):共识收敛,三个方向同时收口

**解决了什么**:把第三代各路线榨到硬件极限,并开始"混合"。

- **opSparse 2022** ☆(Du et al.,IEEE Access,Xie 组):binning 的 smem 利用率优化——
  行长分桶 + 桶内同构kernel,**"混合分域"思想的工程化**;
- **HSMU 2025 ★**(Wu et al.,**HPCA**):专治 hash 累积器的 smem 浪费,新累积器+分立
  symbolic;评测用 **338 阵集**;
- **MMSpGEMM 2025 ★**(PACT):多路归并分区——全局中间积按值序等大 rank 窗切块
  (tournament tree 精确断点),离线 split+在线块内 ESC;
- **局部性方向**:Wolfson-Pou ICS 2025(生成数据局部性)、Islam 2025 ☆(Buluç 组,
  重排+聚类分块)——"把矩阵变乖再算";
- **估计方向苏醒**:Du 2022b ☆(采样压缩比预测输出结构,Xie 组)——理论种子开始发芽;
- **综述收口**:Gao 2023 ☆(CSUR)给领域画了官方地图。

**遗留问题(直通 2026)**:
① 精确 symbolic 的 28% 税还在——**没人真正砍它**;
② 混合分域停在累积器级/行长维——**整个管线形态不可选**;
③ 估计一旦出错怎么办——没人给估计配"安全网"。

## 五、范式转变期(2025-2026):估计上位,调度补完

**解决了什么**:三代遗留的 ①③ 两问同时被攻击。

- **ocean(ICS 2026)★**(Li & Guidi,Cornell):三连击——
  **HLL 估计替代精确 symbolic**(Cohen→Amossen→Flajolet→Heule 四十年伏笔线的收网);
  **分析步**用输入统计+采样预测估计成本,动态选 estimation-based vs symbolic-based 工作流;
  **三类混合累积器**。自纠=溢出 retry。337 阵评测(vs spECK 1.4×/opSparse 2.6×/
  TileSpGEMM 3.5×/HSMU 2.0×);
- **Wang 2025 ★**(TACO):ML 进管线做子决策(选估计方法)——调度轴的"管线内 ML"级;
- **gMAGNUS 2026 ★**:分层 multisplit 攻不规则巨阵;**Trident 2026** ☆(Bellavita+Guidi,
  ocean 同组):多 GPU 通信避免——ocean 组同时在单机范式和多机扩展两线作战;
- **基准集传承现象**:HSMU 338 → ocean 337——领域在中大阵舒适区内部互比成为惯性。

**当前的空白(领域留给下一个时代的问题)**:
1. 补救机制停留在 **retry**——估计错了只是重跑,**没有判废证书链、没有跨形态回退目标**;
2. 分域停在累积器级——**管线形态级**(相位结构整体切换)的分发不存在;
3. 调度全是单层(阈值或单一模型)——**能解析的不解析、该状态机的没反馈**,
   分层裁定原则没人提出;
4. 评测停在自选战场——双基准几乎不相交的问题没人指出。

> (以上四条空白 = 我们工作的入口;此文档只立脉络,主张部分见 02-field-survey.md。)

---

## 六、三条贯穿线(把 48 年收进三句话)

1. **Kernel 线:一分为五,五合于混**——累积器从 Gustavson 的隐式 SPA 分化出
   dense/hash/ESC/归并/cursor 五谱系,2022 年后又收敛为"混合分域"共识;
   均衡从朴素映射走到 binning→merge-path 等大窗→tiling→分层分裂;
2. **调度线:决策权持续迁移**——编译期(选库)→ 调优期(IA-SpGEMM 2019 逐阵预测)→
   运行期选工作流(ocean 分析步)→ 运行期子决策(Wang 2025);每迁一步,
   更多"最坏情况保险"被换成"运行时信息";
3. **精确性线:精确 → 估计 + 运行时保证**——symbolic 的 28% 税终于被砍,
   代价是正确性从"算法保证"变成"运行时机制保证"(retry);这条线的下一站
   是"带证书的估计"(判废证据链+回退目标)——目前空白。

**一句话总结全脉络**:SpGEMM 四十八年 = 先把 Gustavson 的行并行在 GPU 上榨干
(谱系+均衡+混合),再把"决策"从设计期搬进运行期(格式→算法→工作流→相位),
现在轮到把"精确"换成"估计+保证"——三线各自走到最后一格,而最后一格同时是
形态调度与证书安全的空白地带。
