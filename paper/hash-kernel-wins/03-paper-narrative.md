# 论文级创新点叙述（hash kernel 线，2026-09-18 定格）

> 用法：§1 英文 contributions 块可直接进 paper intro；§2 每条贡献的完整叙述
> （Challenge → Insight → Mechanism → Evaluation → Novelty）是正文技术段的骨架；
> §4 新颖性防御矩阵备 rebuttal；所有数字与 `study/legacy-inno/hash_opt_2026/`、根 `76/77`、
> `paper/hash-kernel-wins/01` 台账逐一对得上。

---

## 0. 主论题（Thesis）

**在 SpGEMM 中，累加器不是一种算法选择，而是一个逐行的经济决策；它的正确答案可以从
硬件不变量推导出来，且切换代价可以做到零。**

SpGEMM 的输出行总体在基数（est）、列跨度（span）、操作数长度（a_len）、重复度（dup）上
横跨四个数量级。每个累加器家族——hash 表、dense 窗口、merge、counting——的成本都是这些
特征的闭式函数，而平衡点是硬件常数（字节足迹等式、WARP_SIZE、占用悬崖的 32 寄存器线）。
既有系统在矩阵级（nsparse/spECK 单法选择）或 workflow 级（Ocean 三工作流+七档）提交选择，
选错一代价高达 26×（矩阵级实测）或 4×（行级实测：TSOPF 错送 dense 窗 22.2ms vs 正确
hash 路由 5.6ms）。我们把决策下沉到行级，用多维成本模型驱动，把模式切换融进同一 kernel
使其免费，用无偏估计器压缩估计误差、行级 retry 有界兜底残差。结果：在 Ocean 的主场矩阵
上取得系统级首胜（c-64 0.981×），全刀弧线四阵全胜。

---

## 1. Contributions（英文，论文 intro 直接用）

> This work makes the following contributions:
>
> - **Per-row accumulator economics.** We cast accumulator selection in SpGEMM — hash table, dense window, or counting — as per-row cost minimization over cheaply measurable features (estimated output cardinality, column span, operand length, duplication factor). The balance points are derived from hardware invariants rather than tuned thresholds: byte-footprint equality between window and table (`span·8.125B ≤ est·12B`), a WARP_SIZE boundary at `a_len = 32`, and a counting/table equilibrium at duplication ≈ 2. A falsification archive of four single-feature gates — each defeated by counterexamples — motivates the multi-dimensional model. (in-2004 −30.1% and wb-edu −48.5% over our prior release, paired same-window binaries.)
> - **Cursor-resumable dense accumulation.** Wide rows are processed in column windows; instead of re-locating contributing operand rows at every window (binary search from scratch, as in Ocean), we persist a per-chain scan cursor in shared memory and resume. A 4-byte-per-column bookmark converts re-search into resume — to our knowledge absent from prior GPU SpGEMM dense kernels — and recovers part of the 8× bitmap-density advantage of dedicated dense kernels. (vsp −56%, mult_dcop −48%, bloweya −47%.)
> - **Register occupancy as a first-class dispatch dimension.** We identify the register-file cliff on H100: at 1024 threads per block, exceeding 32 registers per thread halves resident warps — a step function invisible to shared-memory budgeting and the only discontinuous occupancy constraint. We formalize the register-compression trade as an occupancy identity, *net throughput = occupancy gain ÷ spill tax* (2 ÷ 1.47 = 1.36×; predicted 13.6 ms vs measured 13.8 ms), and characterize the spill tax itself as address-arithmetic recompute inflation (+32% instructions), not bandwidth dilution. Three one-line `__launch_bounds__` corrections yield c-64 −19% (the system-level crossover vs Ocean), a second kernel winning on all four guard matrices, and a counting-sort variant at 75% occupancy where the radix variant achieves 12.5%. To our knowledge, register residency enters the SpGEMM routing calculus for the first time; it also retrodicts a +1071% anomaly archived months earlier.
> - **Zero-table counting accumulation for short rows.** For rows with ≤ 32 products, each product's output rank equals the population count of smaller keys in the warp: direct write, no table, no zeroing, and output ordered by construction — eliminating both the table cost and the extraction-sort pass. The governing economics, counting ∝ flop vs table ∝ est, has a measured equilibrium at duplication ≈ 2.
> - **An unbiased cardinality estimator with bounded-error row retry.** We replace the Jensen-biased harmonic-mean MinHash estimator with an arithmetic unbiased form — the single largest win of our optimization campaign — removing an entire failure cascade (table bloat → memory wall → full-table contention; Σest overflow → pointer corruption). Residual estimation error is corrected by row-level retry at flop-provisioned table sizes, bounding the cost of any wrong routing decision to one re-execution of the affected rows (0.58 ms fixed cost on our closest matrix).
> - **A conservative evaluation protocol.** Phase-sum timing that excludes host gaps (all timing asymmetries favor the baseline), paired dual-binary same-window medians instead of cross-day means (±1% machine drift demonstrated), and round-over-round reproduction to the millisecond digit.

---

## 2. 各贡献的完整叙述

### C1 行级累加器经济学（Per-row accumulator economics）

**Challenge。** 累加占 SpGEMM 运行时 ~72%（画像实测，pre.md Slide 3）。行总体异质性横跨
四个数量级，任何单一累加器只在一个 regime 最优：我们实测行级错配代价 4×（TSOPF 行群
错送 dense 窗 22.2ms，正确 hash 路由 5.6ms，根 76 §十）；矩阵级错配代价至 26×（pre.md）。
既有系统的选择粒度都粗于行：nsparse/spECK 按矩阵选单法；Ocean 按矩阵统计三选一 workflow、
行再落七档。

**Insight。** 每个累加器家族的每行成本是可测特征的闭式函数：hash 表成本 ∝ est（表几何）、
dense 窗成本 ∝ span（窗数×重扫）、counting 成本 ∝ flop、cursor 增强后的 dense 成本 ∝
est + a_len。**平衡点不是调出来的阈值，是硬件不变量**：①窗口/表字节足迹等式
`span·8.125B ≤ est·12B`（Ocean 档位号比较的连续形式，我们结构化移植为双钥比较）；②
`a_len = 32` 边界 = WARP_SIZE 硬件常数（28,440 行双内核逐行周期数据实证：≤46 老内核
97-100% 赢、≥142 结构内核统治，根 76 §26b）；③counting/表平衡点 dup ≈ 2（germany
dup 1.4 vs patents dup ≥ 2 的分界 = 成本模型的教科书实证，根 76 §9）。

**Mechanism。** 20 桶流水线 + 双侧路由（count 侧与 numeric 侧各自分流，**路由永不驱逐
老赢家**——v33 早期版本把 19,444 个宽行逐出 cursor 内核导致 +95% 回归，人口普查
DUMP_ROUTING 定位后修复为 per-tier 发射）；host 侧重排行一律按档位索引稳定排序保住
CSR 段内原序（L2 局部性，根 76 §24）。DISPATCH_MODEL 将判据结构化（disp_train/disp_synth
训练，first100 作 oracle 验收）。

**Evidence。** 单维门四连证伪档案（饱和门/dup 门/密度门/全档缩窗，每个都有反例阵）；
多维模型下 in-2004 −30.1%、wb-edu −48.5%、rajat20 −19%（配对口径）。

**Novelty。** vs **Misam**（决策树从矩阵统计选 dataflow，硬件侧）——我方行级、平衡点
来自硬件不变量；vs **Ocean** 档位号单比较——我方多维键+不驱逐原则；vs **nsparse/spECK**
单法选择——我方行级分箱。据我们所知，成本维度还首次包含寄存器占用（C3）。

### C2 游标可续扫 dense 累加（Cursor-resumable dense accumulation）

**Challenge。** 列跨度超过 SMEM 窗预算的宽行必须分窗计算。每换一窗，乘积来自哪些操作数
行要重新定位：Ocean 的 denseNumericIter 每窗从头二分重搜，代价 ∝ 窗数 × 搜索长。

**Insight。** 同一条 CSR 链内的乘积列号单调——扫描位置是可续的状态，不是每窗重算的
函数。一个 4B/链的书签把"每窗重搜"变成"从停点续扫"。行越宽、窗越多，书签越值钱。

**Mechanism。** 窗内每列布局 val 8B + flag 1B + pref 4B + cursor 4B（我方 17B/列 vs
Ocean 9B/列——多付的字节买的正是可续扫性）；游标随窗推进在 SMEM 中持久化。

**Evidence。** vsp −56% / mult_dcop −48% / bloweya −47%（study/legacy-inno/30）；c-64 的 19,444 个
宽行 99.5% 走 cursor 模式、承载 96.4% 输出（study/legacy-inno/82）。

**Novelty。** Ocean dense 内核无此机制（根 77 第二章：我方在 dense 侧唯一的机制性胜出项，
"部分对消其 8× 位图密度优势"）；docs/30 系列为原创出处，未见于此前后作（保守表述：
to our knowledge）。

### C3 寄存器占用：第一类公民成本维度（Register occupancy as a dispatch dimension）

**Challenge。** 我们的 cursor 内核比 Ocean 的 dense 内核**少发 14% 指令（4.01G vs 4.67G），
却慢 31%**（26.9 vs 20.6ms，同阵同法 ncu 头对头）。指令数优化在此失效。

**Insight（发现）。** H100 每 SM 64K 个 32 位寄存器、2048 线程驻留上限。1024 线程的块
要 2 块/SM 驻留，当且仅当 REG ≤ 32（1024×32×2 = 65536 恰好）。REG=50 时
65536/(1024×50) = 1.28 → 1 块/SM = **50% warp 驻留的阶跃损失**。SMEM 预算给足也无济于事
（我们 113KB×2 ≤ 228KB 设计本就按 2 块配）——**寄存器是 SMEM 与线程数之外的第三占用
约束，且是唯一不连续的那个**。ncu 实测我方 warps 49.97% vs 彼 99.67%，与 cuobjdump
REG 50 vs 32 互为铁证。

**Mechanism（占用恒等式）。** 越崖压缩寄存器的净收益 = 占用增益 ÷ 溢出税：
`2 ÷ 1.47 = 1.36×`。预测 dense_direct 18.5×1.47/2 = 13.6ms，实测 13.8ms——物理闭环。
溢出税本身被解剖定型：+47%/行、cursor 与 dense 两模式同幅（模式分层账）= 纯溢出税；
ncu 指令 +32% = **地址算术重算型膨胀**（SASS：LDG 68=68、LDS 反降、LDL +10），
非带宽分摊型。刀法 = 一行 `__launch_bounds__(1024, 2)`，cuobjdump 一行审计定位靶点。

**Evidence。** 三次兑现：dense_direct REG 50→32 → c-64 全阵 −19% → **0.981 系统首胜**；
hash_spa REG 40→32 → 四阵全胜（F2 −16% 累计）；csort_counting REG 40（6 块 75% 驻留）
替换 radix REG 229（1 块 12.5%）。**回溯预言力**：同一物理解释了数月前存档的 LADDER
c-64 鲸鱼 +1071%（>4K 槽表）——可回溯旧异常是模型科学性的检验。

**Novelty。** 既有 SpGEMM 路由/成本模型只考虑访存与工作量；寄存器驻留进入路由计算
据我们所知属首次（③阶段 occgate 即把它写进 DISPATCH_MODEL）。占用恒等式给出
"压缩寄存器是否划算"的事前判据，替代试错调参。

### C4 零表计数累加（Zero-table counting accumulation）

**Challenge。** flop ≤ 32 的短行（road_usa 24M 行 avg 乘积 6.7）在表累加器下付出 ∝ est
的表成本与清零成本；hash 提取天然无序，还要付一道提取排序税（road 族 compact 差距 23%
的来源）。

**Insight。** 短行的输出位次可以在寄存器空间内直接算出：每个乘积的位次 =
popc(warp 内比我小的 key)；leader 求和直写。**零表、零清零、输出按构造有序**——表成本
与排序税同时消失。经济学判据：counting ∝ flop vs 表 ∝ est，平衡点 dup ≈ 2。

**Mechanism & Evidence。** 门演化 v2（flop ≤ 32）→ v3（est 门，误杀 germany）→
v4（**dup ≤ 2 终判据**，germany dup 1.4 / patents dup ≥ 2 分界实证）；compact+sort 侧
csort_counting 以 75% 驻留替换 radix 的 12.5%（crankseg −3.3 / F2 −5.7）。

**Novelty（血缘声明）。** 位次计数机制源自我们自己的 csort；Ocean 的 ESC 内核证明了
该机制放在 accumulate 位置成立（根 76 §五）——借鉴的是位置，机制与门控自有。

### C5 无偏估计器 + 有界行级重试（Unbiased estimator + bounded retry）

**Challenge。** MinHash 基数估计的调和平均形式带 Jensen 偏差（~ln 倍高估）。高估 est
级联出三类灾难：表膨胀 → 内存墙 + 满载原子争用（exdata_1 17.85×、TSOPF 20× 落后）；
Σest 爆 int → 偏移绕负 → 野指针写（TSOPF 147 亿教训）；旧版钳 HASH_CAP 掩盖 →
67 阵回退 merge3 → 28.5× 灾难。

**Insight & Mechanism。** 算术无偏式替代调和式（本战役最大单项收益，study/legacy-inno/03）；
est > span 的高估怪物行存在 → 任何"est 定表"路径都要防高估放大（六档梯内化）。
残差估计误差由**行级 retry** 有界兜底：欠估行按 flop 定表重跑，错误路由决策的代价
被界定为受影响行的一次重执行（cage15 全局重炮 16GB 教训 → 行级轻量兜底）。

**Evidence & 升级方向。** F2 固定 retry 成本 0.58ms = 距系统跨线的最后一刀；
设计模板将采用 Harmonia 的 Gain > α·Cost 成本门 + 滞回（连续 T 周期超阈才触发）+
最坏有界回退（study/23 §6）。

### C7 连续估计驱动的表配准（est-continuity table provisioning，09-21 补——kernel 级三阵全胜的根基）

**Insight。** 表长若按 pow2 档离散取值，行的负载因子 LF 随档位跳变（0.3↔0.6），链长方差被档位边界放大；est 从离散档改**连续插值**配表后，行行 LF 收敛到均匀 ≈0.37，探测链系统性变短。
**Evidence（09-18 凌晨，口径 C 全对齐，study/21+22）**。kernel 级 vs Ocean hashNumeric 三阵全胜：F2 0.356（胜率 94.1%）/ c-64 0.906（57.1%）/ crankseg 0.581（91.1%）；对照方素数表 LF 0.57、链 2-3 跳、SMEM 停顿 36% vs 我方 21%——"用停顿换指令"在 warp 并行藏延迟下净赚。
**Novelty。** 估计精度不止用于免溢出，还作为**表配准的控制信号**（estimation → provisioning 闭环）；与 C5（无偏性）和行级置信度余量（σ=1.04/√k，待收割）构成一条估计器信息复用链。

### C6 保守评估协议（Conservative evaluation protocol）

- **相位和口径**：HashProf 相位 cudaEvent 求和（不含 host 间隙）vs Ocean 单括号全程
  （内部 host 空洞全计）——审计出的三处不对称**全部偏向对方**，所报比值为保守口径
  （论文注记义务，根 76 §三）。
- **配对双二进制同窗中位**：跨日 CSV geomean 含 ±1% 机器漂移（实测 v32real 4.36 vs
  其 CSV 3.907）→ 一切版本判定走配对 ×3/×5/×7 中位（v34 = v32 × 0.9909 由此定谳）。
- **逐位复现**：F2 8.026 ≈ 8.007、crankseg 50.030 ≈ 50.012（×5 与 ×7 轮）。

---

## 3. 头条结果（eval 摘要）

| 结果 | 数字 | 口径 |
|---|---|---|
| **对 Ocean 系统级首胜** | c-64 20.41 vs 20.80 = **0.981×**（官宣 0.979，双边同日新数据） | 相位和；×5 轮；×7 轮 21.20 挂星号待净窗 |
| 全刀弧线四阵全胜 | msc −14.3 / c-64 −18.0 / crankseg −15.7 / F2 −21.9 | ×7 配对（无刀 vs 全刀） |
| 套件级（dispatcher 全系统） | 1.54× geomean / 84 胜 /100 | pre.md 存档口径 |
| 接口层 | D2H：Ocean 1499ms vs 我方 ~80ms（c-64 同输出） | 端到端叙事 |
| 距跨线一步 | F2 1.045×（retry 0.58ms 刀）/ crankseg 1.832×（occgate 阵地） | 相位和 |

## 4. 新颖性防御矩阵（rebuttal 备用）

| 我方主张 | 最近前作 | 差异一句话 |
|---|---|---|
| 行级多维成本路由 | Misam（决策树选 dataflow）；Ocean（workflow+档位） | 前作矩阵级/档位号单钥；我方行级+硬件不变量平衡点+寄存器维 |
| 游标可续扫 | Ocean denseNumericIter（每窗重搜） | 机制为其内核所无；单调性→可续状态是我们的观察 |
| 寄存器占用进路由 | （未见 SpGEMM 前作；占用仅作 kernel 内调参） | 阶跃发现+占用恒等式+回溯验证；hedge: to our knowledge |
| 零表计数 | Ocean ESC（证明位置成立） | 机制血缘=我方 csort；dup≈2 门控与经济学自有 |
| 无偏 MinHash | Ocean 用 ER/采样/上界 | 调和→算术修复+失败级联根因档案 |
| 硬件侧同构链 | Flexagon→Trapezoid→Harmonia | 我方 = GPU 软件侧同构（分箱↔tile、kernel 族↔dataflow、retry↔动态调优），无需专用硬件 |

## 5. 论文叙事弧线建议

1. **动机**：累加占 72%；行异质性四数量级；错配代价 4×（行级）/26×（矩阵级）
2. **主张**：累加器选择 = 行级经济决策，答案可从硬件不变量推导（thesis）
3. **机制四件**：成本模型路由（C1）+ 融合双模式 kernel 使切换免费（C1/C2）+
   占用第三约束（C3）+ 零表短行（C4），上游用无偏估计+有界重试兜底（C5）
4. **评估**：保守协议（C6）下的系统级首胜与四阵全胜（§3）
5. **定位**：硬件侧层级调度（Harmonia 链）的 GPU 软件同构——无需专用硬件得到
   自适应数据流的好处
