# 创新点与论文定位（每条：机理 → 证据 → 与前作的差异）

> 铁律口径（idea-adoption-not-copy）：只吸收机理不 copy 形态；本文档区分
> 【原创机制】【借鉴后超越】【方法论创新】三类，防止论文叙事把借来的说成发明的。

## 一、原创机制（我方独有，未见于此前后作）

### 1. smap 游标（cursor）机制 —— docs/30 系列原创
- **机理**：宽行分多窗计算，换窗时记住每条 B 链上次搜到的位置，下一窗从停点续搜
  （"夹书签"）；ocean 每窗从头二分重搜。行越宽、窗越多，书签越值钱
- **证据**：vsp −56% / mult_dcop −48% / bloweya −47%（study/legacy-inno/30）；c-64 的 19,444 个
  diter 行 99.5% 走 cursor 模式
- **差异**：ocean denseNumericIter 无此机制；这是我们在 dense 侧唯一"机制性胜出"项
  （根 77 第二章 §3），部分对消其位图 8× 记号优势

### 2. cursor/dense 双模式同 kernel（row-level mode selection）
- **机理**：`hash_dense_direct_kernel` 一个内核按行选模式——rowmode=avgB≥阈值→cursor
  游标合并，否则 dense 位图窗口；共享同一 SMEM 窗布局（val8+flag1+pref4+cursor4/列）
- **证据**：mode 分层账（study/legacy-inno/82 §mode 分层）；c-64 输出 96.4% 由 diter 行产出
- **差异**：ocean 的 hash/dense/ESC 是分立 kernel 族+矩阵级 workflow 三选一；
  我们是行级模式内切换——与 per-row 分箱体系天然咬合

### 3. G-group 行内分工（争用分摊）
- **机理**：胖行交给一个 warp 后再分组、每组认领若干条 CSR 链——专治"多乘积挤同列"
  的同槽争用（重复只来自不同链，链内有序无重复）
- **差异**：根 77 第一章 §8："我们的 G-group 是独门"；ocean 行内分工只随档位变线程数

### 4. 探测经济学（1 指令哈希 + 2^k 表）
- **机理**：Knuth 黄金比例常数乘法散列（1 条乘法）+ 2 的幂表长（按位与取格，1 周期）
  vs ocean MurmurHash3（~10 条搅拌指令）+ 素数表（整数除法数十周期）
- **证据**：根 77 第一章 §2/§3 判定"我们胜"——表半满时撞车率本低，均匀性不值 10 条指令；
  乘以几亿次乘积 = 大钱

### 5. MinHash 无偏估计器（数学层）
- **机理**：调和平均（Jensen 高估 ~ln 倍）→ 算术无偏式
- **证据**：study/legacy-inno/03（本系列最大单项收益）；高估的下游灾难链见 study/legacy-inno/README 一页纸
- **差异**：估计器数学是我们的修复，ocean 用 ER/采样/上界另一套

### 6. ESC 零表短行内核（counting 前移）
- **机理**：flop≤32 的行不要表——每线程一乘积，位次=popc(比我小的 key)，leader 求和直写；
  零表零清零，输出天然有序（compact 的 csort 税同时消失）
- **证据**：study/legacy-inno/22 设计+三连修，门演化 v4（dup≤2 = counting/表成本平衡点，germany vs
  patents 教科书实证）
- **差异（血缘声明）**：机制是我们自己的 csort 位次计数，ocean 只证明了它放在 accumulate
  位置成立（根 76 §五）

### 7. 多维路由判据体系（DISPATCH_MODEL）
- **机理**：20 桶选形 + 字节模型（SROUTE）/占用门/dup 门/a_len 成本律（阈值 32=WARP_SIZE
  硬件常数），矩阵统计→结构化模型（disp_train/disp_synth 训练）
- **差异**：ocean 只比档位号（`dense_bin_id ≤ bin_id`）；nsparse/spECK 是单法 per-matrix
  选择——我们是 per-row 分箱+多维成本模型
- **防线**：Misam（决策树选 dataflow）是最近前作**必引**；定位=硬件侧，我方 GPU 软件侧

### 8. 行级 retry 纠偏机制
- **机理**：est 欠估行 flop 定表重跑（vs 全局重炮）；配合 est>span 高估怪物的档内化
- **升级候选**：Harmonia Gain>α·Cost 成本门+滞回（T=2-4）+最坏有界回退模板（study/23 §6）

### §一补：09-17~18 已收割机理（study/22 创新点总账，此前本文件遗漏——最新层）

1. **连续估计驱动表配（v38 est 连续化）**：est 从 pow2 档离散值改连续插值 → 行行 LF 均匀 ≈0.37 → 短探测链——**这是 kernel 级三阵全胜（F2 0.356/c-64 0.906/crankseg 0.581）的根基**，与彼素数表 LF 0.57 长 2-3 跳链形成停顿-指令交换的活例（我停顿 21% vs 彼 36%，warp 并行藏延迟）
2. **交织 B 布局（④，db21731）**：col/val 两窄流→16B 单股（uint4 装载），访存请求 −18%，系统 geomean −1.9%；B16 分裂人格（kernel 级 +16% vs 系统 −1.6%）= 口径学素材
3. **read-first 探测机理自有化（转正刀）**：对拍归因→自有形态，引 nsparse 公共先例
4. **合并 atomicAdd 四代经济学（negative result）**：H100 SMEM RED 本廉 + dup 时间性分散 = 负期望杠杆（四代全值级正确、全性能负 6-15%）；nsparse/spECK 无先例 = 首个系统测明的设计空间分析——**证伪也是贡献**



| 借的机理 | 我方形态（超越点） | 赢面 |
|---|---|---|
| ocean largest 位图+全局值流 | LARGEST 内核（v30 默认） | c-big **−63%** |
| ocean 七档联动（墙×块×驻卡） | 档位梯 6 档 **per-tier 发射** + **档位序稳定排序保 CSR 局部性** + count/数值**双侧分流**（路由不驱逐） | in-2004 −30.1（配对）；wb-edu −48.5 |
| ocean 零表 counting-sort | ESC（前移至 accumulate）+ `csort_counting` 家族（REG40=6 块 75% vs radix 229=1 块 12.5%，指令+占用双优） | crankseg −3.3 / F2 −5.7 / CSORT F2 −9.6 |

超越点的共性：**借来的每一件都被我们的行分布打过回票，最终以我方路由体系重组后才赢**
（根 76 §18 终局判断：ocean dense 优势是三者咬合的整体，单件移植必败——per-tier 发射+
稳定排序+双钥分流就是这个"整体"在我方栈上的重生物）。

## 三、方法论创新（可写进论文的测量/工程贡献）

1. **寄存器占用审计刀系 + 占用恒等式**：cuobjdump -res-usage 一行审计 → launch_bounds
   单行刀 → 电池守门；2×占用÷1.47 溢出税=1.34× 净吞吐（预测 13.6 vs 实测 13.8 ✓）。
   发现"寄存器是 SMEM/线程数之外的第三占用约束"+ 指令瘦 14% 仍慢 31% 的指令经济性定律
2. **相位和口径 + 配对双二进制电池**：Σ相位（不含 host 间隙）vs 单括号的口径审计；
   跨日 CSV geomean 不可比（±1% 漂移实证）→ 一切版本判定走配对同窗 ×3/×5/×7 中位
3. **物理台账驱动的优化循环**：分相实测→nsys/ncu 双画像→DUMP_ROUTING 人口普查→
   假设定谳（含证伪存档：单维门四连败、SEGRTS wash、bin7 门三轮）——负结果也是档案

## 四、论文定位（related work 骨架）

- **硬件侧同题**：Flexagon(ASPLOS'23，多 dataflow 可重构) → Trapezoid(HPCA'24) →
  Harmonia(ISCA'26，三层层级调度) —— "固定 dataflow 只适合特定 pattern"是 ISCA 级论证
- **我方 = GPU 软件侧同构**：分箱↔tile、kernel 家族↔dataflow、retry/occgate↔动态调优层；
  无需专用硬件。"同架构只换调度器"的对照方法论借鉴自 Harmonia 评估设计
- **必引**：Misam（决策树选 dataflow，DISPATCH_MODEL 最近前作）；nsparse/spECK（单法选择）
- **双生态位叙事**：ocean337（hash 主场）+ first100（merge3 主场 0.614×/59 阵）=
  dispatcher 的存在意义实证（关联线，见根 78）
