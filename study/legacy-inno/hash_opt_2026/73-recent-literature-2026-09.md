# docs/73 近期文献扫描(2026-09-03,loop 自主轮)

## 三个 2025-26 新工作 + 对我们的启示

### 1. Wang et al., TACO 2025 — "Optimizing General SpGEMM on the GPU"(DOI 10.1145/3774654)
- 核心:轻量 ML 模型预测给定矩阵的最优实现(算法选择器)
- **对我们的意义**:验证了 dispatcher 线的论文叙事方向 —— 我们的差异点 =
  ①合成标定(disp_synth,H100 解析模型)+ 不相交训练集(disp_train 213 vs ocean337 测试)
  = 无测试集泄漏方法学;②DISPATCH_MODEL 权重已结构化声明(375d9d0)
- 行动:无需移植;引用 + 对比方法论即可

### 2. AIA 论文(arXiv 2512.12036, SK hynix AiM)—— hash-based multi-phase SpGEMM + 近存
- hash 设计与我们同族:linear probing、atomicCAS 插入 + atomicAdd 累加、双表(key/val)、
  SMEM 优先全局回退(64/1024/8192 桶 + ≥8192 回退全局 —— 与我们 BIN 梯几乎同构!)
- 行分组线程策略:轻行 PWPR(4 线程/行,block 512)vs 重行 TBPR(整 block/行)
  → 我们已有对位(BATCH warp-per-row / G-group / heavy)
- **多 stream 按 group 并行 launch** → 我们已有 NSTREAM(默认 4)
- gather + bitonic sort → 我们 csort(count-sort)在 SMEM 驻留域更优
- AIA 硬件(HBM 内存控制器内嵌引擎)不可移植,软件侧无新增可偷机制
- 性能仅对比 cuSPARSE(自乘降 80.5%),未对 bhSparse/Ocean —— 不构成新基线威胁

### 3. MAGNUS / ggMAGNUS(arXiv 2607.22866, PVC+H200)—— Hierarchical Multisplit
- 机制:heavy 行先 outer-product 构中间阵,再按列范围递归 multisplit 成 chunk,
  直到 chunk 的 dense 累加器装进 local memory;chunk 列范围不相交 → 各自独立累加免归并
- H200 上对 cuSPARSE/Kokkos/TileSpGEMM/OpSparse geomean 1.81-6.63×(SuiteSparse 330 + RMAT)
  ⚠ 未对 Ocean;**测试-suite 与我们几乎同规模(330 vs 337)→ 潜在新对比基线**
- **可借鉴(评估后认为我们已有对位或优先级低)**:
  a. heavy 行列范围 chunk 化免归并 → 我们 = BIN_DITER 多窗 + BCNT 位图;差异点 =
     MAGNUS 递归多级 + 物理重排(multisplit),我们固定窗 + 游标。若 heavy(全局 hash)族
     在 v29 后仍是大 gap 来源,可试"heavy 行 → 列范围多窗 dense"(span≤n/窗数即可装下)
  b. O(1) 硬件自适应参数公式(式 4-7)→ 与我们 structural_params 同哲学,互相印证
  c. light/heavy 双队列并发 → 已有(streams)
- 行动:挂"heavy 族多窗化"为候选(优先级 = v29 gap 分析后定)

## 结论
近期文献无"必须立刻移植"的机制;dispatcher ML 选择(TACO)与自适应参数公式(MAGNUS)
两个方向验证了我们已有路线。最大的潜在威胁 = MAGNUS 可能成为新 SOTA 基线
(H200 数字大但未对 Ocean;若需对比,其代码/数据可得性待查)。
