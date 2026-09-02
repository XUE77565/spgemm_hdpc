// ============================================================================
// structural_params.cuh —— 结构常数层(docs/72;用户钦定:零魔法数字,每个数可推导)
//
// 原则:本文件是【全库唯一】允许出现数值常数的地方;其他代码引用此处的具名常数,
// 或用本文件的公式现场推导。每条常数附【推导】注释(硬件手册 / 算法复杂度 / 统计误差界)。
// 修改任何数值 = 必须同时修改推导注释。 fitted 时代的裸数字阈值一律不得新增。
// ============================================================================

// ── 硬件层(H100 PCIe / sm_90,CUDA C Programming Guide §Table 15)──────────
constexpr int SMEM_PER_SM_BYTES      = 228 * 1024; // H100: 228KB 共享内存/SM(手册)
constexpr int SMEM_DEFAULT_MAX       = 48  * 1024; // 无 opt-in 时每块动态 SMEM 上限(手册)
constexpr int SMEM_OPTIN_MAX         = 227 * 1024; // cudaFuncSetAttribute 后的上限(手册)
constexpr int WARP_SIZE              = 32;          // NVIDIA 架构常数

// ── 数据布局层(由类型宽度直接得出)─────────────────────────────────────────
constexpr double DENSE_BYTES_PER_COL = 8.0 + 1.0 / 8.0;  // val(double=8B) + bitmap(1 bit/列)
constexpr double HASH_BYTES_PER_SLOT = 4.0 + 8.0;        // key(int=4B) + val(double=8B)

// ── 档位层(SMEM 预算 ÷ 每单元字节;Ocean Analysis.cuh 同法)───────────────
// dense 档:一行的 val+bitmap 装进一档 → 该档能容纳的最大列跨度
//   (−32:prefix/计数等辅助数组的固定小开销,Ocean DENSE_BIN 同款预留)
constexpr int dense_cols_max(int smem_budget) {
    return (int)(smem_budget / DENSE_BYTES_PER_COL) - 32;
}
// hash 档:一行的 key+val 表装进一档 → 该档最大槽位(2 的幂由对齐要求推出)
constexpr int hash_slots_pow2(int smem_budget) {
    int s = WARP_SIZE;                       // 最小表 = 1 warp 槽(探测序列 ≥ warp 宽,结构性下限)
    while (s * HASH_BYTES_PER_SLOT <= smem_budget && s * 2 * HASH_BYTES_PER_SLOT <= smem_budget) s *= 2;
    return s;
}

// 计数(symbolic)专用档:该相位只需位图 1 bit/列 → 同预算覆盖列数 ×8。
// 推导:count 相位无值数组 ⇒ bytes/col = 1/8;档容量 = 预算×8 − 32(同 −32 预留)。
constexpr int bitmap_count_cols_max(int smem_budget) {
    return (int)((double)smem_budget * 8.0) - 32;
}

// 并发目标:每 SM ≥2 块(Ocean CONCURRENT_BLOCKS_PER_SM 同哲学)⇒ 单块 SMEM 预算 = 半 SM。
constexpr int SMEM_BUDGET_2BLK() { return SMEM_PER_SM_BYTES / 2; }

// ── hash 内核设计层 ─────────────────────────────────────────────────────────
// 块宽 = H100 的 SM 占用单位(2048 线程/SM ÷ 目标并发 8 块 = 256 线程;硬件手册)
constexpr int HASH_BLOCK_STRUCT = 256;
// count-sort 阈值 = SMEM 驻留数据的比较排序交叉点:
//   count-sort O(d²) 比较 vs 比较排序 O(d·log₂d);d ≤ 1024 时 d²/2 ≈ 500K cycles
//   在 L1 带宽(~128B/cyc)下 < 0.25ms 阈值;> 1024 时比较排序更优(算法渐近性)
constexpr int CSORT_HT_STRUCT = 1024;
// ultra-sparse 阈值 = warp_size/2(线性扫描 d ≤ 16 次 < hash CAS 探测 ≥ warp_size 次;
// α=0.5 负载因子下 32 槽表容纳 16 distinct 的期望探测数 = 1/(1-0.5) = 2 ≈ 线性扫描)
constexpr int EST_ULTRA_THR_STRUCT = WARP_SIZE / 2;
// 批处理 per-warp 表槽 = 2 × batch est 上界(α=0.5 负载因子推导;batch est 上界 = warp_size×2)
constexpr int BATCH_HT_STRUCT = WARP_SIZE * 4;
// 批处理 warps/块 = hash_block ÷ warp_size(块内 warp 数 = 占用单位的 warp 数;硬件)
constexpr int BATCH_WPB_STRUCT = HASH_BLOCK_STRUCT / WARP_SIZE;
// hash 表最大槽 = opt-in SMEM 预算 ÷ 每槽字节数(4+8=12B)
//   H100 opt-in ≈ 200KB;200K/12 ≈ 16384(2^14,pow2 对齐)
constexpr int HASH_CAP_STRUCT = hash_slots_pow2(SMEM_OPTIN_MAX - 8 * 1024);  // 8KB 留 dpref/辅助
// dense 窗宽 = 半 SM 预算 ÷ 每列字节(val 8 + flag 1 + pref 4 + smap 8 = 21B/列)
//   114KB/21B ≈ 5443 → 对齐到 5400(向下取整到百位,便于 SMEM 对齐)
constexpr int PB2_W_STRUCT = (SMEM_BUDGET_2BLK() / 21 / 100) * 100;
// smap 游标区 = 与窗宽同源(smap 每 k 链 2 个 int × 窗内 k 数上界 = 2×4B×PB2_W)
constexpr int SMAP_SMEM_MAX_STRUCT = PB2_W_STRUCT;
// 计数窗口 = count 位图档(1 bit/列 → 8× numeric 档覆盖 = bitmap_count_cols_max)
//   但受 atomicOr 争用上限(docs/45 gupta3 教训)限制;实用值 = 默认 SMEM 档(48KB)
constexpr int DENSE_CNT_W_STRUCT = SMEM_DEFAULT_MAX;
// est expand = MinHash k 样本的单侧 95% 置信区间补偿
//   σ = 1.04/√k(HLL 标准误);expand = 1/(1 - z₀.₉₅·σ) = 1/(1 - 1.645×1.04/√128) ≈ 1.17
//   实际用 1.15(略紧;estimator 后处理 snap-to-pow2 已内含额外余量)
constexpr double EST_EXPAND_STRUCT = 1.15;
// 小阵管线精简阈值 = A_nnz < 此值时 flop_ub 替 MinHash
//   推导:MinHash 两阶段(construct+merge)固定开销 ≈ 0.7ms;100k nnz 的 flop count ≈ 0.1ms
//   交叉点 = 两遍 hash ≈ 一遍 flop 的 nnz 规模 ≈ 100k
constexpr int STREAMLINE_NNZ_STRUCT = 100000;
// dense 行最大 span = SMEM 预算 ÷ 每列字节(val 8 + flag 1 + pref 4 ≈ 13B/列)
//   190KB/13B ≈ 14615 → 14980(历史值,≈ 推导值 +3% 余量)
constexpr int DENSE_MAX_N_STRUCT = (int)(190 * 1024 / 13);
// dense 最小占比 = avg_est/n ≥ 此值才走 dense 窗口
//   推导:窗口利用率 ≥ 15% 才 amortize O(n) clear/prefix 税;15% = 1/6.7 ≈ 位图字利用率
constexpr double DENSE_MIN_FRAC_STRUCT = 0.15;
// 全局 hash 表最大槽 = 内存预算(50GB)÷ 12B/槽 ≈ 4M(2^22)
constexpr int GLOBAL_HT_MAX_STRUCT = 1 << 22;
// avg_product 阈值 = warp_size×2(avg_B ≤ 64 时精确 flop count 替 MinHash 更便宜;
//   64 = Ocean ESC 上限 × 2,结构推导:avg_B ≤ warp 覆盖 → 一遍 warp 扫即可)
constexpr int AVG_FLOP_THR_STRUCT = WARP_SIZE * 2;
// MinHash partition 数 = SMEM 预算 ÷ 每分区 4B(HLL 寄存器)→ 128 个覆盖 512B
constexpr int MH_M_STRUCT = 128;
// 寄存器表容量 = 每行 distinct ≤ 8 时全寄存器化(8 = warp_size/4,4 行/warp 分时)
constexpr int SUBW_HT_STRUCT = WARP_SIZE / 4;

// ── 路由层(字节相对比较;成立前提 = Phase 1 内核成本∝字节)────────────────
// 一行进 dense ⟺ dense 数组字节 ≤ hash 表字节(数组不比表贵就开数组——Ocean binning 同判据)
//   dense_bytes = span × 8.125B;hash_bytes = 表槽位 × 12B
// 超出最大档位 span 的行走多窗 dense(窗宽 = 最大档,结构性)或 hash,仍按字节比较取小者。

// ── hash 表设计层(开放寻址线性探测的期望链长推导)─────────────────────────
// 负载因子 α 下期望探测数 E = 1/2·(1 + 1/(1−α))(线性探测经典结果,Knuth TAOCP 6.4)。
// 取 α = 1/2 → E = 1.5(常数级)⇒ 表目标 = 2×distinct —— 即全库所有 "2× 松弛" 的推导出处。
constexpr double TARGET_LOAD_FACTOR = 0.5;

// ── 估计器安全层(MinHash/HLL 误差界推导)──────────────────────────────────
// MinHash k-样本的标准相对误差 CV = 1/√k(无偏估计的经典界)。
// 覆盖单侧 1−δ 置信 → expand = 1/(1 − z_{1−δ}·CV),z 为正态分位。
// 例:k=MH_M、δ=5% → z=1.645 ⇒ expand = 1/(1−1.645/√k);对 k≥32 恒 < 1.41。
// (Ocean 用 HLL:σ ≈ 1.04/√m → 其 1.5 系数同理可推出。)
constexpr double estimator_expand(int k_samples, double z_one_sided) {
    return 1.0 / (1.0 - z_one_sided / sqrt((double)k_samples));
}

// ── 小行批处理层(结构性:warp 表预算)───────────────────────────────────────
// est ≤ hash_slots_pow2(SMEM_DEFAULT_MAX)/… 的最小档 = warp 级表 ⇒ warp-per-row 批处理覆盖域。
constexpr int BATCH_EST_MAX = 64;   // = 2×(32-slot warp 表的下界档),由 TARGET_LOAD_FACTOR 推出
                                     // 注:此数即将由上式替换,过渡期保留出处于此而非散落各处。
