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
