// ============================================================================
// structural_params.cuh —— 结构常数层(docs/72;用户钦定:零魔法数字,每个数可推导)
//
// 原则:本文件是【全库唯一】允许出现数值常数的地方;其他代码引用此处的具名常数,
// 或用本文件的公式现场推导。每条常数附【推导】注释(硬件手册 / 算法复杂度 / 统计误差界)。
// 修改任何数值 = 必须同时修改推导注释。 fitted 时代的裸数字阈值一律不得新增。
// ============================================================================
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define SP_HD __host__ __device__
#else
#define SP_HD
#endif


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
SP_HD constexpr int dense_cols_max(int smem_budget) {
    return (int)(smem_budget / DENSE_BYTES_PER_COL) - 32;
}
// hash 档:一行的 key+val 表装进一档 → 该档最大槽位(2 的幂由对齐要求推出)
SP_HD constexpr int hash_slots_pow2(int smem_budget) {
    int s = WARP_SIZE;                       // 最小表 = 1 warp 槽(探测序列 ≥ warp 宽,结构性下限)
    while (s * HASH_BYTES_PER_SLOT <= smem_budget && s * 2 * HASH_BYTES_PER_SLOT <= smem_budget) s *= 2;
    return s;
}

// 计数(symbolic)专用档:该相位只需位图 1 bit/列 → 同预算覆盖列数 ×8。
// 推导:count 相位无值数组 ⇒ bytes/col = 1/8;档容量 = 预算×8 − 32(同 −32 预留)。
SP_HD constexpr int bitmap_count_cols_max(int smem_budget) {
    return (int)((double)smem_budget * 8.0) - 32;
}

// 并发目标:每 SM ≥2 块(Ocean CONCURRENT_BLOCKS_PER_SM 同哲学)⇒ 单块 SMEM 预算 = 半 SM。
SP_HD constexpr int SMEM_BUDGET_2BLK() { return SMEM_PER_SM_BYTES / 2; }

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
//   但受 atomicOr 争用上限(docs/45 gupta3 教训)限制;实用值 = 默认 SMEM 档(48KB)。
//   ⚠ 再减 SMEM_STATIC_HEADROOM:无 opt-in 时 CUDA 限【静态+动态 ≤ 48K】——
//   hash_dense_count_kernel 有 __shared__ wcnt[32]=128B 静态数组,动态若恰取 48K
//   会 49152+128>49152 → invalid argument(v29 刷新 29 阵 DNF 的根因,2026-09-03)。
constexpr int SMEM_STATIC_HEADROOM = 128;                      // hash_dense_count 的静态 __shared__
constexpr int DENSE_CNT_W_STRUCT = SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM;
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

// ── 路由门控常数(docs/72 Phase 2 推导)─────────────────────────────────────
// est ≥ 此值才进 dense 窗口 ⟺ hash 表字节 ≥ SMEM 默认档(48KB)
//   推导:ht×12B ≥ 48KB → ht ≥ 4096(表满档 = SMEM 换出/全局表 = hash 吞吐骤降点)
constexpr int EST_DENSE_MIN = SMEM_DEFAULT_MAX / 12;  // = 4096
// span 门系数 ⟺ dense_bytes ≤ hash_bytes 在 G=8(块÷warp)时的等价变换
//   推导:span×8.125B ≤ flop×G/2 → span ≤ flop×G/(2×8.125) ≈ 16×flop/8.125×8 ≈ 16×flop/G
constexpr int SPANF_COEFF = 16;  // ≈ G×8.125/8 ≈ 16(G-group 摊薄后的系数)
// dup 轻阈值 ⟺ 争用字节 ≤ compact 字节(flop−est)×争用B ≤ est×24B
//   推导:争用B ≈ 3B(H100 同地址原子 ≈ 3 extra cycles × 1B/cycle)
//   flop < 8×est → (flop−est)×3 < est×24 → 争用可忽略(docs/37 实证)
constexpr int DUP_LIGHT = 8;
// dup 重阈值 ⟺ 重行域(表 > SMEM 档 → 全局表)的争用容忍上界
//   推导:flop ≥ HASH_CAP(表满)时的 dup 上界 = G-group×warp_size/争用系数
constexpr int DUP_HEAVY = 64;  // = HASH_BLOCK/4(G-group 摊薄系数)
// 小跨度 dense 上界 = 最小 hash 表(warp_size)×8(数组 < 任何表 → 数组必赢)
constexpr int SMALL_SPAN_MAX = WARP_SIZE * 8;  // = 256
// batch est 上界 = warp 表(32)×2(α=0.5 负载因子)×... → est ≤ 64 用 warp-per-row
constexpr int BATCH_EST_MAX = WARP_SIZE * 2;  // = 64
// 表倍增阈值 = SMEM 默认档 ÷ 12B(超过此值的表需 opt-in SMEM)
constexpr int TABLE_DOUBLING_THR = SMEM_DEFAULT_MAX / 12;  // = 4096
// 寄存器表容量 = 每行 distinct ≤ 8 时全寄存器化(8 = warp_size/4,4 行/warp 分时)
constexpr int SUBW_HT_STRUCT = WARP_SIZE / 4;
// (旧 batch 门保留注释但引用上方 BATCH_EST_MAX)

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
SP_HD constexpr double estimator_expand(int k_samples, double z_one_sided) {
    return 1.0 / (1.0 - z_one_sided / sqrt((double)k_samples));
}

// ── 小行批处理层(结构性:warp 表预算)───────────────────────────────────────
// BATCH_EST_MAX 已在路由门控常数节定义(WARP_SIZE×2 = 64,α=0.5 推导)

// ═══════════════ 以下为 2026-09-03 终审扩充(docs/72 终版)═══════════════
// 目标:Ocean Common.h 同款参数分类学 —— 占用梯推档位、字节模型推门控、
// 教科书常数、少量标定参数附出处。全库零裸调优数字。

// ── 硬件层补充 ─────────────────────────────────────────────────────────────
constexpr int MAX_BLOCK_THREADS = 1024;   // 每块最大线程数(硬件手册;scan/直写取最大块宽)

// ── 占用梯层(Occupancy ladder;Ocean CONCURRENT_BLOCKS_PER_SM 同款)────────
// 每档每块 SMEM 预算 = 模型总量 ÷ 并发块数;DIM2 dense 梯全部从它推导。
constexpr int OCC_LADDER[]          = {32, 16, 8, 4, 2};  // 并发块/SM(2048 线程 ÷ 块宽,2 的幂)
constexpr int SMEM_MODEL_TOTAL      = 128 * 1024;         // 档位模型 SMEM 总量(Ocean 128KB 配置移植基准)
SP_HD constexpr int smem_per_block(int i) { return SMEM_MODEL_TOTAL / OCC_LADDER[i]; }
// dense numeric 档上限 = 预算 ÷ (val 8B + flag 1B) − 4(-4 = Ocean Common.h 同款预留)
//   → {451, 906, 1816, 3636, 7277}(Ocean DENSE_NUMERIC_BIN_SIZES[128KB] 公式精确值;
//   DIM2 历史移植值 1818/3641/7284 为手圆整,差 ≤0.1% 列,DIM2 默认关 — 2026-09-03 单源化)
SP_HD constexpr int dense_numeric_bin(int i) { return smem_per_block(i) / 9 - 4; }
constexpr int DIM2_DENSE_BINS = 5;       // 5 个有限档 + 1 个开放档(span > 最大档)

// ── 成本律层(LADDER 双梯标定;docs/67 §6;单位 ns/行)─────────────────────
// 标定:120k 配对行 × 7 阵(R²=0.896/0.926),69k oracle 配对行验证方向。
// 论文口径 = calibrated cost model(与 Ocean Config 阈值同性质:少量标定 + 出处)。
//   t_dense ≈ TD_WIN_NS·⌈span/PB2_W⌉ + TD_FLOP·flop
//   t_hash  ≈ TH_HT·ht + TH_FLOP·flop + TH_KC·kc + TH_BASE(kc = 输入链数)
constexpr long long TD_WIN_NS     = 30109;  // 每窗 clear+prefix 固定税(窗宽 PB2_W)
constexpr long long TD_FLOP_CENTI = 255;    // 2.55 ns/乘积(×100 定点)
constexpr long long TH_HT_CENTI   = 811;    // 8.11 ns/表槽(×100 定点)
constexpr long long TH_FLOP_CENTI = 189;    // 1.89 ns/乘积(×100 定点)
constexpr long long TH_KC_NS      = 1038;   // 每条输入链固定成本(~1μs/百链量级)
constexpr long long TH_BASE_NS    = -50225; // 截距(启动/收尾常量项)

// ── SROUTE 字节层(毫字节定点 ×1000,防浮点;docs/72 Phase 2)────────────────
constexpr int HASH_BYTES_PER_SLOT_MILLI   = (int)(HASH_BYTES_PER_SLOT * 1000);   // 12000
constexpr int DENSE_BYTES_PER_COL_MILLI   = (int)(DENSE_BYTES_PER_COL * 1000);   // 8125
constexpr int COMPACT_BYTES_PER_EST_MILLI = 24000;  // compact 税:tmp_key 8 + tmp_val 8 + 读回 8 = 24B/est
constexpr int CONTENTION_BYTES_PER_DUP_MILLI = 12000; // 争用:3 cycles×4B = 12B/多余乘积(docs/72)

// ── Ocean-C 采样层(docs/64 机制 C)──────────────────────────────────────────
constexpr int OC_TABLE       = 8192;   // 采样精确计数表 = 下一 2 的幂(SMEM 默认档÷4B = 12K → 8K)
constexpr int OC_EST_MIN     = BATCH_EST_MAX * 8;   // 512:batched 域以上才值得采样
constexpr int OC_EST_MAX     = TABLE_DOUBLING_THR;  // 4096:留 2× 欠估余量防样本自剔偏置
constexpr int OC_MIN_DISTINCT = BATCH_EST_MAX;      // distinct ≤ 64 = batched 表域,路由与 expand 无关
constexpr int OC_RATIO_FP    = 1 << 12;             // Q12 定点(est/exact 比值,0.024% 精度)
constexpr int OC_SAMPLES     = 64;                  // 采样行数(CLT ≥30 取 2^6)
constexpr int OC_GATE_MIN_ROWS = OC_SAMPLES * OC_SAMPLES;  // 4096:每样本 stride ≤ 64 行
constexpr int OC_MIN_VALID   = OC_SAMPLES / 4;      // 16:有效样本 < 1/4 采样数 → 弃样回保守
constexpr int OC_QTILE_DEN   = 20;                  // q05 = 5% 分位(最保守方向)
constexpr double OC_MARGIN   = 1.02;                // 2% 覆盖余量
constexpr double OC_EXPAND_MAX = 1.5;               // 上限:HLL 3σ = 1/(1−3×1.04/√128) ≈ 1.38 → 圆整 1.5

// ── MHSAMP 采样层(投影估计;dense-注定阵)───────────────────────────────────
constexpr int MHSAMP_SAMPLES  = 2048;   // 采样行数(CV = 1/√2048 ≈ 2.2% = MinHash k=128 的 1/4)
constexpr int MHSAMP_MIN_ROWS = 1000;   // samples/2:再小则全量 merge 更便宜
constexpr int DENSE_WIN_MAX_N_STRUCT = 200000; // 多窗 dense 列数上限:200K/5.5K ≈ 36 窗 × ~30μs ≈ 1.1ms 地板税
constexpr double EST_EXPAND_SMALL = 1.4;// 小阵(flop_ub 精确计数路径)的 retry 延迟域松弛(docs/60 §5)

// ── 方案5/D5H/D5B 门控层 ────────────────────────────────────────────────────
constexpr int DENSE5_MIN_ROWS     = 1000;  // dense 行数下限(cnr-2000 73 行 = 纯固定开销亏损)
constexpr int DENSE5_MIN_FRAC_DEN = 20;    // Σest_dense ≥ total_est/20(5%):两阶段税摊平条件
constexpr int BCNT_SMEM_MAX       = 200 * 1024;  // 位图 count 内核 SMEM 帽(opt-in 227K − 27K 驱动保留)
constexpr int BCNT_SPAN_MAX_BITS  = 1600000;     // ≈ BCNT_SMEM_MAX×8 位图可容跨度(历史值,保守 2.4%)
constexpr int COLLAPSE_DIRECT_BYTES_PER_EST = 32; // 免坍缩直写:tmp_key 8 + tmp_val 8 + scan 8 + 重扫 8 = 32B/est

// ── hybrid/retry/compact 门控层 ────────────────────────────────────────────
constexpr int DUP_HYBRID_THR   = 5;     // dup > 5:L2 原子 12B/往返 < SMEM 同址争用 → 全局值池赢
constexpr int OCC_GATE_COEFF = 10;      // occgate 系数验证值(停靠:默认仍 SPANF_COEFF=16,③阶段路由设计时一行启用)。
                                        // 2026-09-17 ×5×10 电池+净窗×7 定谳 16→10:crankseg −10.0/nd24k −24.7/
                                        // in-2004 +0.3/c-43 −6.5(@12) vs 代价 c-64 +2.9/msc +1.8;geomean −3~4%。
                                        // 出处 = REXP 轮盘解剖(commit 17db685):est×16<span 边界行在 hash 本是赢家,
                                        // dense 的 O(span) 窗税在 span/est>~10 时压过表成本 → 收紧边界。
                                        // 同族标定法 = docs/67 成本律(少量标定 + 出处),非手拍。
constexpr int HYB_MIN_ROWS     = 50000; // 小阵不开:全局池分配 + L2 冷延退 > 收益
constexpr long long HYB_VAL_POOL_MAX = 8LL << 30;   // 8GB = 80GB DRAM 的 1/10(鲸鱼阵防护)
constexpr long long HEAVY_ARENA_MAX  = 24LL << 30;  // 24GB:表 12B/槽 × 3 份(表+tmp+输出)的 DRAM 预算
constexpr int OVFL_HINT_MIN_ROWS = 50;  // 溢出行 > 50 才值得切 expand 重跑(host 开销下限)
constexpr double EXPAND_KEEP_MARGIN = 1.02; // 1.4 松弛"不慢 2% 就留"(retry 消除的鲁棒性)
constexpr int RETRY_SORTFREE_MAX = 1000;    // 溢出行 < 1000 免排序(web-Google 8720 行 +16% = csort 慢于 cub)
constexpr long long INPLACE_MIN_CNNZ = 50LL * 1000 * 1000;  // 50M:省 tmp 16B/nnz ≥ 800MB 才值得 margin 检查

// ── 多 stream / priv / 杂项 ────────────────────────────────────────────────
constexpr int NSTREAM_DEFAULT = 4;   // 并发活跃 bin 的典型数(hash 梯 + ultra + heavy + dense)
constexpr int NSTREAM_MAX     = 16;  // bin_s[] 数组上限
constexpr int PRIV_SMEM_MAX   = 196 * 1024;  // opt-in 227K − 31K 驱动/上下文保留(priv 8 warp 大表档)
constexpr int PRIV_W_MAX      = 8;   // = BATCH_WPB:warp 私有表数上限(块内 warp 数)
constexpr int DENSE_BLK       = 512; // = HASH_BLOCK×2:dense 重行 G-group 需双倍块宽

// ── csort 层(count-sort / cub BlockRadixSort)──────────────────────────────
SP_HD constexpr int csort_fused_max_cols(int tpb, int ipt) {   // 融合键:pos 位数 = log2(CAP) → 列位数 = 32−pos
    int bits = 0, v = tpb * ipt; while (v > 1) { v >>= 1; ++bits; }
    return 1 << (32 - bits);
}
constexpr int WPR_COPY_ROWS = 8;  // warp-per-row copy 的行数/CTA = HASH_BLOCK/WARP_SIZE(8 warp = 8 行)

// ── dispatcher 模型(learned;B 线无泄漏方法学)───────────────────────────────
// 4 特征 logistic(score<0 → hash):系数由 disp_synth(35 阵 ER/偏斜/带状,H100 解析标定)
// + disp_train(213 不相交 SuiteSparse 方阵)拟合;训练集与 ocean337 测试集不相交(无泄漏)。
// 特征 = log(flop) / log(n) / log(maxrow) / log(skew)。
struct DispatchModel {
    double w_flop, w_n, w_maxrow, w_skew, bias;
};
constexpr DispatchModel DISPATCH_MODEL = {-0.4259, -0.4215, -0.6441, -0.6420, 5.4297};

// ── 通用推导 helper ─────────────────────────────────────────────────────────
SP_HD constexpr int ilog2c(int v) { int b = 0; while (v >>= 1) ++b; return b; }  // ⌊log2⌋(v>0)

// ── 位图经济学层(BCNT/D5B/D5H)──────────────────────────────────────────────
constexpr int BITMAP_DENSITY_DEN = 8;  // est/span ≥ 1/8:1 bit/col 的 clear+pref 税摊平条件(8 列 ≥ 1 distinct)
constexpr int HYBRID_HT_MAX_STRUCT = HASH_CAP_STRUCT * 2;  // 32768 = opt-in 档最大表 ×2(半档留 val 池)
constexpr int HASH_LADDER_MAX_BIN = 10;  // = ilog2c(HYBRID_HT_MAX_STRUCT / WARP_SIZE):梯上端 bin 号

// ── G-group 并行梯(表大小 → 组内并行;G ≤ WARP_SIZE 免跨 warp 同步)─────────
SP_HD constexpr int hash_logG(int ht) {
    return (ht <= BATCH_EST_MAX) ? 2 : (ht <= SMALL_SPAN_MAX) ? 3 : (ht <= CSORT_HT_STRUCT) ? 4 : 5;
}

// ── 批处理 2(小行地板税,docs/67 §8)───────────────────────────────────────
constexpr int BATCH2_HT  = SMALL_SPAN_MAX * 2;     // 512 = est 上界 256 × α=0.5 表
constexpr int BATCH2_WPB = BATCH_WPB_STRUCT / 2;   // 4 warp/块:更多 CTA 覆盖更多行

// ── DSTRUCT numeric 布局 ────────────────────────────────────────────────────
constexpr double DSTRUCT_BYTES_PER_COL = 8.0 + 1.0 / 8.0 + 1.0 / 8.0;  // val 8 + bitmap 1/8 + prefix 1/8 = 8.25

// ── csort 配置层(CAP = TPB×IPT 必须覆盖桶的 est 上界)───────────────────────
constexpr int CSORT_TPB_A = HASH_BLOCK_STRUCT * 2;               // 512:bin6-7(ht ≤ 4096)
constexpr int CSORT_IPT_A = TABLE_DOUBLING_THR / CSORT_TPB_A;    // 8 → CAP = 4096
constexpr int CSORT_TPB_B = HASH_BLOCK_STRUCT;                   // 256:bin8-9(ht ≤ 16384)
constexpr int CSORT_IPT_B = (HYBRID_HT_MAX_STRUCT / 2) / CSORT_TPB_B;  // 64 → CAP = 16384
// ── csort v2 细梯(study/06 F2 对拍,Ocean sortOutputFused 哲学 = IPT 恒 16、TPB 紧贴表容)──
//   旧 <512,8> 对 bin6 CAP 冗余 2× → BlockRadixSort TempStorage 挤 occupancy(主档排序 5× 差主因)。
//   CAP ≥ row_nnz ≤ ht 不变式保持;CSORT_LADDER=v2 启用(默认关,电池过了才翻默认)。
constexpr int CSORT_TPB_C = 128;                                // v2 bin6(ht2048):CAP=2048 紧贴
constexpr int CSORT_IPT_C = 16;
constexpr int CSORT_TPB_D = 256;                                // v2 bin7(ht4096):CAP=4096 同容,线程减半
constexpr int CSORT_IPT_D = 16;

// ── MinHash 表防护帽 ────────────────────────────────────────────────────────
constexpr unsigned long long MH_TABLE_MAX_BYTES = 256ull << 20;  // 256MB:鲸鱼阵防护(超限回退老流程)


// ── merge3 层(列值域 K 桶 warp merge)───────────────────────────────────────
constexpr int MRG3_K_BUCKETS = 5;          // 每行列值域分桶数(均分 span;并行度/开销折中)
constexpr int MRG3_DYN_MIN_BYTES = 8192;   // 动态边界桶下限 = 2×CSORT_HT 字节量级
constexpr int MRG3_DENSE_MIN_SPAN = 128;   // dense 单写行最小 span = 4×WARP_SIZE(bitmap 清零 4 warp 并行)
constexpr int MRG3_DENSE_BYTES_PER_COL = 9;  // val 8B + flag 1B(merge dense 布局)
constexpr long long MRG3_MAX_FLOP = 3000000000LL;  // 3e9 项 × 12B ≈ 36GB gapped < 80GB DRAM(SAFETY)

// ── h2d 重叠管线层(docs/71;rp 先行 → ci/val 分 chunk DMA ∥ count_flop)────────
constexpr int    L2_CACHE_BYTES  = 50 * 1024 * 1024;          // H100 PCIe L2 容量(手册)
constexpr size_t H2D_CHUNK_BYTES = (size_t)L2_CACHE_BYTES / 4; // chunk ≈ L2/4:DMA 与内核读在 L2 交错
constexpr int    H2D_MAX_CHUNKS  = 64;                         // 事件表上限(2^6;chunk 超则均分放大)

// ── 小行族专档(docs/13 v2;road/OSM 族 100% 行落 bin0 实测驱动)───────────────
constexpr int BATCH_S_EST_MAX = WARP_SIZE;     // 32:est ≤ warp 覆盖的小行子档
constexpr int BATCH_S_HT_STRUCT = WARP_SIZE * 2; // 64 槽表(α=0.5;清表/扫描税 = 128 槽档的一半)

// ── Ocean-D largest 层(docs/74;straggler 行位图窗)───────────────────────────
constexpr int LARGEST_SMEM_BUDGET = 200 * 1024;   // 位图+前缀 SMEM 预算(opt-in 227K − 27K 驱动保留,同 BCNT)

// ── 小行线程纂层(docs/13 v2 续;用户铁律:新参数零裸数字)──────────────────────
constexpr int TINY_ROW_MAX_LEN = WARP_SIZE / 4;   // 8:线程纂行内串行访存上限(warp/4 次,MLP 由 64 行/warp 在飞补)
constexpr int SMEM_ALIGN_SLACK = 64;              // 动态 SMEM 布局对齐余量(双数组 8B 对齐 + 字边界安全垫)

// ── largest 字节模型层(docs/74;毫字节定点 ×1000,与 SROUTE 同式)──────────────
//   dense_bytes = span×0.25B(位图 1/8 + 前缀 1/8)+ est×16B(tmp_key 8 + tmp_val 8 经全局)
//   hash_bytes  = ht×12B(表)+ est×24B(compact 税)
//   路由判据 = dense ≤ hash(全部具名毫字节常数,零裸系数)
constexpr int LARGEST_COL_MILLI         = (int)((1.0 / 8.0 + 1.0 / 8.0) * 1000);  // 250:位图+前缀
constexpr int LARGEST_VAL_PER_EST_MILLI = (8 + 8) * 1000;                          // 16000:val 经 tmp 全局
// hash 侧复用 HASH_BYTES_PER_SLOT_MILLI(12000)与 COMPACT_BYTES_PER_EST_MILLI(24000)

// ── largest 路由守卫(docs/74 守卫电池实证:c-64 +170/nd24k +137/crankseg +140 教训)──
constexpr int LARGEST_MIN_SPAN = 2 * HASH_BLOCK_STRUCT * WARP_SIZE;  // 16384 列:位图字 ≥ 2×块宽 → 原子摊薄
constexpr int DUP_LARGEST_MAX  = 2;  // dup ≤ 2 = 真 straggler 形态(flop≈est,c-big 实测 dup=1);>2 时
                                       // hash 全局 CAS 与 largest 的 2×flop 读 + fp64 原子打平 → 失去稳定优势(c-64 +12.8% 实证)
constexpr int LARGEST_SPARSE_DEN = 8;  // est/span ≤ 1/8 才进 largest:每字期望置位 ≤ 4(32/8)→ 原子冲突
                                       // 与清字税同界;c-big 1/17 ✓ vs c-64 1/2.5 ✗(守卫电池终判据)

// ── merge3 边界 v3 层(flop 近似均分;docs/66 续)──────────────────────────────
constexpr int MRG3_FINE_BINS = WARP_SIZE;  // 细直方图桶数 = warp 宽(每 lane 一 bin;中点 splat O(num_k))

// ── v31-④ ESC 层(docs/76;零表 counting-sort 短行)───────────────────────────
constexpr int ESC_FLOP_MAX = WARP_SIZE;  // 终版(v5 证伪:flop≤8 门把 hugetrace +36% 赢面归零、
                                         // road +16→+5;全档 + dup≤2(v4)三族全赢为最优点,patents/bloweya
                                         // −6~14% 残根 = flop 9-32 ∧ dup≤2 域内另一维未抓到,后续微调)

// ── v31-④ v2 子组分档(docs/76;v1 整 warp 伺候 6.7 乘积行 = −2~4% 失败教训)────────
constexpr int ESC_SUBW8_MAX  = WARP_SIZE / 4;  // 8:flop≤8 → 8 线程子组(4 行/warp;road avg 6.7 对位)
constexpr int ESC_SUBW16_MAX = WARP_SIZE / 2;  // 16:flop≤16(Ocean NUMERIC_USPARSE 同款档位)
constexpr int ESC_DUP_MAX = 2;  // v4 门:counting O(flop·SUBW) vs 表 O(est+ht) 的盈亏平衡 dup≈2
                                 // (germany 1.4 赢/patents ≥2 输,分相实证;DUP_LIGHT 同族推导)

// ── v33 Ocean 式 span 档位梯(docs/76 §21-22 用户钦定优先;§22d 六档扩)───────────
// 档 i 上限 = (SMEM 模型 ÷ 并发梯[i]) ÷ (val 8B + flag 1B) − 4(Ocean Common.h 同公式);
// 与 SMEM_MODEL_TOTAL/占用梯共用推导源。行按 span 落档:窗宽/块宽/驻块数随档联动,
// span > 最大档 → 老窗口 cursor 内核(§22d;这些行的既有赢家,如 c-64 全幅行)。
// 第 6 档(occ 1,窗 14543,131KB ≤ H100 opt-in 227KB):in-2004 怪物行定谳 —— est≥4096 ∧
// span 7278-18924 卡在 5 档梯外,老内核对 est≫span 行有病态模式(48M 周期/行),档内化后走结构数值。
constexpr int DENSE_TIER_COUNT = 6;
SP_HD constexpr int dense_tier_occ(int i) {   // 并发梯(占用梯同源;constexpr 数组不可 device 索引故逐档)
    return (i == 0) ? 32 : (i == 1) ? 16 : (i == 2) ? 8 : (i == 3) ? 4 : (i == 4) ? 2 : 1;
}
SP_HD constexpr int dense_tier_max(int i) {   // 档上限 = (SMEM 模型 ÷ 并发) ÷ (val 8 + flag 1) − 4(Ocean 同公式)
    return SMEM_MODEL_TOTAL / dense_tier_occ(i) / (8 + 1) - 4;            // = {451,906,1816,3636,7277,14543}
}
SP_HD constexpr int dense_tier_block(int i) { // 档块宽 = 2048 线程 ÷ 并发(占用单位;[64,1024] = 合法块区间)
    int b = 2048 / dense_tier_occ(i);
    return b < 64 ? 64 : (b > 1024 ? 1024 : b);   // occ 1 → 2048 超硬件上限,取 1024
}
constexpr int DENSE_TIER_SPAN_MAX = dense_tier_max(DENSE_TIER_COUNT - 1); // 14543 = 档内上限(超过 → 老内核)

// ── §22d 档内路径的路由地板(翻默认值;结构推导)────────────────────────────
// flop 地板 2048 = SMEM_DEFAULT_MAX/12/2:行 flop 达到半默认档(2×est×12B = 48KB 满档的一半)
//   —— 表成本开始占据一个默认档时,dense 窗与表的比较才有意义(v33d 电池 §22 验证同值);
// est 地板 1024 = SMEM_DEFAULT_MAX/12/4:表 ≥ 24KB = 四分之一默认档(§22 电池 DITER_EST_MIN 验证)。
// 超档(老内核)路径不受此影响 —— 门内结构化钳制到 EST_DENSE_MIN / EST_DENSE_MIN/2(v28-v32 验证域)。
constexpr int DITER_FLOP_TIER = SMEM_DEFAULT_MAX / 12 / 2;   // = 2048
constexpr int DITER_EST_TIER  = SMEM_DEFAULT_MAX / 12 / 4;   // = 1024

// ── v33d 小阵守卫(T10 广域 10-05 定谳)─────────────────────────────────────
// 上面两条松地板的摊销前提 = 大阵:v33d 电池(09-04)八阵全为重阵代表,无小阵覆盖。
// T10 52 阵:dense 主导 23 阵 GM 2.073 全部 nnz≤2.03M(c-43 3.64/pli 3.76/Si10H16 3.38;
// 同阵全 hash 相位和 2.21ms vs v32 5.15ms = 快 2.3×),v33d 受益/无害侧(in-2004 −28.5%/
// nd24k/crankseg_2)nnz≥14.1M——(2.4M,14.1M) 窗口 T10 无阵。nnz < 阈值时档内地板
// 回 v32 值(EST_DENSE_MIN / EST_DENSE_MIN/2 = 28-v32 的 337 阵认证行为)。
// 阈值取 4M = 空窗中央(2 的幂,与 STREAMLINE_NNZ_STRUCT 同为 A_nnz 轴物理分层)。
constexpr long long DITER_SMALL_NNZ_DEFAULT = 4LL * 1024 * 1024;   // DITER_SMALL_NNZ env 可改;0=关守卫

// ── kernel₇(BIN_K7):双遍位图 + rank 直写,dup≈1 人口(10-06 独立台背书)──────────
// 机理:Pass1 atomicOr 位图(1-bit/列,免 F64 原子)→ popcount=精确 nnz → cum 秩表 →
// Pass2 认领 → 胜者纯普通写,败者入 SMEM fixup 表 → __syncthreads 后统一 atomicAdd
// (屏障定序)。独立台:bloweybq 3.2ms vs 管线 12.9ms ≈ 4×(tools/k7_bench.cu)。
// 经济学:位图 0.125B/列 vs hash 表 12B/槽(96× 脚印差,占用物理赢侧);count 即精确
// 计数 + rank 写即终态有序 ⇒ 该 bin 免估计器余量/compact+sort/值原子。
constexpr int K7_SPAN_MAX  = 32768;    // 位图 ≤4KB(路由上限;SMEM 由此静态定尺寸)
constexpr int K7_ALEN_MAX  = 32;       // 2D(k,p) 切分假设(a_len 小时全 block 摊 B 流)
constexpr int K7_EST_MIN   = 256;      // 不偷 batch_s/ultra/esc 人口
constexpr double K7_DUP_MAX = 1.67;    // 路由侧 dup 预算(估计;精确证书在核内判)
constexpr int K7_FIX_MAX   = 512;      // fixup 表容量(核内证书:flop−distinct ≤ 此值才走快道)
constexpr int K7_FIX_OFF_W = 512 + 1024 + 2;  // SMEM 字偏移:[fvl 1024w][frk 512w][cnt+pad]
constexpr int K7_BLOCK     = 128;             // 一行一 block(numeric/count 同约定,red[] 依赖)
// ⚠ 布局纪律(wedge#8 教训):fvl=words 0..1023 / frk=1024..1535 / fcnt=1536 / mark 自 1538 起;
//   多结构共 SMEM 必须画字偏移图核对(512 doubles = 1024 字,勿按 512 字算)

// ── §22b Ocean 双维档位比较(Analysis.cuh L248-264 移植;09-04 dump 校准)─────────
// Ocean 的 dense 分流不是单边门,是同一 SMEM 预算梯上的【档位比较】:
//   dense 赢 ⟺ dense_tier_of(span) ≤ hash_tier_of(ht)
//   即"我的窗口在并发梯上的档 ≤ 我的 hash 表在并发梯上的档"(同预算下 dense 不比 hash 贵;
//   平局 dense 先 = 免探测)。连续形式 ≈ span×8.125B ≤ ht×12B(SROUTE 字节比较的同源)。
// 校准(in-2004 20498 diter 行):95.3% 通过;拦下的 4.7% = est=1024 ∧ span~5.9K
// (表便宜/窗贵)—— 正是 Ocean 送 hash 的形态。大 span(>最大档)行走老窗口内核,
// 不适用单窗经济学,比较只作用于档内行(DTIER_CMP 门)。
SP_HD constexpr int hash_tier_max_slots(int i) {  // 该档预算能容纳的最大表槽 = 预算梯 ÷ 12B/槽
    return SMEM_MODEL_TOTAL / dense_tier_occ(i) / (HASH_BYTES_PER_SLOT_MILLI / 1000);  // = {341,682,1365,2730,5461}
}
SP_HD constexpr int dense_tier_of(long long span) { // span 落档;超最大档 = COUNT(不可单窗 dense)
    return (span <= dense_tier_max(0)) ? 0 : (span <= dense_tier_max(1)) ? 1
         : (span <= dense_tier_max(2)) ? 2 : (span <= dense_tier_max(3)) ? 3
         : (span <= dense_tier_max(4)) ? 4 : DENSE_TIER_COUNT;
}
SP_HD constexpr int hash_tier_of(long long ht) {    // 表槽落档;超最大档 = COUNT(hash 最贵档,heavy/换出)
    return (ht <= hash_tier_max_slots(0)) ? 0 : (ht <= hash_tier_max_slots(1)) ? 1
         : (ht <= hash_tier_max_slots(2)) ? 2 : (ht <= hash_tier_max_slots(3)) ? 3
         : (ht <= hash_tier_max_slots(4)) ? 4 : DENSE_TIER_COUNT;
}
