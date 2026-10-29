#include "spgemm.h"
#include "hash_prof.h"
#include "structural_params.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

// hash SPA:C = A·A,SMEM hash 累加器(atomicCAS 插列+atomicAdd 累值);pipeline count→累加→compact→sort;overflow→回退

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

extern __global__ void count_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub);

// 结构化 dense 内核(docs/72 Phase 1a,spgemm_kernel_struct.cu)
extern __global__ void dense_struct_count_kernel(
    const int *A_rp, const int *A_ci,
    const int *B_rp, const int *B_ci,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols, int *row_nnz);

// docs/72 Phase 1b:numeric 直写(位图字级 popcount 前缀发射,无 warp0 串行)
void dsn_rt_attach(unsigned long long *p);        // §26 采数代理(struct.cu 内 device symbol)

// ===== P1 per-file 三层(study/58 附录A.1 v2,探针定谳 0964aa8)=====
// 本文件双编:TU-90(缺省/显式 =90,sm_90 原生,全 host 区)= 现行为;
// TU-61(=61,compute_61 gencode)= 仅 device 模板区 + 大表(bins≥4)accumulate 发射器。
// 探针三定谳:非模板 kernel 已全 static(P3);host 发射区整域门防未打签模板实例弱合并(P2);
// 运行时形态实例重叠由 TB_TAG 模板参解(cutover 段)。
#ifndef HASH_TB
#define HASH_TB 90
#endif
template<int TPB>
extern __global__ void dense_struct_numeric_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz,
    const int *row_flop);   // 刀D2a:精确 avg_B(G 分配用)
// D3(T7c):低 dup 短 a_len merge 核(pre2 型;env D3EXP)
extern __global__ void d3_merge_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *rows, int n_rows,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz,
    unsigned long long *rowtime);
// T3(10-04):两行共块变体(命名屏障+SMEM 半分区;env T3EXP)——模板声明(实例子段在 struct.cu)
template<int TPB> __global__ void dense_struct_numeric_2r_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows, int nr_bucket, const int *span_lo, const int *span_len,
    int win_cols,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz,
    const int *row_flop);

// 二分:返回 [lo,hi) 内第一个 arr[idx] >= val 的下标(与 spgemm_merge.cu 同款,跨 TU 内联用本地拷贝)
__device__ __forceinline__ int dev_lower_bound(const int *arr, int lo, int hi, int val) {
    while (lo < hi) { int mid = (lo + hi) >> 1; if (arr[mid] < val) lo = mid + 1; else hi = mid; }
    return lo;
}

// 并行 count flop_ub:warp-per-row(2026-08-25 重映射;旧版每行一 block,3.7M 行矩阵 launch 5.4ms → 此版 O(rows/8) blocks)
static __global__ void count_intermediates_par_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = warp;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        s += A_row_ptr[k + 1] - A_row_ptr[k];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if (lane == 0) ub[i] = s;
}

// 融合版(docs/27 §4.6):warp-per-row 一趟出 flop + span_lo/len + max_b_len(省 1 launch + 一趟 A×B 遍历)
// ⚠ 仅 AA 路径(count 的外层 = dB_rp 与 row_span 的 dA_rp 在 att 下语义不同,att 保留双核)
static __global__ void count_flop_span_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *ub, int *span_lo, int *span_len, int *max_b_len)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = warp;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0, lo = INT_MAX, hi = -1, mb = 0;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        s += b1 - b0;
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            lo = min(lo, c0); hi = max(hi, c1); mb = max(mb, b1 - b0);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, off);
        lo = min(lo, __shfl_down_sync(0xFFFFFFFF, lo, off));
        hi = max(hi, __shfl_down_sync(0xFFFFFFFF, hi, off));
        mb = max(mb, __shfl_down_sync(0xFFFFFFFF, mb, off));
    }
    if (lane == 0) {
        ub[i] = s;
        span_lo[i] = (lo == INT_MAX) ? 0 : lo;
        span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
        if (max_b_len) max_b_len[i] = mb;
    }
}

// count 小行线程版(docs/13 v2):avg row ≤ 8 的 road/OSM 族 —— warp 版 27/32 lane 闲;
// 线程级行串行 4-8 次访存,MLP 由 64 行/warp 在飞补(count 7.6ms → 目标 ~4)
static __global__ void count_flop_tiny_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *ub, int *span_lo, int *span_len, int *max_b_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0, lo = INT_MAX, hi = -1, mb = 0;
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        s += b1 - b0; mb = max(mb, b1 - b0);
        if (b0 < b1) { int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1]; lo = min(lo, c0); hi = max(hi, c1); }
    }
    ub[i] = s;
    span_lo[i] = (lo == INT_MAX) ? 0 : lo;
    span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
    if (max_b_len) max_b_len[i] = mb;
}

// span-only tiny 形(avg≤8;线程级行)
static __global__ void row_span_only_tiny_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *span_lo, int *span_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int lo = INT_MAX, hi = -1;
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            lo = min(lo, c0); hi = max(hi, c1);
        }
    }
    span_lo[i] = (lo == INT_MAX) ? 0 : lo;
    span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
}

// h2d 重叠管线的行区间版(docs/71):只算 flop + max_b_len —— 数据依赖 = row_ptr(先行小拷贝)
// + 本 chunk 的 A_ci 行区间;span(B_ci 随机访问,须全量)由同步后的 row_span_only_kernel 补。
static __global__ void count_flop_range_kernel(
    const int *A_row_ptr, const int *A_col_idx, int r0, int r1,
    const int *B_row_ptr, int *ub, int *max_b_len)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = r0 + warp;
    if (i >= r1) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0, mb = 0;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        s += b1 - b0;
        mb = max(mb, b1 - b0);
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, off);
        mb = max(mb, __shfl_down_sync(0xFFFFFFFF, mb, off));
    }
    if (lane == 0) {
        ub[i] = s;
        if (max_b_len) max_b_len[i] = mb;
    }
}

// 重叠管线 tiny 形(avg≤8 矩阵):线程级行 + 行区间(与 COUNT_TINY 正交组合;
// H2DOVL 曾旁路 COUNT_TINY → M6/HTC 族 warp 形 count 超窗 +23% 回归,2026-09-04 隔离电池实证)
static __global__ void count_flop_range_tiny_kernel(
    const int *A_row_ptr, const int *A_col_idx, int r0, int r1,
    const int *B_row_ptr, int *ub, int *max_b_len)
{
    int i = r0 + (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= r1) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0, mb = 0;
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        s += b1 - b0; mb = max(mb, b1 - b0);
    }
    ub[i] = s;
    if (max_b_len) max_b_len[i] = mb;
}

// span-only 补趟(h2d 重叠管线;A_ci/B_ci 已全量在 device):每 k 只读 B 行首/末 2 元素
static __global__ void row_span_only_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *span_lo, int *span_len)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = warp;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int lo = INT_MAX, hi = -1;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            lo = min(lo, c0); hi = max(hi, c1);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        lo = min(lo, __shfl_down_sync(0xFFFFFFFF, lo, off));
        hi = max(hi, __shfl_down_sync(0xFFFFFFFF, hi, off));
    }
    if (lane == 0) {
        span_lo[i] = (lo == INT_MAX) ? 0 : lo;
        span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
    }
}

#ifndef HASH_CAP
// HASH_CAP:opt-in SMEM 预算÷12B/槽(硬件推导,见 structural_params.cuh)
#define HASH_CAP HASH_CAP_STRUCT
#endif
#define HASH_BLOCK HASH_BLOCK_STRUCT  // SM 占用单位(硬件手册)
#define N_BINS 26    // …+ largest(20) + esc8/16/32(21-23) + esc64(24,study/50) + esc_rank128(25,T3)
#define BIN_SSPAN0 14   // span≤256(64线程/3.3KB,32+行/SM —— Ocean dense bin0 同位)
#define BIN_ULTRA 11
#define BIN_HEAVY 12
#define BIN_DITER 13
#define BIN_BATCH2 18    // docs/67 §8:est 65-256 ∧ kc≤32 小行批处理(<512,4> warp-per-row)
#define BIN_BATCH_S 19   // docs/13 v2:est≤32 ∧ kc≤32 小行 64 槽表批处理(road 族 100% bin0 实测驱动)
#define BIN_LARGEST 20   // docs/74 Ocean-D:straggler 行位图窗(SMEM 0.25B/列 + val 全局流过)
#define BIN_ESC8 21     // v31-④ v2:flop≤8 → 8 线程子组(4 行/warp;road avg 乘积 6.7 的对位档)
#define BIN_ESC16 22    // flop≤16 → 16 线程子组
#define BIN_ESC32 23    // flop≤32 → 32 线程子组(=v1 形态)
#define INT_SENTINEL 0x7ffffffe   // ESC64 装载内部哨兵(区别于键哨兵 0x7fffffff)
#define BIN_ESC64 24     // study/50 靶1(10-20):flop∈(32,64] → 1 warp/行、IPT=2;delaunay 网格族 79.3% 行在此带(批437 flop 直方图),原落 bin0 表路径=ocean usparse 吃这带的 1.8× accumulate 残差
#define BIN_ESC128 25    // T3(study/53):flop∈(64,128] → esc_rank_kernel<128>(smem 广播秩排序,对位 ocean usparse)
#define BIN_K7 15        // kernel₇:双遍位图+rank 直写,dup≈1∧窄span∧短a_len 人口(10-06;K7 env,默认关)
#define BATCH_S_HT BATCH_S_HT_STRUCT  // 64 = 2×32(α=0.5)
#define GLOBAL_HT_MAX_SLOTS GLOBAL_HT_MAX_STRUCT  // 内存预算÷12B/槽
#define AVG_FLOP_THR AVG_FLOP_THR_STRUCT  // warp_size×2(结构推导)
#define CSORT_HT CSORT_HT_STRUCT  // 算法渐近交叉(count-sort vs 比较排序)
// spec 81 两级计数排序(HASH_EXTRACT=2):小行 extract 的 O(d²) rank → O(d + 1024)。
// 门槛 ht≥512:d≤256 档 O(d²) 税轻(≤65K op),且小 bin 表仅 3-6KB,加 12KB hist 会压 occupancy。
// 寄存器 staging 上限 4 槽/线程(count ≤ 4×blockDim;块梯窄配/异常布局自动回退 legacy,零风险)。
#define CS2_MIN_HT 1024  /* study/54(10-21 定谳):smem 预算线 bin5+(行级门见 CS2_MIN_COUNT) */
#define CS2_MIN_COUNT 384 /* study/54:行级 d 门——rank O(d²) 与 count-sort 五趟的实测交叉;
                             * d≤273(ldoor)rank 赢 / pre2 bin5 大行 count-sort 赢;bins2-4 天然 d≤256 免预算 */
#define CS2_STAGE 4

// MinHash 概率基数估计:替 flop_ub 定 tmp buffer + hash 表大小
#define MH_M MH_M_STRUCT  // SMEM÷4B(HLL 寄存器)
#define EST_EXPAND EST_EXPAND_STRUCT  // MinHash σ=1.04/√k 的 95% 置信补偿
#define EST_ULTRA_THR EST_ULTRA_THR_STRUCT  // warp_size/2(线性扫描 < hash 探测)
#define STREAMLINE_NNZ STREAMLINE_NNZ_STRUCT  // MinHash 两阶段 vs flop count 交叉
#define WARP_SIZE 32  // NVIDIA 架构常数(硬件手册)

// ── 刀F 前置(study/57 §四十五):compute_61 兼容层——sm_80+ warp 原语的软件回退。
// 生产(-arch=sm_90)走硬件原语零变更;compute_61 编译(编译 flags 实验刀)走软件版。
// WM/warp-merge 与 scatter 聚合路径在实验刀中默认关,软件版保编译通过+语义正确。
#if __CUDA_ARCH__ && __CUDA_ARCH__ < 800
SP_HD static unsigned sp_match_any_sync(unsigned mask, unsigned val) {
    unsigned parts = __activemask() & mask, r = 0;
    for (int off = 0; off < 32; off++)
        if ((parts >> off) & 1) { unsigned ov = __shfl_sync(parts, val, off); if (ov == val) r |= (1u << off); }
    return r & mask;
}
SP_HD static unsigned sp_reduce_max_sync(unsigned mask, unsigned val) {
    for (int off = 16; off; off >>= 1) { unsigned o = __shfl_down_sync(mask, val, off); if (o > val) val = o; }
    return __shfl_sync(mask, val, 0);
}
#define __match_any_sync(mask, val)  sp_match_any_sync((mask), (val))
#define __reduce_max_sync(mask, val) sp_reduce_max_sync((mask), (val))
#endif

// MurmurHash3(full mix body+fmix32);MinHash 两阶段复用,SPA 累加暂仍用 Knuth 乘法
__device__ __forceinline__ unsigned int rotl32(unsigned int x, int r) { return (x << r) | (x >> (32 - r)); }
__device__ __forceinline__ unsigned int murmur_hash3(unsigned int key) {
    unsigned int h1 = 1234u;                       // MURMUR3_SEED
    unsigned int c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
    unsigned int k1 = key;                         // index_t = uint32 → 单 block,len=4
    k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2;
    h1 ^= k1; h1 = rotl32(h1, 13); h1 = h1 * 5u + 0xe6546b64u;
    h1 ^= 4u;                                      // ^= len
    h1 ^= h1 >> 16; h1 *= 0x85ebca6bu;             // fmix32
    h1 ^= h1 >> 13; h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
}

// MinHash 概率基数估计(partition bottom-k sketch)
#define MH_EMPTY 0xffffffffu

// Phase 1:对 B(=A)每行建 MinHash sketch(线性扫 CSR,O(nnz))。每元素 atomicMin keep 最小完整 hash(uint32)。
static __global__ void mh_construct_kernel(
    const int *B_row_ptr, const int *B_col_ind,
    int B_rows, int B_nnz,
    unsigned int *global_mh,           // [B_rows * MH_M] uint32
    int rows_per_block,
    const int *avg_gate = nullptr)     // PIPEFUSE:门命中(avg_product≤64)→ 空转(est=flop 由 merge 直填)
{
    if (avg_gate && *avg_gate == 1) return;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, B_rows);
    if (row_start >= B_rows) return;

    int elem_start = B_row_ptr[row_start];
    int elem_end = (row_end < B_rows) ? B_row_ptr[row_end] : B_nnz;
    int num_rows = row_end - row_start;

    extern __shared__ int smem_raw[];
    unsigned int *smem = (unsigned int*)smem_raw;          // [num_rows * MH_M] uint32 scratch(atomicMin 用)
    int total_items = num_rows * MH_M;
    for (int i = threadIdx.x; i < total_items; i += blockDim.x) smem[i] = MH_EMPTY;   // init 空(非 0)
    int *row_offsets = smem_raw + total_items;
    for (int i = threadIdx.x; i <= num_rows; i += blockDim.x)
        row_offsets[i] = (row_start + i < B_rows) ? B_row_ptr[row_start + i] : elem_end;
    __syncthreads();

    int current_row = 0;
    unsigned int *current_scratch = smem;
    for (int e = elem_start + threadIdx.x; e < elem_end; e += blockDim.x) {
        unsigned int col = (unsigned int)B_col_ind[e];
        while (e >= row_offsets[current_row + 1]) {
            current_row++;
            current_scratch = smem + current_row * MH_M;
        }
        unsigned int h = murmur_hash3(col);
        int idx = h & (MH_M - 1);                  // 低 log2(MH_M) 位 → partition
        atomicMin(&current_scratch[idx], h);       // MIN 完整 hash(keep 最小值)
    }
    __syncthreads();

    unsigned int *global_ptr = global_mh + row_start * MH_M;
    for (int i = threadIdx.x; i < num_rows * MH_M; i += blockDim.x) global_ptr[i] = smem[i];
}

// Phase 2:对 A 每行逐 partition min 合并(单 warp,vectorized uint4 merge)
// 1a-1'(10-13 取证终报后重写):PPT×EST_CONT 双模板 + running min 全程寄存器化。
// 取证账本(578 warp指令/行):循环区 229/行 = 19条/元素 —— 真凶 = min 经 smem 往返
// (每元素 8 次 LDS/STS,1warp/block 时代遗产);est 尾部 188/行 = 双路径 DP + FSEL
// (est_cont 运行时参数 → snap/连续两路全编;DP 在 1:64 单元)。修法:①min 入寄存器,
// smem 与两处 __syncthreads 整体消失;②EST_CONT 模板特化(生产恒 1 → snap 死码);
// ③尾部降 SP(估计器固有 CV 9%,float 1e-7 误差零风险)。数学(逐 partition min、
// linear counting / 算术均值公式、flop 封顶)与 legacy 逐位同源。
template <int PPT, bool EST_CONT>
__global__ void mh_merge_kernel(
    const int *A_row_ptr, const int *A_col_ind,
    int A_rows,
    const unsigned int *b_mh,           // [B_rows * MH_M] uint32 from Phase 1
    const int *row_flop,                // [A_rows] 每行精确乘积数(精确上界,封顶 MinHash 高估)
    double expand,                      // EST_EXPAND(运行时)
    int *est_nnz,                       // [A_rows] output(est_cont 语义由模板承担)
    int sample_stride = 0,              // MHSAMP:>1 时紧凑采样 grid(rid=bid*R+w → row=rid*stride)
    const int *avg_gate = nullptr,      // PIPEFUSE:门命中 → est=flop 直填
    double rexp_lo = 0.0, double rexp_hi = 0.0,   // 行级 expand(study/22):E 域两锚点 log 插值
    int rexp_elo = 0, int rexp_ehi = 0,
    int n_work = 0)                     // grid 工作项数(全量=A_rows;MHSAMP=S)
{
    constexpr int K = PPT * WARP_SIZE;                 // mh_k 编译期(32/64/128)
    constexpr float K_SCALE = (float)((double)MH_M / (double)(PPT * WARP_SIZE));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rows_per_blk = blockDim.x >> 5;
    const int rid = (int)blockIdx.x * rows_per_blk + warp;
    const int row = (sample_stride > 1) ? rid * sample_stride : rid;
    const bool valid = (rid < n_work) && (row < A_rows);

    if (avg_gate && *avg_gate == 1) {          // device 门(PIPEFUSE):grid 一致,先于一切
        if (valid && lane == 0) est_nnz[row] = row_flop[row];
        return;
    }
    if (!valid) return;                        // 零 smem/零 syncthreads → 早退安全

    // ★ running min 寄存器化:每 lane 自持 PPT 个 partition(模板常量,完全展开)
    unsigned int rmin[PPT];
    #pragma unroll
    for (int k = 0; k < PPT; k++) rmin[k] = MH_EMPTY;

    const int start_elem = A_row_ptr[row];
    const int end_elem = A_row_ptr[row + 1];
    if (PPT == 4) {
    // 逐引用 B 行,vectorized uint4 min-merge(与 legacy 逐位一致;min 全程在寄存器)
    for (int e = start_elem; e < end_elem; e++) {
        int row_b = A_col_ind[e];
        const uint4 *bsk4 = (const uint4*)(b_mh + (size_t)row_b * MH_M);
        uint4 v = bsk4[lane];
        if (v.x < rmin[0]) rmin[0] = v.x;
        if (v.y < rmin[1]) rmin[1] = v.y;
        if (v.z < rmin[2]) rmin[2] = v.z;
        if (v.w < rmin[3]) rmin[3] = v.w;
    }
    } else {
    for (int e = start_elem; e < end_elem; e++) {
        int row_b = A_col_ind[e];
        const unsigned int *bsk = b_mh + (size_t)row_b * MH_M;
        #pragma unroll
        for (int k = 0; k < PPT; k++) {
            unsigned int v = bsk[lane * PPT + k];
            if (v < rmin[k]) rmin[k] = v;
        }
    }
    }

    // 归约+估计(零 smem:直接从寄存器读;SP 数学)
    float sum_min = 0.f;
    int V = 0;
    #pragma unroll
    for (int k = 0; k < PPT; k++)
        if (rmin[k] != MH_EMPTY) { sum_min += (float)rmin[k]; V++; }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_min += __shfl_down_sync(0xFFFFFFFF, sum_min, off);
        V += __shfl_down_sync(0xFFFFFFFF, V, off);
    }

    if (lane == 0) {
        float E;
        int v = V;
        if (v == 0) {
            E = 0.f;
        } else if (v < K) {
            E = (float)K * logf((float)K / (float)(K - v)) * K_SCALE;   // linear counting(小范围)
        } else {
            // min_j/2^32 ~ Exp(1/(D/m)) ⇒ D ≈ m²·2^32/Σmin(算术均值,无偏;2026-08-26 修
            // 调和均值 Jensen 与漏因子 m 双 bug —— 公式不动,仅精度降 SP)
            E = (float)K * K * 4294967296.f / sum_min * K_SCALE;
        }
        float ex = (float)expand;
        if (rexp_ehi > 0) {   // 行级 expand:两锚点 log 线性插值
            float t = (E <= (float)rexp_elo) ? 0.0f
                   : (E >= (float)rexp_ehi) ? 1.0f
                   : (logf(E) - logf((float)rexp_elo)) / (logf((float)rexp_ehi) - logf((float)rexp_elo));
            ex = (float)rexp_lo + ((float)rexp_hi - (float)rexp_lo) * t;
        }
        int temp = (int)(E * ex);
        if (temp < 1) temp = 1;
        int est;
        if (EST_CONT || temp <= EST_ULTRA_THR) {   // 模板特化:生产 EST_CONT=true → snap 死码
            est = temp;
        } else {
            int ht = WARP_SIZE;
            while (ht < temp && ht < GLOBAL_HT_MAX_SLOTS) ht <<= 1;
            est = ht;
        }
        int fl = row_flop[row];
        if (fl > 0 && est > fl) est = fl;
        est_nnz[row] = est;
    }
}

static __global__ void est_fill_kernel(const int *row_flop, int n, int *est, int A_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < A_rows) { int f = row_flop[i]; est[i] = f < n ? f : n; }
}

// Ocean-C v1(docs/64 机制 C):采样行精确 distinct(SMEM 开放寻址计数)→ est/exact 定点比值。
// 一行一块;行太轻(est<512)/太重(distinct 装不下 8192 表)→ 写 0 = 无效样本。
// host 取 q05 → 数据驱动 expand 替换手调 1.15/1.4(Ana2 的 safe=avg−z·√var 同哲学)。
#define OC_TAB OC_TABLE  // 采样精确计数表(结构推导见 structural_params)
static __global__ void oc_sample_kernel(
    const int *A_row_ptr, const int *A_col_idx, const int *B_row_ptr, const int *B_col_idx,
    const int *est, const int *row_flop, int A_rows, int stride,
    int *ratio_fp)   // [S] = est_i * 4096 / exact_i(定点);0 = 无效
{
    int i = blockIdx.x * stride;
    int tid = threadIdx.x;
    if (i >= A_rows) { if (tid == 0) ratio_fp[blockIdx.x] = 0; return; }
    int e = est[i];
    if (tid == 0) ratio_fp[blockIdx.x] = 0;   // 默认无效,成功路径覆写
    if (e < OC_EST_MIN || e > OC_EST_MAX) return;   // 重试易发中段;e≤OC_EST_MAX 留 2× 欠估余量防样本自剔偏置
    __shared__ int tab[OC_TAB];
    for (int s = tid; s < OC_TAB; s += blockDim.x) tab[s] = -1;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int cnt = 0;
    bool full = false;
    for (int p = rs; p < re && !full; p++) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + tid; q < ke; q += blockDim.x) {
            int col = B_col_idx[q];
            unsigned slot = ((unsigned)col * 2654435761u) & (OC_TAB - 1);
            int probes = 0;
            while (true) {
                int old = atomicCAS(&tab[slot], -1, col);
                if (old == -1) { cnt++; break; }
                if (old == col) break;
                slot = (slot + 1) & (OC_TAB - 1);
                if (++probes >= OC_TAB) { full = true; break; }
            }
            if (full) break;
        }
    }
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, off);
    __syncthreads();
    __shared__ int warp_tot[9];   // [8]=满表 OR 共识(任意线程满 → 整行无效)
    if (tid == 0) warp_tot[8] = 0;
    __syncthreads();
    if ((tid & 31) == 0) warp_tot[tid >> 5] = cnt;
    if ((tid & 31) == 0 && full) warp_tot[8] = 1;
    __syncthreads();
    if (tid == 0) {
        int distinct = 0;
        for (int w = 0; w < (blockDim.x >> 5); w++) distinct += warp_tot[w];
        if (warp_tot[8] == 0 && distinct > OC_MIN_DISTINCT && row_flop[i] > 0)
            ratio_fp[blockIdx.x] = (int)((long long)e * OC_RATIO_FP / distinct);
    }
}

// Ocean-C est 统一缩放(en/ed 定点);flop 紧界仍成立(est ≤ flop 恒真)
static __global__ void oc_rescale_kernel(int *est, const int *row_flop, int A_rows, int en, int ed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    long long e2 = (long long)est[i] * en / ed;
    int f = row_flop[i];
    est[i] = (int)(e2 > f ? f : e2);
}

// S1 死码剔除(第二处,10-29):MHKAPPA 两核休眠——死码伤活码(§S1 定谳,第六例)
#if 0
// ── 刀② MHKAPPA(study/59):采样κ标定 est=flop×κ(免全量 MinHash)──────────────
// kappa_sample:oc_sample 同体(精确 distinct smem 计数),比值反向 = distinct×Q12/flop
// (flop→distinct 折减因子);kappa_est:est[i]=flop[i]×κ_en/4096(flop=0 行 est=0)。
static __global__ void kappa_sample_kernel(
    const int *A_row_ptr, const int *A_col_idx, const int *B_row_ptr, const int *B_col_idx,
    const int *row_flop, int A_rows, int stride,
    int *ratio_fp)   // [S] = distinct_i * 4096 / flop_i(定点);0 = 无效(满表/flop0/distinct≤64)
{
    int i = blockIdx.x * stride;
    int tid = threadIdx.x;
    if (i >= A_rows) { if (tid == 0) ratio_fp[blockIdx.x] = 0; return; }
    if (tid == 0) ratio_fp[blockIdx.x] = 0;
    __shared__ int tab[OC_TAB];
    for (int s = tid; s < OC_TAB; s += blockDim.x) tab[s] = -1;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int cnt = 0;
    bool full = false;
    for (int p = rs; p < re && !full; p++) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + tid; q < ke; q += blockDim.x) {
            int col = B_col_idx[q];
            unsigned slot = ((unsigned)col * 2654435761u) & (OC_TAB - 1);
            int probes = 0;
            while (true) {
                int old = atomicCAS(&tab[slot], -1, col);
                if (old == -1) { cnt++; break; }
                if (old == col) break;
                slot = (slot + 1) & (OC_TAB - 1);
                if (++probes >= OC_TAB) { full = true; break; }
            }
            if (full) break;
        }
    }
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, off);
    __syncthreads();
    __shared__ int warp_tot[9];
    if (tid == 0) warp_tot[8] = 0;
    __syncthreads();
    if ((tid & 31) == 0) warp_tot[tid >> 5] = cnt;
    if ((tid & 31) == 0 && full) warp_tot[8] = 1;
    __syncthreads();
    if (tid == 0) {
        int distinct = 0;
        for (int w = 0; w < (blockDim.x >> 5); w++) distinct += warp_tot[w];
        if (warp_tot[8] == 0 && distinct > OC_MIN_DISTINCT && row_flop[i] > 0)
            ratio_fp[blockIdx.x] = (int)((long long)distinct * OC_RATIO_FP / row_flop[i]);
    }
}
static __global__ void kappa_est_kernel(const int *row_flop, int *est, int A_rows, int en) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    est[i] = (int)((long long)row_flop[i] * en / OC_RATIO_FP);   // κ 内含覆盖余量;est≤flop 恒真
}

#endif

// PIPEFUSE step-1(docs/58):count_flop 后不再回传 host —— device 侧规约 total_flop + avg_product
// 门,construct/merge 无条件背靠背 launch(门命中时自退化),host 读数推迟到 binning 已有同步。
// 动机:reduce 的 host 空档是 DVFS toll 源之一(345MHz 冷频率下小 kernel 全家变慢)。
static __global__ void flop_reduce_kernel(const int *d_flop, int A_rows, unsigned long long *d_total) {
    __shared__ unsigned long long s[256];
    int t = threadIdx.x;
    unsigned long long p = 0;
    for (int i = blockIdx.x * blockDim.x + t; i < A_rows; i += gridDim.x * blockDim.x)
        p += (unsigned long long)d_flop[i];
    // block 归约(256T = 8 warp:warp shfl → per-warp partial → 8 lane 终归约 → block atomic)
    for (int off = 16; off; off >>= 1) p += __shfl_down_sync(0xffffffff, p, off);
    if (t % 32 == 0) s[t / 32] = p;
    __syncthreads();
    if (t < 8) {
        unsigned long long v = s[t];
        for (int off = 4; off; off >>= 1) v += __shfl_down_sync(0xff, v, off);
        if (t == 0) atomicAdd(d_total, v);
    }
}
static __global__ void avg_gate_kernel(const unsigned long long *d_total, int A_rows, int thr, int *d_gate) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *d_gate = (A_rows > 0 && (double)*d_total / A_rows <= (double)thr) ? 1 : 0;
}

// localLoadBalance 移植(docs/27 §4.2,Ocean AccumulatorCommon.cuh:67 同款语义):
// 按行 (a_len, flop, max_b_len) 动态选 2^log_nthr 线程/k —— 起步均值【排除最长 B 行】(它是
// straggler,由后续双向夹逼吸收),再按 max_sub_iter vs num_iters 的 2× 失衡双向调 G。
// 我们无 warp 内在依赖,G 上限 = HASH_BLOCK(整 block 伺候一个 k)。返回 log2(G)。
__device__ __forceinline__ int local_load_balance(int n_element_a, int num_products, int max_elements_b,
                                                  int lbound, int ubound)
{
    if (num_products <= 0) return lbound;
    int avg_ops = (num_products - max_elements_b) / (n_element_a > 1 ? n_element_a - 1 : 1);
    if (avg_ops < 1) avg_ops = 1;
    int log_nthr = 31 - __clz((unsigned)avg_ops);
    if (log_nthr < 1) log_nthr = 1;
    if ((1 << log_nthr) * 3 < avg_ops * 2) log_nthr += 1;   // 取最近 2^k(非线性截断)
    if (log_nthr > ubound) log_nthr = ubound;               // 预钳位:防 1<<(ubound-log_nthr) 负移位 UB(Ocean 原版隐患)
    int a_elem_per_iter = 1 << (ubound - log_nthr);
    int num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    int max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
    while (max_sub_iter > num_iters * 2) {
        if (log_nthr >= ubound) break;
        log_nthr++;
        max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (num_iters > max_sub_iter * 2) {
        if (log_nthr <= lbound) break;
        log_nthr--;
        max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (a_elem_per_iter > n_element_a && (1 << log_nthr) < max_elements_b) {
        log_nthr++;
        a_elem_per_iter = 1 << (ubound - log_nthr);
    }
    if (log_nthr < lbound) log_nthr = lbound;
    if (log_nthr > ubound) log_nthr = ubound;
    return log_nthr;
}

// ===================== dense 累积器路径(2026-08-26,超越点) =====================
// 触发:n ≤ DENSE_MAX_N 且 输出足够稠密(avg_est/n ≥ DENSE_MIN_FRAC)—— exdata_1 类
// (6001²,est 1878/行 = 31% 稠密,76× dup)。SMEM 直接寻址表 vals[n]+flags[n]:
// 零探测、零 CAS(槽位即列号),纯 atomicAdd —— 高 dup 下原子链是 hash 与 Ocean 共有的
// 结构瓶颈,稠密阵直接绕开。对标 Ocean AccumulatorDense(他们的 FAT 工作流)。
// 有序性免费:j 升序扫 flags → 输出行有序,compact 全走 copy。
#define DENSE_MAX_N DENSE_MAX_N_STRUCT  // 190KB÷13B/列
#define DENSE_MIN_FRAC DENSE_MIN_FRAC_STRUCT  // 位图字利用率

static __global__ void hash_dense_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    int multi = 0, int smap_cap = 0)   // T6-v4:host 判 multi 传入(不逐行判)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int ak = re - rs;

    constexpr int T6_WIN = 1024;   // T6-v5:窄窗 1024(13KB)→ 16 块/SM 理论上限
    const int w = multi ? T6_WIN : n;
    const int nwin = multi ? (n + w - 1) / w : 1;

    extern __shared__ __align__(8) unsigned char dsmem[];
    int *smap = multi ? (int*)dsmem : nullptr;                         // [smap_cap]
    double *dval = multi ? (double*)(dsmem + (size_t)smap_cap * 4) : (double*)dsmem;
    unsigned char *dflag = (unsigned char*)((char*)dval + (size_t)w * 8);
    int *dpref = (int*)((char*)dval + ((((size_t)w * 9) + 3) / 4 * 4));   // ceil4(9w) 对齐(exdata n=6001 教训)

    long long base2 = row_off[i];
    int cap = est[i];
    int total_nz = 0;
    bool ovf = false;

    if (!multi) {
        for (int j = tid; j < n; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
        __syncthreads();
        for (int p = rs + warp; p < re; p += nw) {
            int k = A_col_idx[p];
            double a = A_val[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_col_idx[q];
                if (upper_tri && j < i) continue;
                atomicAdd(&dval[j], a * B_val[q]);
                dflag[j] = 1;
            }
        }
        __syncthreads();
        if (warp == 0) {
            int seglen = (n + 31) / 32;
            int lo = lane * seglen, hi = min(lo + seglen, n);
            int run = 0;
            for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
            int excl = run;
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up_sync(0xffffffff, excl, off);
                if (lane >= off) excl += v;
            }
            for (int j = lo; j < hi; j++) dpref[j] += excl - run;
        }
        __syncthreads();
        total_nz = dpref[n - 1] + dflag[n - 1];
        for (int j = tid; j < n; j += blockDim.x) {
            if (dflag[j]) {
                int rank = dpref[j];
                if (rank >= cap) {
                    if (!atomicExch(&row_ovf[i], 1)) { int q = atomicAdd(ovf_cnt, 1); ovf_rows[q] = i; }
                    atomicExch(overflow_flag, 1);
                    continue;
                }
                tmp_key[base2 + rank] = ((unsigned long long)i << 32) | (unsigned int)j;
                tmp_val[base2 + rank] = dval[j];
            }
        }
    } else {
        // T6-v4:多窗 + 游标(host 判,全行统一)
        for (int j = tid; j < ak && j < smap_cap; j += blockDim.x)
            smap[j] = B_row_ptr[A_col_idx[rs + j]];
        __syncthreads();

        for (int wi = 0; wi < nwin; wi++) {
            int c0 = wi * w, c1 = min(c0 + w, n);
            int ww = c1 - c0;
            for (int j = tid; j < ww; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
            __syncthreads();

            for (int g = warp; g < ak && g < smap_cap; g += nw) {
                int k = A_col_idx[rs + g];
                double a = A_val[rs + g];
                int bke = B_row_ptr[k + 1];
                int q0;
                if (g < smap_cap) { q0 = smap[g]; if (q0 < 0) continue; }
                else { q0 = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0); }   // ak 超 smap 帽:二分

                int q_break = bke;
                for (int q = q0 + lane; q < bke; q += 32) {
                    int bcol = B_col_idx[q];
                    if (bcol >= c1) { q_break = q; break; }
                    int j = bcol - c0;
                    if (j >= 0 && j < ww) {
                        if (!upper_tri || bcol >= i) {
                            atomicAdd(&dval[j], a * B_val[q]);
                            dflag[j] = 1;
                        }
                    }
                }
                for (int off = 16; off > 0; off >>= 1) {
                    int v = __shfl_down_sync(0xffffffff, q_break, off);
                    if (v < q_break) q_break = v;
                }
                if (lane == 0 && g < smap_cap) smap[g] = (q_break >= bke) ? -1 : q_break;
            }
            __syncthreads();

            if (warp == 0) {
                int seglen = (ww + 31) / 32;
                int lo = lane * seglen, hi = min(lo + seglen, ww);
                int run = 0;
                for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
                int excl = run;
                for (int off = 1; off < 32; off <<= 1) {
                    int v = __shfl_up_sync(0xffffffff, excl, off);
                    if (lane >= off) excl += v;
                }
                for (int j = lo; j < hi; j++) dpref[j] += excl - run;
            }
            __syncthreads();
            int wnz = (ww > 0) ? dpref[ww - 1] + dflag[ww - 1] : 0;
            for (int j = tid; j < ww; j += blockDim.x) {
                if (dflag[j]) {
                    int rank = total_nz + dpref[j];
                    if (rank >= cap) { ovf = true; continue; }
                    tmp_key[base2 + rank] = ((unsigned long long)i << 32) | (unsigned int)(c0 + j);
                    tmp_val[base2 + rank] = dval[j];
                }
            }
            total_nz += wnz;
            __syncthreads();
        }
        if (ovf) {
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
        }
    }
    if (tid == 0) row_nnz[i] = (total_nz <= cap) ? total_nz : cap;
}

// ===================== 小跨度 dense 4 子桶(docs/37 = 35 §3 设计 + 36 号 spECK/nsparse 弹药)=====================
// span∈{256,512,1024,2048} × 块{64,128,256,256}:1 行/块但小块+小 SMEM → 8-32 行/SM(Ocean dense 梯同位);
// 直接寻址免 CAS 免探测;warp0 前缀有序 extract → compact 走 copy。est≥span/2 由路由门保证利用率。
template<int SPAN, int TPB>
__global__ void hash_sspan2_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows,
    const int *span_lo, const int *span_len,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = TPB >> 5;
    extern __shared__ __align__(8) unsigned char ssm2[];
    double *dval = (double*)ssm2;                                    // [SPAN]
    unsigned char *dflag = ssm2 + (size_t)SPAN * sizeof(double);     // [SPAN]
    int *dpref = (int*)(ssm2 + (((size_t)SPAN * 9) + 3) / 4 * 4);    // [SPAN]
    int lo = span_lo[i];
    int w_ = span_len[i];
    if (w_ <= 0) w_ = 1;
    if (w_ > SPAN) w_ = SPAN;   // SAFETY:路由门已限
    for (int j = tid; j < w_; j += TPB) { dval[j] = 0.0; dflag[j] = 0; }
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs + warp; p < re; p += nw) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q] - lo;
            if (j < 0 || j >= w_) continue;
            int col = lo + j;
            if (upper_tri && col < i) continue;
            atomicAdd(&dval[j], a * B_val[q]);
            dflag[j] = 1;
        }
    }
    __syncthreads();
    if (warp == 0) {
        int seglen = (w_ + 31) / 32;
        int slo = lane * seglen, shi = min(slo + seglen, w_);
        int run = 0;
        for (int j = slo; j < shi; j++) { dpref[j] = run; run += dflag[j]; }
        int excl = run;
        for (int off = 1; off < 32; off <<= 1) {
            int v = __shfl_up_sync(0xffffffff, excl, off);
            if (lane >= off) excl += v;
        }
        for (int j = slo; j < shi; j++) dpref[j] += excl - run;
    }
    __syncthreads();
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    int count = dpref[w_ - 1] + dflag[w_ - 1];
    for (int j = tid; j < w_; j += TPB) {
        if (dflag[j]) {
            int rank = dpref[j];
            if (rank >= cap) {
                atomicExch(overflow_flag, 1);
                if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
                break;
            }
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)(lo + j);
            tmp_val[base + rank] = dval[j];
        }
    }
    if (tid == 0) row_nnz[i] = (count <= cap) ? count : (int)cap;
}

// ===================== dense 窗口版(2026-08-26 Step3,治 TSOPF 类) =====================
// n > SMEM 预算但输出稠密的矩阵:每行一 CTA,列域切成 ≤ DENSE_MAX_N 的窗口,
// 逐窗口 SMEM 稠密累加(每 k 用 lower_bound 切出窗口内切片,免重复处理),
// 窗口升序 → 提取拼接天然全局有序,零跨 CTA 同步。对标 Ocean denseNumericIterKernel。
static __global__ void hash_dense_window_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *bucket_rows /*nullable:bin 行列表;null=矩阵级模式,blockIdx 即行号*/,
    const int *span_lo /*nullable:每行列跨度起点;null=0(矩阵级模式扫全 [0,n))*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr /*Phase B v2-lite 动态组*/)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) unsigned char wsmem[];
    double *dval = (double*)wsmem;                                       // [W]
    unsigned char *dflag = wsmem + (size_t)DENSE_MAX_N * sizeof(double);
    int *dpref = (int*)(wsmem + (((size_t)DENSE_MAX_N * 9) + 3) / 4 * 4); // [W]

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    long long base = row_off[i];
    int cap = est[i];
    int out = 0;
    bool ovf = false;
    // 跨度窗口 v2(docs/24):窗口对准本行【真实列跨度】[span_lo, n)(由 span_kernel 预计算),
    // 不再从 0 扫到 n —— F2 类带状 FEM(span~3k vs n=71k)从 5 个全窗(99% 空扫)变 1 个紧窗。
    // 矩阵级模式(span_lo=null)保持扫全 [0,n)。extract 仍窗口升序 → 全局有序契约不变。
    int lo0 = span_lo ? span_lo[i] : 0;
    int nw_win = span_lo ? (n - lo0 + DENSE_MAX_N - 1) / DENSE_MAX_N
                         : (n + DENSE_MAX_N - 1) / DENSE_MAX_N;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * DENSE_MAX_N, c1 = min(c0 + DENSE_MAX_N, n);
        int w = c1 - c0;
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
        __syncthreads();
        for (int p = rs + warp; p < re; p += nw) {
            int k = A_col_idx[p];
            double a = A_val[p];
            int ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
            int ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_col_idx[q] - c0;
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                atomicAdd(&dval[j], a * B_val[q]);
                dflag[j] = 1;
            }
        }
        __syncthreads();
        // 窗口内有序前缀(warp0 分段 + 段基址扫,同 dense v1)
        if (warp == 0) {
            int seglen = (w + 31) / 32;
            int lo = lane * seglen, hi = min(lo + seglen, w);
            int run = 0;
            for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
            int excl = run;
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up_sync(0xffffffff, excl, off);
                if (lane >= off) excl += v;
            }
            for (int j = lo; j < hi; j++) dpref[j] += excl - run;
        }
        __syncthreads();
        for (int j = tid; j < w; j += blockDim.x) {
            if (dflag[j]) {
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                int rank = out + dpref[j];
                if (rank >= cap) { ovf = true; continue; }
                tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)col;
                tmp_val[base + rank] = dval[j];
            }
        }
        out += dpref[w - 1] + dflag[w - 1];   // 窗口计数(w>0 因 n≥1 时首窗非空)
        __syncthreads();
    }
    if (ovf) {
        atomicExch(overflow_flag, 1);
        if (!atomicExch(&row_ovf[i], 1)) { int q = atomicAdd(ovf_cnt, 1); ovf_rows[q] = i; }
    }
    if (tid == 0) row_nnz[i] = (out <= cap) ? out : cap;
}

// ===================== 方案5:dense 行精确计数 + 直写终态(docs/27 §4.1,2026-08-27)=====================
// Ocean type1 工作流的核心经济性:两遍直写 vs 单遍 gapped 的交通量临界 = dup≈6~8(v4 门 dup<8 对齐)。
//   count pass:只标记 flag(1B/列,免 dval 8B+atomicAdd)→ row_nnz = 精确 distinct(无 cap,永不 ovf);
//   numeric pass:窗口升序直写 dC 精确偏移 —— dense 行的 gapped 写 + compact copy 整体消失。
// hash 行不动(结构已同 Ocean type2 est 工作流)。DIRECT5=1 开启,默认关。
#define DENSE_CNT_W DENSE_CNT_W_STRUCT  // count 位图档(默认 SMEM 档)

// count pass:窗口扫描标 flag + 块归约;窗口划分与数值 pass 不同也无妨( disjoint cover 计数同)。
// docs/70 位图 symbolic(Ocean denseSymbolicKernel 同构):span 一次位图【单遍】,
// 免窗口免每窗 per-k lower_bound 重搜(旧窗式的税 = 窗数×num_k×2×log 探测/行,Ga41 ≈16.5 万/行)。
// SMEM = span/32 字(dynamic,~220KB 档位 → span≤1.76M 全覆盖);QUERY_BITMAP 先读后 or
// 降争用(gupta3 教训);upper_tri 过滤同口径。BCNT=1 启用,span 超帽回退旧窗式。
// docs/70 §6 阶梯②:hash 行(bin 1-10)的位图 count —— 替 D5H 的 MODE=1(hash 表计数)
// 为位图置位+popcount(免 CAS 探测),复活 D5H 经济学。行级门 span ≤ 8×est ∧ 位图装得下
// (成本律:位图 O(span/32) vs 表 O(est),宽 span 低 est 行位图反亏 38×,in-2004 教训);
// 不合格行由 hash_spa<1> 双 launch 各司其职(bcnt_mode=1 只处理合格行,=0 只处理其余)。
static __global__ void hash_row_bcnt_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int upper_tri, int A_rows, int n,
    const int *bucket_rows, int n_in_bucket,
    const int *est, const int *span_lo, const int *span_len,
    int *row_nnz,
    int bcnt_mode = 0,        // 1=只处理 span≤8×est 行;0=其余(早退)
    int smem_words = 0)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int lo0 = span_lo ? span_lo[i] : 0;
    int span = span_len ? span_len[i] : (n - lo0);
    bool eligible = (8LL * est[i] >= (long long)span) && (span <= (long long)smem_words * 32);
    if (bcnt_mode == 1 ? !eligible : eligible) return;
    if (bcnt_mode == 0) return;   // 本 launch 只负责位图侧;hash 侧由 hash_spa<1>(带跳过谓词)跑
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) uint32_t bmp[];
    int words = (span + 31) >> 5;
    for (int w = tid; w < words; w += blockDim.x) bmp[w] = 0;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs + warp; p < re; p += nw) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int col = B_col_idx[q];
            if (col < lo0 || col >= lo0 + span) continue;
            if (upper_tri && col < i) continue;
            int bit = col - lo0;
            uint32_t m = 1u << (bit & 31);
            if ((bmp[bit >> 5] & m) == 0) atomicOr(&bmp[bit >> 5], m);
        }
    }
    __syncthreads();
    __shared__ int wcnt[32];
    int cnt = 0;
    int w4 = words & ~3;
    const uint4 *b4 = (const uint4*)bmp;
    for (int j4 = tid; j4 < (w4 >> 2); j4 += blockDim.x) {
        uint4 v = b4[j4];
        cnt += __popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w);
    }
    for (int w = w4 + tid; w < words; w += blockDim.x) cnt += __popc(bmp[w]);
    for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
    if (lane == 0) wcnt[warp] = cnt;
    __syncthreads();
    if (tid == 0) { int s = 0; for (int x = 0; x < nw; x++) s += wcnt[x]; row_nnz[i] = s; }
}

static __global__ void hash_dense_bcnt_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int upper_tri, int A_rows, int n,
    int *row_nnz,
    const int *bucket_rows,
    const int *span_lo,
    const int *span_len)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) uint32_t bmp[];   // [words] 位图
    int lo0 = span_lo ? span_lo[i] : 0;
    int span = span_len ? span_len[i] : (n - lo0);
    int words = (span + 31) >> 5;
    for (int w = tid; w < words; w += blockDim.x) bmp[w] = 0;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    // 单遍乘积:每 k 的【整条】B 行(无窗口切片,无 lower_bound)
    for (int p = rs + warp; p < re; p += nw) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int col = B_col_idx[q];
            if (col < lo0 || col >= lo0 + span) continue;      // 位图界外(理论不达,防御)
            if (upper_tri && col < i) continue;
            int bit = col - lo0;
            uint32_t m = 1u << (bit & 31);
            int w = bit >> 5;
            if ((bmp[w] & m) == 0) atomicOr(&bmp[w], m);        // QUERY_BITMAP:先读后 or 降争用
        }
    }
    __syncthreads();
    // 计数:向量化 popcount(4 字/iter)
    __shared__ int wcnt[32];
    int cnt = 0;
    int w4 = words & ~3;
    const uint4 *b4 = (const uint4*)bmp;
    for (int j4 = tid; j4 < (w4 >> 2); j4 += blockDim.x) {
        uint4 v = b4[j4];
        cnt += __popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w);
    }
    for (int w = w4 + tid; w < words; w += blockDim.x) cnt += __popc(bmp[w]);
    for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
    if (lane == 0) wcnt[warp] = cnt;
    __syncthreads();
    if (tid == 0) { int s = 0; for (int x = 0; x < nw; x++) s += wcnt[x]; row_nnz[i] = s; }
}

// ===================== kernel₇:双遍位图 + rank 直写(dup≈1 人口,10-06)=====================
// 独立台背书(tools/k7_bench.cu,bloweybq 3.2ms vs 12.9ms ≈4×;布局纪律见
// structural_params.cuh K7_FIX_OFF_W 注 —— wedge#8 = fvl/mark 重叠教训,勿改布局)。
// 路由前置:BIN_K7 只收 span≤K7_SPAN_MAX ∧ a_len≤K7_ALEN_MAX ∧ !att 的行,
// 故 SMEM 按上限静态定尺寸(W=K7_SPAN_MAX/32 字),无需 per-launch gather。
constexpr int K7_W = K7_SPAN_MAX / 32;                    // 1024 字 = mark/claim/cum 各一区
constexpr int K7_SMEM_W = K7_FIX_OFF_W + 3 * K7_W + 2;    // 总字数(含 fcnt+pad)
static __global__ void k7_count_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int A_rows, const int *bucket_rows, const int *span_lo,
    int *row_nnz, int *k7_err /*nullable:越界产品计数,守门响亮报错*/)
{
    extern __shared__ __align__(8) unsigned bmp[];        // [K7_W]
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    for (int w = threadIdx.x; w < K7_W; w += blockDim.x) bmp[w] = 0u;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int base = span_lo[i];
    int ak = re - rs, tg = blockDim.x / (ak > 0 ? ak : 1); if (tg < 1) tg = 1;
    int k_idx = threadIdx.x / tg, t_in = threadIdx.x % tg;   // 2D(k,p) 切分:a_len 小则全 block 摊 B 流
    if (k_idx < ak) {
        int j = A_col_idx[rs + k_idx];
        for (int p = B_row_ptr[j] + t_in; p < B_row_ptr[j + 1]; p += tg) {
            int c = B_col_idx[p] - base;                   // span_lo = 精确行 min(count_flop 算好)
            if ((unsigned)c < (unsigned)K7_SPAN_MAX)       // 防御带:路由外值响亮计数不静默
                atomicOr(&bmp[c >> 5], 1u << (c & 31));
            else if (k7_err) atomicAdd(k7_err, 1);
        }
    }
    __syncthreads();
    __shared__ unsigned red[K7_BLOCK];
    unsigned cnt = 0;
    for (int w = threadIdx.x; w < K7_W; w += blockDim.x) cnt += __popc(bmp[w]);
    red[threadIdx.x % K7_BLOCK] = cnt;                     // blockDim = K7_BLOCK(128)约定
    __syncthreads();
    for (int s = K7_BLOCK >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_nnz[i] = (int)red[0];
}

// numeric:Pass1 标记+flop 计 → 核内证书(flop−distinct ≤ K7_FIX_MAX ? 快道 : exch 慢道)
// → Pass2 写终态(有序+精确位)。快道 = 胜者纯普通写 + 败者 SMEM fixup 表 + 屏障后统一加。
static __global__ void k7_numeric_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int A_rows, const int *bucket_rows, const int *span_lo,
    const int *exact_rp, int *out_col, double *out_val,
    int *row_nnz /*nullable*/, int *k7_err /*nullable*/)
{
    extern __shared__ __align__(8) unsigned k7sm[];       // [fvl 1024w][frk 512w][fcnt+pad][mark][claim][cum]
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    unsigned *mark = k7sm + K7_FIX_OFF_W, *claim = mark + K7_W, *cum = mark + 2 * K7_W;
    double *fvl = (double *)k7sm;                         // words 0..1023(8B 对齐)
    int *frk = (int *)(k7sm + 1024);                      // words 1024..1535
    volatile int *fcnt = (volatile int *)(k7sm + 1536);   // word 1536
    for (int w = threadIdx.x; w < K7_W; w += blockDim.x) { mark[w] = 0u; claim[w] = 0u; }
    if (threadIdx.x == 0) *fcnt = 0;
    __shared__ unsigned red[K7_BLOCK];
    red[threadIdx.x] = 0u;
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int base = span_lo[i];
    int ak = re - rs, tg = blockDim.x / (ak > 0 ? ak : 1); if (tg < 1) tg = 1;
    int k_idx = threadIdx.x / tg, t_in = threadIdx.x % tg;
    unsigned myfl = 0;
    if (k_idx < ak) {
        int j = A_col_idx[rs + k_idx];
        for (int p = B_row_ptr[j] + t_in; p < B_row_ptr[j + 1]; p += tg) {
            int c = B_col_idx[p] - base;
            if ((unsigned)c < (unsigned)K7_SPAN_MAX) atomicOr(&mark[c >> 5], 1u << (c & 31));
            else { myfl++; if (k7_err) atomicAdd(k7_err, 1); }   // 防御带:越界产品不入位图也不写值
        }
    }
    red[threadIdx.x] = myfl;
    __syncthreads();
    for (int s = K7_BLOCK >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    int fl_tot = (int)red[0];
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int w = 0; w < K7_W; w++) { cum[w] = acc; acc += __popc(mark[w]); }
        cum[K7_W] = acc;
        if (row_nnz) row_nnz[i] = acc;
    }
    __syncthreads();
    // 核内证书:flop−distinct ≤ K7_FIX_MAX(含防御带越界产品)→ 快道;否则 exch 慢道
    bool fast = (fl_tot - (int)cum[K7_W] <= K7_FIX_MAX);
    int obase = exact_rp[i];
    if (k_idx < ak) {
        int j = A_col_idx[rs + k_idx];
        double a = A_val[rs + k_idx];
        for (int p = B_row_ptr[j] + t_in; p < B_row_ptr[j + 1]; p += tg) {
            int c = B_col_idx[p] - base;
            if ((unsigned)c >= (unsigned)K7_SPAN_MAX) continue;   // 防御带:与 count 同口径跳过
            unsigned bit = 1u << (c & 31);
            int word = c >> 5;
            unsigned old = atomicOr(&claim[word], bit);           // 认领:旧值无此位 = 首写
            int rank = (int)cum[word] + __popc(mark[word] & (bit - 1));
            double prod = a * B_val[p];
            if (fast) {
                if (!(old & bit)) {
                    out_col[obase + rank] = c + base;             // 胜者:纯普通写(免一切值原子)
                    out_val[obase + rank] = prod;
                } else {
                    int slot = atomicAdd((int *)fcnt, 1);
                    if (slot < K7_FIX_MAX) { frk[slot] = rank; fvl[slot] = prod; }
                    else atomicAdd(&out_val[obase + rank], prod); // 证书下不应达;达 = 响铃值门
                }
            } else {
                if (!(old & bit)) {
                    out_col[obase + rank] = c + base;
                    atomicExch(reinterpret_cast<unsigned long long *>(&out_val[obase + rank]),
                               __double_as_longlong(prod));       // 慢道:值原子(次序任意但只此路径)
                } else atomicAdd(&out_val[obase + rank], prod);
            }
        }
    }
    __syncthreads();
    if (fast) {
        int nf = *fcnt;
        for (int s = threadIdx.x; s < nf && s < K7_FIX_MAX; s += blockDim.x)
            atomicAdd(&out_val[obase + frk[s]], fvl[s]);          // 屏障后:确定性追加
    }
    __syncthreads();
}

static __global__ void hash_dense_count_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int upper_tri, int A_rows, int n,
    int *row_nnz,
    const int *bucket_rows /*nullable,同 window kernel*/,
    const int *span_lo /*nullable*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr /*Phase B v2-lite*/,
    int cnt_w = DENSE_CNT_W /*DCFUSE:动态窗宽 = min(DENSE_CNT_W, ceil4(n)),小阵 occupancy↑*/,
    int cnt_cursor = 0)                       // spec87:跨窗游标化(COUNT_CURSOR;0=每窗双二分旧行)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) unsigned char csmem[];   // [cnt_w] byte flags
    __shared__ int wcnt[32];                                 // blockDim 1024 = 32 warp
    // spec87:游标区=csmem 动态扩段(发射侧 csm 已加 CCUR_CAP×4B;禁静态 __shared__——首轮教训)
    int *scur = (int*)(csmem + (((cnt_w + 3) >> 2) << 2));
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    bool use_cur = cnt_cursor && (re - rs) <= 1024;   // 超帽(链>1024)回退双二分
    int total = 0;
    int lo0 = span_lo ? span_lo[i] : 0;
    int nw_win = (n - lo0 + cnt_w - 1) / cnt_w;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * cnt_w, c1 = min(c0 + cnt_w, n);
        int w = c1 - c0;
        for (int j = tid; j < w; j += blockDim.x) csmem[j] = 0;
        __syncthreads();
        for (int p = rs + warp; p < re; p += nw) {
            int k = A_col_idx[p];
            int ks, ke;   // spec87:首窗双二分;后续窗 ks=上窗 ke 直传(免 c0 二分),ke 再 lb 一次
            if (use_cur && wi > 0) {
                ks = scur[p - rs];
                if (ks < 0) continue;                       // 链耗尽哨兵
            } else {
                ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
            }
            ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
            if (use_cur) scur[p - rs] = (ke >= B_row_ptr[k + 1]) ? -1 : ke;
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_col_idx[q] - c0;
                if (j < 0 || j >= w) continue;
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                csmem[j] = 1;   // byte 写无争用(bitmap atomicOr 在窄列范围 = gupta3 +53% 教训)
            }
        }
        __syncthreads();
        // DCFUSE:向量化归约 —— 4 flag/iter(1×LDS.32 + popc)替代逐字节 LDS。flag 值域 {0,1}
        // ⇒ 每字节非零 ⟺ bit0 置位,__popc(v & 0x01010101) 即计数。
        // ⚠ 首版 v|v>>8|v>>16|v>>24 的"折叠"是字节渗漏 bug:高字节混入低字节位 → 零字节被
        // 误计(c-64 cnnz 155,006,041→164,556,377 = +6.2% 实锤;v23 前全对)。flag 单比特
        // 根本无需折叠。
        int cnt = 0;
        int w4 = w & ~3;
        const uint32_t *csm4 = (const uint32_t*)csmem;
        for (int j4 = tid; j4 < (w4 >> 2); j4 += blockDim.x) {
            uint32_t v = csm4[j4];
            if (v) cnt += __popc(v & 0x01010101u);
        }
        for (int j = w4 + tid; j < w; j += blockDim.x) cnt += csmem[j];
        for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
        if (lane == 0) wcnt[warp] = cnt;
        __syncthreads();
        if (tid == 0) { int s = 0; for (int x = 0; x < nw; x++) s += wcnt[x]; total += s; }
        __syncthreads();
    }
    if (tid == 0) row_nnz[i] = total;
}

#ifndef PB2_STAGE
#define PB2_STAGE 4
#endif
// dense 行 a_len 采集/散射(全局 cursor 区偏移构造)
static __global__ void span_gather_kernel(const int *rows, int nr, const int *span_len, int *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) out[j] = span_len[rows[j]];   // docs/70 BCNT:dense 行 span 收集
}

// §22d 分流体:span+行号 融合收集到 [nr, 2nr) 单缓冲(单次 D2H 取双数组,省 2 个同步点)
static __global__ void span_row_gather_kernel(const int *rows, int nr, const int *span_len, int *out2n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) { out2n[j] = span_len[rows[j]]; out2n[nr + j] = rows[j]; }
}

// §25 密度三元组 → §26 四元组:span+行号+est+a_len → [nr,4nr)
// (a_len = A_row_ptr 差分,在内核里算免多传一数组)
static __global__ void span_row_est_gather_kernel(const int *rows, int nr, const int *span_len, const int *est,
                                           const int *A_rp, int *out4n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) {
        int r = rows[j];
        out4n[j] = span_len[r]; out4n[nr + j] = r; out4n[2 * nr + j] = est[r];
        out4n[3 * nr + j] = A_rp[r + 1] - A_rp[r];
    }
}

// docs/70 §6②:合格行(span≤8×est ∧ ≤cap)的 span,否则 0 —— 供 host 定位图 SMEM 上限
static __global__ void elig_span_kernel(const int *rows, int nr, const int *est, const int *span_len,
                                 long long cap, int *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) {
        int i = rows[j];
        long long s = span_len[i];
        out[j] = (8LL * est[i] >= s && s <= cap) ? (int)s : 0;
    }
}

static __global__ void alen_gather_kernel(const int *rows, int nr, const int *A_rp, int *alen) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) { int r = rows ? rows[j] : j; alen[j] = A_rp[r + 1] - A_rp[r]; }
}
static __global__ void scatter_alen_kernel(const int *rows, int nr, const int *alen, int *off_by_row /*[A_rows],已清零*/) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) off_by_row[rows ? rows[j] : j] = alen[j];
}

// numeric pass:直写终态 + Ocean iter 内核机制(docs/30 Phase B v2):start_map 游标 + 动态组
// + 数据驱动窗口。三件套消除旧窗口内核的两个结构税:①每窗每 k 的 lower_bound 重复搜索
// (G>32 时冗余 = G×2log(avgB),c-58/case39/gupta1/TSOPF 回归主因);②固定 warp-per-k 的
// 线程利用率 32×a_len/512(mult_dcop 31% 占用 = 6.9× 差距主因)。游标:每 k 一个 SMEM 槽,
// 首窗一次 lower_bound 定位到 span_lo,越窗列回写 atomicMin(= O(1) 推进,免重复二分);
// 下一窗起点 = 全体越窗列的最小列(数据驱动,跳空窗)。窗口宽 PB2_W(SMEM 17B/列:
// val8+flag1+pref4+cursor4 → 12900 列 ≈ 219KB)。
#define PB2_W PB2_W_STRUCT  // 半SM预算÷21B/列(结构推导)
#define SMAP_SMEM_MAX SMAP_SMEM_MAX_STRUCT  // 与 PB2_W 同源
__device__ int g_dflag_probe_row = -1;   // DFLAG_PROBE 探针行(diter 雷判别;host 经 cudaMemcpyToSymbol 装入)
// __launch_bounds__(1024,2)(09-18,c-64 占用悬崖刀):cuobjdump 实测本 kernel REG:50 → 1024thr 下
// 65536/(1024×50)=1 块/SM = 50% warps(ncu 实测 49.97);ocean denseNumericIter REG:32 → 2 块 = 99.67%。
// c-64 头对头:我指令 4.01G(瘦 14%)仍慢 31% = 纯占用差。强制 2 块/SM → 编译器压 ≤32 regs
// (溢出换 2× 占用的交易,电池裁决)。
// K1 首写免原子(study/31):认领制——首写者普通写+fence,重复者才 atomicAdd(原子占比 100%→dup 比)
#define K1_ACC(dval_, j_, flag_, v_) atomicAdd(&(dval_)[j_], (v_))   // K1 判负废弃:降级原样(k1fw 死码)
static __global__ void __launch_bounds__(MAX_BLOCK_THREADS, 2) hash_dense_direct_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const int *exact_rp, int *dC_ci, double *dC_val,
    int *row_nnz,
    const int *bucket_rows /*nullable*/,
    const int *span_lo /*nullable*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr,
    int *smap_base = nullptr, const int *smap_off = nullptr, int smir_bias = 0,
    int use_cursor = 1, int k1fw = 0,
    unsigned long long *rowtime = nullptr,   // DD_ROWTIME:逐行执行周期(docs/66 采数)
    int *rowmode = nullptr,                  // 0=search 路径 / 1=cursor 路径(同数组落盘)
    int avgB_thr_param = 0,                  // PB2_AVGB 复审门(docs/66 §8;0 = 默认 64)
    const uint4 *B16p = nullptr,             // spec86-A:④交织 B 单股流(col+val 一次 uint4);空走旧双流
    int pf_on = 0,                           // spec86-B:书签预取(窗尾 touch 下窗链头进 L2;CURSOR_PF)
    int batch_on = 0,                        // spec89-#1:批量装载 2 发在飞(CURSOR_BATCH;默认关)
    int *tmp_col = nullptr, double *tmp_val = nullptr,   // spec89-#4'-L1:est-gapped tmp 输出对(非空=写 tmp 而非 dC 终态)
    const long long *gap_off = nullptr,     // #4'-L1:gapped 行基址(d_off);null=旧路逐位不变
    int *drop_cnt = nullptr)                // #4'-fix:静默丢检出器(免计数态 rank>=cap 即丢;host 读取判废
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    unsigned long long rt0 = rowtime ? clock64() : 0;
    extern __shared__ __align__(8) unsigned char wsmem[];
    double *dval = (double*)wsmem;
    unsigned char *dflag = wsmem + (size_t)PB2_W * sizeof(double);   // K1 布局回退(int 升格未门控曾致 cursor SMEM+33% → dense_direct 2× 污染 T10)
    int *dpref = (int*)(wsmem + (((size_t)PB2_W * 9) + 3) / 4 * 4);
    int *smap_g = smap_base + smap_off[i];                  // 全局游标(大 a_len 回退)
    int *smir_g = smap_base + smir_bias + smap_off[i];
    int *smap_s = dpref + PB2_W;                            // SMEM 游标区(与 dpref 分立)
    int *smir_s = smap_s + SMAP_SMEM_MAX;
    __shared__ int s_next_lo;

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int a_len = re - rs;
    int *smap = (a_len <= SMAP_SMEM_MAX) ? smap_s : smap_g; // docs/35 §3:SMEM 消每窗 O(a_len) 全局往返
    int *smir = (smap == smap_s) ? smir_s : smir_g;
    int base = tmp_col ? (int)gap_off[i] : exact_rp[i];   // #4'-L1:双形态基址(gapped 或精确)
    int cap = row_nnz[i];
    int out = 0;
    int lo0 = span_lo ? span_lo[i] : 0;

    // LLB 界:logG ∈ [log2(WARP_SIZE)=5, log2(MAX_BLOCK_THREADS)=10](G ≥ warp 才有组同步语义;G ≤ 块宽)
    int logG = (row_flop && row_maxbl) ? local_load_balance(a_len, row_flop[i], row_maxbl[i],
                                                            ilog2c(WARP_SIZE), ilog2c(MAX_BLOCK_THREADS))
                                       : ilog2c(WARP_SIZE);
    int G = 1 << logG, num_groups = blockDim.x >> logG;
    int my_group = tid >> logG, my_id = tid & (G - 1);

    for (int p = rs + tid; p < re; p += blockDim.x) {
        int k = A_col_idx[p];
        smap[p - rs] = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], lo0);
    }
    if (tid == 0) s_next_lo = n;
    __syncthreads();
    if (PB2_STAGE < 1) return;

    int avgB = (row_flop && a_len > 0) ? row_flop[i] / a_len : 0;
    if (rowmode && tid == 0) rowmode[i] = (avgB >= AVG_FLOP_THR && use_cursor == 1) ? 1 : 0;   // DD_ROWTIME 采数
    // docs/39 路由 v3 判据收回 avgB(host 级 dup 门见下);PB2_AVGB 复审门(docs/66 §8:
    // docs/39 时代判据未经逐行验证,0 = 全 cursor)。默认 64 = 原行为。
    int avgB_thr = AVG_FLOP_THR;   // 默认 = AVG_FLOP_THR(warp 覆盖交叉,docs/39)
    if (avgB_thr_param > 0) avgB_thr = avgB_thr_param;
    if (avgB < avgB_thr || use_cursor != 1) {
        // 混合路由(docs/30):avgB(=flop/a_len,平均 B 行长)< 64 的行走旧固定窗搜索路径 ——
        // 游标镜像/复位是每窗 O(a_len) 全局往返,开销/工作量 ∝ a_len×窗数/flop = 1/avgB;
        // TSOPF(avgB≈16)游标版 +22% 实锤,mult_dcop(avgB≈4000)游标 -42%。旧路径 = 52b89da 行为。
        int nw_win = span_lo ? (n - lo0 + PB2_W - 1) / PB2_W
                             : (n + PB2_W - 1) / PB2_W;
        for (int wi = 0; wi < nw_win; wi++) {
            int c0 = lo0 + wi * PB2_W, c1 = min(c0 + PB2_W, n);
            int w = min(PB2_W, n - c0);
            for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
            __syncthreads();
            for (int p = rs + warp; p < re; p += nw) {
                int k = A_col_idx[p];
                double a = A_val[p];
                int ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
                int ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
                for (int q = ks + lane; q < ke; q += 32) {
                    int j; double bv;                       // spec86-A:B16 单股流
                    if (B16p) { uint4 e = B16p[q]; j = (int)e.x - c0; bv = __hiloint2double(e.w, e.z); }
                    else { j = B_col_idx[q] - c0; bv = B_val[q]; }
                    if (upper_tri && (c0 + j) < i) continue;
                    if (k1fw) K1_ACC(dval, j, dflag, a * bv);
                    else { atomicAdd(&dval[j], a * bv); dflag[j] = 1; }
                }
            }
            __syncthreads();
            if (warp == 0) {
                int seglen = (w + 31) / 32;
                int lo = lane * seglen, hi = min(lo + seglen, w);
                int run = 0;
                for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
                int excl = run;
                for (int off = 1; off < 32; off <<= 1) {
                    int v = __shfl_up_sync(0xffffffff, excl, off);
                    if (lane >= off) excl += v;
                }
                for (int j = lo; j < hi; j++) dpref[j] += excl - run;
            }
            __syncthreads();
            for (int j = tid; j < w; j += blockDim.x) {
                if (dflag[j]) {
                    int rank = out + dpref[j];
                    if (rank >= cap) { if (drop_cnt) atomicAdd(drop_cnt, 1); continue; }   /* #4'-fix:不许无痕丢 */
                    if (tmp_col) { tmp_col[base + rank] = c0 + j; tmp_val[base + rank] = dval[j]; }
                    else { dC_ci[base + rank] = c0 + j; dC_val[base + rank] = dval[j]; }
                }
            }
            out += dpref[w - 1] + dflag[w - 1];
            __syncthreads();
        }
        if (tid == 0) {
            row_nnz[i] = out;   /* #4': tmp 模式需无条件写(count 已免,陈旧值会毒化晚 cnnz) */
            if (rowtime) rowtime[i] = clock64() - rt0;   // search 路径出口
        }
        return;
    }
    int win_lo = lo0;
    __shared__ int s_pf_sink;                // spec86-B:预取防 DCE 汇(只写不读)
    while (win_lo < n) {
        int w = min(PB2_W, n - win_lo);
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
        __syncthreads();
        if (tid == 0) s_next_lo = n;
        __syncthreads();
        // 每窗读游标进镜像 + 原槽复位 INT_MAX(Ocean 原版 reset-to-∞ 语义):否则旧游标恒小于
        // 新 break 值 → atomicMin 永不更新 → 游标不前进 → 下窗重读旧位置 → col<win_lo → j<0
        // → SMEM 下越界 → illegal instruction(今晚实锤的坑,docs/30)
        for (int p = rs + tid; p < re; p += blockDim.x) {
            int v = smap[p - rs];
            smir[p - rs] = v;
            smap[p - rs] = 0x7fffffff;
        }
        __syncthreads();
        int my_next = n;
        if (PB2_STAGE >= 2) {
            for (int p = rs + my_group; p < re; p += num_groups) {
                int st = smir[p - rs];
                if (st == 0x7fffffff) continue;   // 哨兵:该 k 已耗尽(INT_MAX+my_id 会溢出为负 → 全局越界)
                int k = A_col_idx[p];
                double a = A_val[p];
                int b_end = B_row_ptr[k + 1];
                if (batch_on) {
                    // spec89-#1:批量装载(2 发在飞)——long_scoreboard 11.78 延迟瓶颈下翻倍每线程
                    // 在飞载数(spec86-B 定谳:冷失载不可预取,唯一杠杆=并行暴露)
                    for (int q = st + my_id; q < b_end; q += 2 * G) {
                        int jcol, jcol2 = -1; double bv, bv2 = 0.0;
                        if (B16p) { uint4 e = B16p[q]; jcol = (int)e.x; bv = __hiloint2double(e.w, e.z);
                                    if (q + G < b_end) { uint4 e2 = B16p[q + G]; jcol2 = (int)e2.x; bv2 = __hiloint2double(e2.w, e2.z); } }
                        else { jcol = B_col_idx[q]; bv = B_val[q];
                               if (q + G < b_end) { jcol2 = B_col_idx[q + G]; bv2 = B_val[q + G]; } }
                        int j = jcol - win_lo;
                        if (j >= w) { atomicMin(&smap[p - rs], q); if (jcol < my_next) my_next = jcol; break; }
                        int col = win_lo + j;
                        if (!(upper_tri && col < i)) { if (k1fw) K1_ACC(dval, j, dflag, a * bv); else { atomicAdd(&dval[j], a * bv); dflag[j] = 1; } }
                        if (jcol2 >= 0) {
                            int j2 = jcol2 - win_lo;
                            if (j2 >= w) { atomicMin(&smap[p - rs], q + G); if (jcol2 < my_next) my_next = jcol2; break; }
                            int col2 = win_lo + j2;
                            if (!(upper_tri && col2 < i)) { if (k1fw) K1_ACC(dval, j2, dflag, a * bv2); else { atomicAdd(&dval[j2], a * bv2); dflag[j2] = 1; } }
                        }
                    }
                } else
                for (int q = st + my_id; q < b_end; q += G) {
                    int jcol; double bv;                    // spec86-A:B16 单股流(cursor 主路径)
                    if (B16p) { uint4 e = B16p[q]; jcol = (int)e.x; bv = __hiloint2double(e.w, e.z); }
                    else { jcol = B_col_idx[q]; bv = B_val[q]; }
                    int j = jcol - win_lo;
                    if (j >= w) {
                        atomicMin(&smap[p - rs], q);
                        if (jcol < my_next) my_next = jcol;
                        break;
                    }
                    int col = win_lo + j;
                    if (upper_tri && col < i) continue;
                    if (k1fw) K1_ACC(dval, j, dflag, a * bv);
                    else { atomicAdd(&dval[j], a * bv); dflag[j] = 1; }
                }
            }
            if (my_next < n) atomicMin(&s_next_lo, my_next);
        }
        __syncthreads();
        // spec86-B:书签预取(最简形)——窗尾 smap 已是下窗各链起点;此处 touch 进 L2,
        // 后续 prefix/extract 段计算掩盖装载延迟(col+val;B16 在则一发 uint4)
        if (pf_on) {
            unsigned pf_acc = 0;
            for (int p = rs + tid; p < re; p += blockDim.x) {
                int s2 = smap[p - rs];
                if (s2 != 0x7fffffff) {
                    int k2 = A_col_idx[p];
                    if (s2 < B_row_ptr[k2 + 1]) {
                        if (B16p) { uint4 e = B16p[s2]; pf_acc += (unsigned)e.x + (unsigned)e.z; }
                        else pf_acc += (unsigned)B_col_idx[s2] + (unsigned)__double2hiint(B_val[s2]);
                    }
                }
            }
            s_pf_sink += (int)pf_acc;
        }
        if (tid == 0 && i == g_dflag_probe_row) {   // 探针v4:stage3 入口全窗 flag 计数(判"晚写" vs "prefix 读零")
            int fl3 = 0; for (int j = 0; j < w; j++) fl3 += dflag[j];
            printf("[D3IN ] row%d win[%d,%d) 全窗flags=%d/%d\n", i, win_lo, win_lo + w, fl3, w);
        }
        if (PB2_STAGE >= 3) {
            // docs/46 全 warp 并行前缀(原 warp0 串行 169 elem/lane → 每 warp 只 w/nw 个)
            int nwarp = blockDim.x >> 5;
            int seglen = (w + nwarp * 32 - 1) / (nwarp * 32);
            int t0 = (warp * 32 + lane) * seglen;
            int t1 = min(t0 + seglen, w);
            int run = 0;
            for (int j = t0; j < t1; j++) { dpref[j] = run; run += dflag[j]; }
            int excl = run;
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up_sync(0xffffffff, excl, off);
                if (lane >= off) excl += v;
            }
            __shared__ int wsum46[32];
            if (lane == 31) wsum46[warp] = excl;
            __syncthreads();
            if (tid == 0) { int acc = 0; for (int x = 0; x < nwarp; x++) { int t = wsum46[x]; wsum46[x] = acc; acc += t; } }
            __syncthreads();
            int woff46 = wsum46[warp];
            for (int j = t0; j < t1; j++) dpref[j] += woff46 + (excl - run);   // ⚠ 第三遍(warp 内低 lane 偏移)
                // 首版只加 woff46(warp 间):warp 内各 lane 前缀互相重叠 → rank 撞车 → diter 全量值雷
                // (msc 3261 垃圾行/420 flag 挤 214 槽,study/18;search 路 1119-1130 同款有此遍 = 对照)
        }
        __syncthreads();
        if (PB2_STAGE >= 4) {
            for (int j = tid; j < w; j += blockDim.x) {
                if (dflag[j]) {
                    int rank = out + dpref[j];
                    if (rank >= cap) { if (drop_cnt) atomicAdd(drop_cnt, 1); continue; }   /* #4'-fix:不许无痕丢 */
                    if (tmp_col) { tmp_col[base + rank] = win_lo + j; tmp_val[base + rank] = dval[j]; }
                    else { dC_ci[base + rank] = win_lo + j; dC_val[base + rank] = dval[j]; }
                }
            }
            out += dpref[w - 1] + dflag[w - 1];
        }
        if (tid == 0 && i == g_dflag_probe_row) {   // DFLAG_PROBE 探针(diter 雷判别:逐窗 flag 数与 prefix 尾)
            int fl = 0; for (int j = 0; j < w; j++) fl += dflag[j];
            printf("[DFLAG] row%d win[%d,%d) flags=%d pref_end=%d dpref[0..15]=",
                   i, win_lo, win_lo + w, fl, dpref[w-1] + dflag[w-1]);
            for (int j = 0; j < 16 && j < w; j++) printf("%d ", dpref[j]);
            printf("\n");
        }
        int next = s_next_lo;
        __syncthreads();
        win_lo = next;
    }
    if (tid == 0) {
        row_nnz[i] = out;   /* #4': tmp 模式需无条件写(count 已免,陈旧值会毒化晚 cnnz) */
        if (rowtime) rowtime[i] = clock64() - rt0;       // cursor 路径出口
    }
}

// dense 行集合的 Σest / Σrow_nnz / Σflop(方案5:tmp 缩容 + est 下溢检查改 hash 侧口径 + dup 门)
// sum 用 unsigned ll 原子(sm_90 无 signed ll atomicAdd 重载;值为非负,位型等同)
static __global__ void dense_sum_kernel(const int *rows, int nr, const int *est, const int *row_nnz,
                                 const int *flop /*nullable:方案5 dup 门 pre-pass*/,
                                 unsigned long long *sum_est, unsigned long long *sum_nnz) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    if (est) atomicAdd(sum_est, (unsigned long long)est[rows ? rows[r] : r]);
    if (row_nnz) atomicAdd(sum_nnz, (unsigned long long)row_nnz[rows ? rows[r] : r]);
    if (flop) atomicAdd(sum_nnz, (unsigned long long)flop[rows ? rows[r] : r]);   // 门 pre-pass 复用 sum_nnz 槽
}

// dense 行 est 置 0(方案5:d_off 重扫前坍缩 gapped 空间;binning 已完成,est 无后续消费者)
static __global__ void zero_est_kernel(int *est, const int *rows, int nr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr) est[rows[r]] = 0;
}

// DCEST V3(study/39 Phase 1b):est 分配(去 expand 泄漏)+ 溢出证书 + 新 buffer 紧凑拷贝
// V2 定谳(a81dc1b/df6ccb6 复盘):①数值核行内【连续直写】(base 起 bitmap/cursor 前缀)
//   且自报精确 row_nnz → fixup 核扫描区间坍缩为实际数据 = no-op 白付 6.8ms,sentinel
//   从未 launch 而值门通过本身就是"尾洞不存在"的证明 → 两者全删;
// ②in-place 全局 shift 有竞态:行内左移 src/dst 重叠跨线程乱序 + 跨行 B 的新区可踩
//   A 未读的旧尾(并行 block 无序,c-64 门通过属调度运气)→ 改读写分离双 buffer 拷贝。

// ① 填 row_nnz = dense 行分配额。est 出口语义(mh_merge):min(raw_E × expand, flop)
//   → flop 封顶行(est==flop,dup≈1 人口)是【精确上界零低估】,直接 ×margin(若除
//   expand 会把上界砍成 0.56×flop 的欠估 → dup≈1 行必溢出);MinHash 行才做 ÷expand
//   还原始估计。V2 的 est×1.2 隐含 raw×2.16(1.8 泄漏)→ V3 = raw×margin。
static __global__ void dcest_fill_kernel(int *row_nnz, const int *est, const int *flop, const int *rows,
                                   int nr, int expand_pct, int margin_pct) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr) {
        int i = rows[r];
        int e = est[i], fl = flop[i];
        long long v = (fl > 0 && e >= fl)
                    ? (long long)e * margin_pct / 100        // flop 封顶 = 精确上界,免还原
                    : (long long)e * margin_pct / expand_pct; // MinHash:去 expand 还原始 E
        row_nnz[i] = (int)(v > 1 ? v : 1);
    }
}

// ② 溢出证书(L2,study/39 3c):数值核自报 actual(row_nnz)> scan-1 槽宽(alloc)
//   → 分配不足。梯子核写位无界已踩邻行(数据损),唯一安全出口 = 判废(-1)回退;
//   cursor 核 rank≥cap 截断(drop_cnt 同语义)。附带 alloc/actual 比值直方图 —— margin
//   选择的实测依据(桶界 = ×8 后 2/4/6/8/10/12/16 → ratio <0.25/<0.5/<0.75/<1.0/<1.25/<1.5/<2/≥2)。
//   ovf_dbg[5*8](取证探针 10-13):前 8 个溢出行各 {row,alloc,act,flop,a_len}。
static __global__ void dcest_ovf_check_kernel(const int *dense_rows, int nr,
                                        const int *dC_rp, const int *row_nnz,
                                        const int *row_flop, const int *A_rp,
                                        int *n_ovf, int *hist8, int *ovf_dbg) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    int i = dense_rows[r];
    int slot = dC_rp[i + 1] - dC_rp[i];   // scan-1 槽宽 = alloc
    int act  = row_nnz[i];                 // 数值核自报实际
    if (act <= 0) return;                  // 空行无风险不入桶
    if (act > slot) {
        int k = atomicAdd(n_ovf, 1);       // atomicAdd 返回旧值 = 判废计数不受取证影响
        if (k < 8) {                       // 取证:溢出行结构(flop/a_len 判估计器盲区形态)
            atomicExch(&ovf_dbg[5 * k], i);
            atomicExch(&ovf_dbg[5 * k + 1], slot);
            atomicExch(&ovf_dbg[5 * k + 2], act);
            atomicExch(&ovf_dbg[5 * k + 3], row_flop ? row_flop[i] : 0);
            atomicExch(&ovf_dbg[5 * k + 4], A_rp ? A_rp[i + 1] - A_rp[i] : 0);
        }
    }
    int r8 = (int)((long long)slot * 8 / act);
    int b = r8 < 2 ? 0 : r8 < 4 ? 1 : r8 < 6 ? 2 : r8 < 8 ? 3 : r8 < 10 ? 4 : r8 < 12 ? 5 : r8 < 16 ? 6 : 7;
    atomicAdd(&hist8[b], 1);
}

// ③ 紧凑拷贝:dense 行 gapped 旧布局 → exact 新布局。写全部落新 buffer(与所有读源
//   不相交,V2 in-place shift 的行内/跨行两类竞态结构性根除)。链位置保证此刻 dC 里
//   只有 dense 行有数据(k7/D5H 在链后写 exact 基址)→ 只搬 dense 行,不动行早退。
static __global__ void dcest_compact_copy_kernel(const int *rows, int nr,
                                          const int *rp_old, const int *rp_new,
                                          const int *row_nnz,
                                          const int *ci_src, const double *v_src,
                                          int *ci_dst, double *v_dst) {
    int ri = blockIdx.x;
    if (ri >= nr) return;
    int row = rows[ri];
    int src = rp_old[row], dst = rp_new[row], cnt = row_nnz[row];
    // 注意:src==dst 的行【不能】早退 —— 数据在旧 buffer 里,新 buffer 同位置仍是零页
    // (值门指纹:row5 整行 (5,1,0) = 未拷贝的零页;V2 in-place shift 语义在此不成立)。
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
        ci_dst[dst + j] = ci_src[src + j];
        v_dst[dst + j]  = v_src[src + j];
    }
}

// ===================== sub-warp 批量 kernel(2026-08-26 方向B,巨型图第二程) =====================
// est ≤ 8 的极稀疏行(germany_osm/road 类 avg_product≈5):每行只派 8 线程(4 行/warp),
// 去重累加在**寄存器**里(≤8 个 col/val 对,零 SMEM 零原子),warp shfl 去重+求和。
// 对标 Ocean hashNumericSubWarpKernel<4>(它 4 线程/行,hash 表在 SMEM;我们寄存器更轻)。
#define SUBW_HT SUBW_HT_STRUCT  // warp_size/4

// 干净版:8 线程/行,SMEM 小表(32 slot),warp 内 4 行不共享表
static __global__ void hash_subwarp8_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int lane = threadIdx.x & 31;
    int row_slot = lane >> 3;                    // 0-3
    int sub_lane = lane & 7;                     // 0-7
    int rows_per_warp = 32 >> 3;                 // 4
    int row_idx = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * rows_per_warp + row_slot;
    if (row_idx >= n_in_bucket) return;
    int i = bucket_rows[row_idx];

    // SMEM 表:每 warp 4 行 × 32 slot × 12B = 1.5KB/warp,256 线程 8 warp = 12KB
    __shared__ int s_col[8][4][32];              // [warp][row_slot][slot]
    __shared__ double s_val[8][4][32];
    int warp = threadIdx.x >> 5;
    int *my_col = s_col[warp][row_slot];
    double *my_val = s_val[warp][row_slot];
    for (int s = sub_lane; s < 32; s += 8) { my_col[s] = -1; my_val[s] = 0.0; }
    __syncwarp(0xFFu << (row_slot * 8));         // 同行 8 线程同步(warp 内子组)

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + sub_lane; q < ke; q += 8) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & 31;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&my_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&my_val[slot], v); break; }
                slot = (slot + 1) & 31;
                if (++probes >= 32) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                    break;
                }
            }
        }
    }
    __syncwarp(0xFFu << (row_slot * 8));

    // 提取 + 插入排序(单线程,8 线程中 sub_lane==0 做)
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    if (sub_lane == 0) {
        int cols[32]; double vals[32]; int cnt = 0;
        for (int s = 0; s < 32; s++) {
            if (my_col[s] >= 0) { cols[cnt] = my_col[s]; vals[cnt] = my_val[s]; cnt++; }
        }
        // 插入排序
        for (int t = 1; t < cnt; t++) {
            int c = cols[t]; double v = vals[t];
            int s2 = t - 1;
            while (s2 >= 0 && cols[s2] > c) { cols[s2+1] = cols[s2]; vals[s2+1] = vals[s2]; s2--; }
            cols[s2+1] = c; vals[s2+1] = v;
        }
        if (cnt > cap) {
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
            cnt = (int)cap;
        }
        for (int t = 0; t < cnt; t++) {
            tmp_key[base + t] = ((unsigned long long)i << 32) | (unsigned int)cols[t];
            tmp_val[base + t] = vals[t];
        }
        row_nnz[i] = cnt;
    }
}


#define BATCH_HT BATCH_HT_STRUCT  // α=0.5 负载因子推导
#define BATCH_WPB BATCH_WPB_STRUCT  // hash_block/warp_size(硬件)
// ===================== 方案1: 预排序二分寻址 (HSMU HPCA'25 思路) =====================
// 两阶段: symbolic 收集列集合并排序 -> numeric 二分定位 + 直接 atomicAdd
// 零 CAS(位置由二分确定), 零冲突(列集合唯一), 输出天然有序

static __global__ void bsearch_symbolic_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    __shared__ int bsmem[BATCH_WPB][5 * BATCH_HT];   // t_col | t_val(2HT int) | pack_col | pack_slot(分离,同 batched 修)
    __shared__ int w_cnt[BATCH_WPB];
    int *t_col = bsmem[warp & (BATCH_WPB - 1)];
    double *t_val = (double*)(bsmem[warp & (BATCH_WPB - 1)] + BATCH_HT);
    int *pack_col = bsmem[warp & (BATCH_WPB - 1)] + 3 * BATCH_HT;
    int *pack_slot = bsmem[warp & (BATCH_WPB - 1)] + 4 * BATCH_HT;

    if (lane == 0) w_cnt[warp & (BATCH_WPB - 1)] = 0;
    for (int s = lane; s < BATCH_HT; s += 32) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            unsigned slot = ((unsigned)(j * 2654435761u)) & (BATCH_HT - 1);
            int probes = 0;
            while (true) {
                int old = atomicCAS(&t_col[slot], -1, j);
                if (old == -1 || old == j) break;
                slot = (slot + 1) & (BATCH_HT - 1);
                if (++probes >= BATCH_HT) { atomicExch(overflow_flag, 1); break; }
            }
        }
    }
    __syncwarp();

    __shared__ int w_cnt2[BATCH_WPB];
    if (lane == 0) w_cnt2[warp & (BATCH_WPB - 1)] = 0;
    __syncwarp();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = lane; s < BATCH_HT; s += 32) {
        int c = t_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&w_cnt2[warp & (BATCH_WPB - 1)], 1);
            if (pos < BATCH_HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = w_cnt2[warp & (BATCH_WPB - 1)];
    if (n > BATCH_HT) n = BATCH_HT;
    for (int idx = lane; idx < n; idx += 32) {
        int c = pack_col[idx];
        int rank = 0;
        for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
        if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
        tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
        tmp_val[base + rank] = 0.0;
    }
    __syncwarp();
    if (lane == 0) {
        if (n > cap && !atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
        row_nnz[i] = (n <= cap) ? n : cap;
    }
}

static __global__ void bsearch_numeric_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *row_nnz,
    unsigned long long *tmp_key, double *tmp_val)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    long long base = row_off[i];
    int nnz = row_nnz[i];

    __shared__ int scol[BATCH_WPB][BATCH_HT];
    for (int s = lane; s < nnz; s += 32) {
        scol[warp & (BATCH_WPB - 1)][s] = (int)(tmp_key[base + s] & 0xffffffffu);
    }
    __syncwarp();
    int *cols = scol[warp & (BATCH_WPB - 1)];

    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            int lo = 0, hi = nnz - 1, pos = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (cols[mid] == j) { pos = mid; break; }
                else if (cols[mid] < j) lo = mid + 1;
                else hi = mid - 1;
            }
            if (pos >= 0) atomicAdd(&tmp_val[base + pos], v);
        }
    }
}

// ===================== 小行批量 kernel(2026-08-26,Phase 2 主杠杆) =====================
// est ≤ 64 的行(bin0-1):warp-per-row + SMEM warp 私有表(128 槽,2× 余量)+ 融合有序 extract。
// 治"1 CTA/行×256 线程伺候 ~6 个积"的每行固定开销(333SP 型 accumulate 55% 差距源);
// 有序直写 tmp → compact 阶段走 copy 免排序。设计:`inno/hash_batched_kernel_design.md`。

// docs/67 §8 模板化:<128,8>=原 bin0(est≤64);<512,4>=BIN_BATCH2(est≤256,消 web 图类
// 55 万小行的 22k cyc/行块地板税 —— est 66-256 行独占 64-512T 块只装 ~384 乘积是浪费)。
template<int HT, int WPB>
__global__ void hash_spa_batched_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    double *global_val = nullptr)    // HYBRID=1: 全局 value 池(L2 原子,吞吐 > SMEM)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    // 静态 SMEM:每 warp 段 = col[HT] | val(2×HT int = HT double) | pack_col[HT] | pack_slot[HT]
    // ⚠ 旧布局 4×HT 把 pack_col 放 +2HT —— 落在 t_val(double 占 [HT,3HT) int)背衬区:extract 写
    //   pack_col[pos] 半改写高槽位(≥HT/2)条目的 val → 整型比特 denormal 垃圾(study/18 雷 #3-b,
    //   pre2 696 点/bit=两列号;est≤64 且哈希槽≥HT/2 的行才中招,三周未察)。改 5×HT 分离。
    __shared__ int bsmem[WPB][5 * HT];
    __shared__ int w_cnt[WPB];
    int *t_col = bsmem[warp & (WPB - 1)];
    // 方案4修正: HYBRID 模式下 value 在全局(L2 原子吞吐高,CAS 仍在 SMEM)
    double *t_val;
    if (global_val) {
        t_val = global_val + (size_t)(warp) * HT;  // 全局池,每 warp 一段
    } else {
        t_val = (double*)(bsmem[warp & (WPB - 1)] + HT);   // int [HT, 3HT)
    }
    int *pack_col = bsmem[warp & (WPB - 1)] + 3 * HT;      // int [3HT, 4HT)
    int *pack_slot = bsmem[warp & (WPB - 1)] + 4 * HT;     // int [4HT, 5HT)

    if (lane == 0) w_cnt[warp & (WPB - 1)] = 0;
    for (int s = lane; s < HT; s += 32) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {                     // k 串行(小行 k 少)
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {      // j 32 路并行
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & (HT - 1);
            int probes = 0;                             // 有界探测:必终止
            while (true) {
                int old = atomicCAS(&t_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&t_val[slot], v); break; }
                slot = (slot + 1) & (HT - 1);
                if (++probes >= HT) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncwarp();

    // extract:pack 非空槽 → count-rank 排序 → 有序写 tmp(rank 是 0..n-1 置换)
    long long base = row_off[i];
    int cap = est[i];                                   // SAFETY:mh 低估时 n 可超 est → 守卫
    for (int s = lane; s < HT; s += 32) {
        int c = t_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&w_cnt[warp & (WPB - 1)], 1);
            if (pos < HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = w_cnt[warp & (WPB - 1)];
    if (n > HT) n = HT;                     // pack 溢出截断(此时必已置 overflow)
    for (int idx = lane; idx < n; idx += 32) {
        int c = pack_col[idx];
        int rank = 0;
        for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
        if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
        tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
        tmp_val[base + rank] = t_val[pack_slot[idx]];
    }
    __syncwarp();
    if (lane == 0) {
        if (n > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
        row_nnz[i] = (n <= cap) ? n : cap;
    }
}

// S1 死码剔除验证(临时):multirow 核整体 #if 0
#if 0
// ══ batched 战役刀(study/59):hash_spa_batched_multirow —— 微行巨阵人口几何修正 ══
// 机理定谳(§batched 战役·机理定谳):warp-per-row 的 j 32 路固定并行在 B 行 ~7 人口
// = lane 利用率 22%(ocean 动态分组自适应);本核 = 每 warp RPW 行,32/RPW-lane 组
// (RPW=2→16 lane;RPW=4→8 lane 配 HT=64),组私有表+组内 k 串行 × j 组内并行,
// extract/rank/溢出语义与 batched 同款。分歧规避:循环一律 warp 统一界(HT),组守卫内置。
template<int HT, int RPW>
__global__ void hash_spa_batched_multirow_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    constexpr int LG = 32 / RPW;                 // 组内 lane 数
    constexpr int WPB = 8;                       // 与原 batched 同块几何(256T)
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if ((long long)warp * RPW >= n_in_bucket) return;   // warp 首行即越界(warp 统一退出)
    const int g = lane / LG, gl = lane % LG;     // 组 id / 组内 lane
    const int ridx = warp * RPW + g;
    const int i = (ridx < n_in_bucket) ? bucket_rows[ridx] : -1;

    __shared__ int bsmem[WPB][RPW][5 * HT];
    __shared__ int g_cnt[WPB][RPW];
    int *t_col = bsmem[warp & (WPB - 1)][g];
    double *t_val = (double *)(bsmem[warp & (WPB - 1)][g] + HT);
    int *pack_col = bsmem[warp & (WPB - 1)][g] + 3 * HT;
    int *pack_slot = bsmem[warp & (WPB - 1)][g] + 4 * HT;

    if (gl == 0) g_cnt[warp & (WPB - 1)][g] = 0;
    for (int s = gl; s < HT; s += LG) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();
    if (i >= 0) {
        int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
        for (int p = rs; p < re; p++) {              // k 组内串行
            int k = A_col_idx[p];
            double a = A_val[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            for (int q = ks + gl; q < ke; q += LG) { // j 组内并行
                int j = B_col_idx[q];
                if (upper_tri && j < i) continue;
                double v = a * B_val[q];
                unsigned slot = ((unsigned)(j * 2654435761u)) & (HT - 1);
                int probes = 0;
                while (true) {
                    int old = atomicCAS(&t_col[slot], -1, j);
                    if (old == -1 || old == j) { atomicAdd(&t_val[slot], v); break; }
                    slot = (slot + 1) & (HT - 1);
                    if (++probes >= HT) {
                        atomicExch(overflow_flag, 1);
                        if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                        break;
                    }
                }
            }
        }
    }
    __syncwarp();
    // extract(组内 count-rank;循环统一界 HT,组守卫内置防分歧 sync)
    long long base = i >= 0 ? row_off[i] : 0;
    int cap = i >= 0 ? est[i] : 0;
    for (int s = gl; s < HT; s += LG) {
        int c = i >= 0 ? t_col[s] : -1;
        if (c >= 0) {
            int pos = atomicAdd(&g_cnt[warp & (WPB - 1)][g], 1);
            if (pos < HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = g_cnt[warp & (WPB - 1)][g];
    if (n > HT) n = HT;
    for (int idx = gl; idx < HT; idx += LG) {         // 统一界(HT),idx<n 组守卫
        if (idx < n && i >= 0) {
            int c = pack_col[idx];
            int rank = 0;
            for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
            if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = t_val[pack_slot[idx]];
        }
    }
    __syncwarp();
    if (gl == 0 && i >= 0) {
        if (n > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; } }
        row_nnz[i] = (n <= cap) ? n : cap;
    }
}


#endif
// ══ 刀C3(study/57 §十七):hash_spa_packed —— bin4 多行封包内核 ════════════════
// 病灶:bin4-5 并发逆差 2.85×(tick7 逐行对拍:我方周期 0.53× 赢,系统 1.51× 输);
// 刀C2 证伪块宽杠杆(共享表组数减单调恶化);本核 = C2 未测设计点:
//   warp 私有表(无跨行争用)+ k 串行 × j 32 路 + 屏障摊销 —— ocean <7,673> 对位形态。
// 保留我方资产:read-first 探测(逐行赢面的探测半)+ 溢出熔断全套(probes≥HT 早退
// → row_ovf → retry 换 flop 表,est 欠估人群保命机制,铁律不剥)。
// extract:表内 bitonic 全序(HT=2^n,-1 哨兵升序沉头部)—— 零额外 smem、免 pack 数组、
// 免 O(n²) rank(n≈300 时 2813 seq-op 判死)与 CS2 直方图 3KB hist;45 趟 × 16 idx/lane。
// smem 静态 24KB/块(4 warp × [512 col | 512 val])→ 9 块/SM = 36 行在飞(现 8,4.5×)。
// 路由:bin4 整桶(env HASH_B4PACK,默认关);est>448 高 LF 行自然走 overflow→retry
// = 现路径逃生。tmp 契约与 batched 同:row_off+rank 有序写,compact 走 copy。
template<int HT>
__global__ __launch_bounds__(128) void hash_spa_packed_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    constexpr int WPB = 4;                       // 128T = 4 warp,每 warp 一行
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];
    __shared__ int    s_col[WPB][HT];
    __shared__ double s_val[WPB][HT];
    int    *t_col = s_col[warp & (WPB - 1)];
    double *t_val = s_val[warp & (WPB - 1)];
    for (int s = lane; s < HT; s += 32) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {              // k 串行(ocean 形态;a_len p50=45 紧凑)
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {   // j 32 路并行
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & (HT - 1);
            int probes = 0;
            while (true) {                        // read-first:命中零 CAS(我方资产)
                int entry = t_col[slot];
                if (entry == j) { atomicAdd(&t_val[slot], v); break; }
                if (entry == -1) {
                    int prev = atomicCAS(&t_col[slot], -1, j);
                    if (prev == -1) { t_val[slot] = v; break; }
                    if (prev == j) { atomicAdd(&t_val[slot], v); break; }
                }
                slot = (slot + 1) & (HT - 1);
                if (++probes >= HT) {             // 熔断器全套(铁律不剥)
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                    break;
                }
            }
        }
    }
    __syncwarp();
    // 表内 bitonic 全序:-1 哨兵 < 任意 col → 沉头部;完美匹配每趟单交换者,免锁
    for (int ks = 2; ks <= HT; ks <<= 1)
        for (int j = ks >> 1; j > 0; j >>= 1) {
            for (int idx = lane; idx < HT; idx += 32) {
                int l = idx ^ j;
                if (l > idx) {
                    bool up = ((idx & ks) == 0);
                    int a = t_col[idx], b = t_col[l];
                    if (a != b && (a > b) == up) {
                        t_col[idx] = b; t_col[l] = a;
                        double va = t_val[idx], vb = t_val[l];
                        t_val[idx] = vb; t_val[l] = va;
                    }
                }
            }
            __syncwarp();
        }
    int cnt = 0;
    for (int s = lane; s < HT; s += 32) cnt += (t_col[s] >= 0);
    for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    int n = __shfl_sync(0xffffffffu, cnt, 0);
    long long base = row_off[i];
    int cap = est[i];
    if (lane == 0) {
        row_nnz[i] = (n <= cap) ? n : cap;
        if (n > cap) { atomicExch(overflow_flag, 1); if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; } }
    }
    for (int s = lane; s < HT; s += 32) {         // 有序写 tmp(-1 头部长度 HT-n)
        if (t_col[s] >= 0) {
            int rank = s - (HT - n);
            if (rank < cap) {
                tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned)t_col[s];
                tmp_val[base + rank] = t_val[s];
            }
        }
    }
}


// 按桶跑:ht_size = 该桶 hash 表大小;轻行小表、重行大表,distinct>ht_size → overflow_flag
// MODE:0=legacy(gapped tmp + packed key,compact 阶段善后)1=count-only(只插 key 出精确 row_nnz,
// 方案5 扩展到 hash 行,docs/31)2=direct(exact_rp 精确偏移直写 dC_ci/d_val,行内无序,原地 csort 善后)
// ④ 交织 B(inno/79):B 的 (col,val) 两窄流 → 16B 对齐 uint4 单条(col 4|pad 4|val 8);行结构/下标不变。
// 靶:B 流 L1 9%/sec-req 3.51/请求 2.5×(study/15 尸检);一次 LDG.128 携带双域,载入指令减半。
static __global__ void b_interleave_kernel(const int *col, const double *val, int n, uint4 *out) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    unsigned long long bits; double v = val[q]; memcpy(&bits, &v, 8);
    out[q] = make_uint4((unsigned)col[q], 0u,
                        (unsigned)(bits & 0xffffffffu), (unsigned)(bits >> 32));
}

// 探测体选择(逃生门):HASH_PROBE_RF 默认 1 = read-first;=0 回 CAS-first(study/15 前原始形态)。
// 全部发射点(主 bin / retry / count<1> / direct<2>)统一走它。
static inline bool probe_rf_on() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("HASH_PROBE_RF"); v = (e && *e) ? atoi(e) : 1; }
    return v != 0;
}

__device__ int g_xbit_d = 0;
__device__ int g_hh_sorted_d = 0;   // 刀Gv3(study/57 §五十):HH_SORTED → host 装载;hash_heavy 表内有序 rank 直写   // 刀E5(study/57 §三一):HASH_XBIT → host cudaMemcpyToSymbol;extract 每行一次分支,零热径成本

// 刀E1/E2/G(study/57 §廿五/廿六/四十九):bin 被无序写内核接管(hash_heavy 或 VG)→ compact 侧 csort 载体
static inline bool h69_route_on() {   // 刀①'(study/59):bin9 胖行→HH(16253)+有序直写+直拷(五阵 −27.9~−47.1% §刀①'定谳);默认 1;HASH_H69=0 逃生门
    static int v = -1;
    if (v < 0) { const char *e = getenv("HASH_H69"); v = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
    return v == 1;
}
static inline bool hh_sorted_host() {   // Gv3v2:compact 侧感知 HH_SORTED(有序输出免载体趟);与 TU-61 loader 同 env
    static int v = -1;
    if (v < 0) { const char *e = getenv("HH_SORTED"); v = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }   // 转正默认 1(与 TU-61 loader 同值)
    return v == 1;
}
static inline bool h45_pilot_on() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("HASH_H45");
        int e1 = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1;   // Gv3v2 转正默认 1
        const char *g = getenv("HASH_VG45");          // E2 需 HYB_TIER 同开(VG 门前置条件)
        const char *h = getenv("HYB_TIER");
        int e2 = (g && *g && atoi(g) > 0 && h && *h && atoi(h) > 0) ? 1 : 0;
        const char *g5 = getenv("HASH_H55");          // 刀G:bin5→hash_heavy;Gv3v2 转正默认 1
        int e3 = (g5 && *g5) ? (atoi(g5) > 0 ? 1 : 0) : 1;
        v = e1 || e2 || e3;
    }
    return v != 0;
}

// PROBE_RF(read-first 探测,模板参默认开):命中先普通读 sh_col,相等即 atomicAdd(val)——
// 命中路径零 CAS;空槽才 atomicCAS 占位(占败 = 他人已入他键,前进重探)。该探测模式为领域
// 公共形态(nsparse kernel_spgemm_hash_s.cu:298-315 / spECK HashMap.cuh 同构先例);散列用
// 自有 Knuth 乘法常数 2654435761u 全值域映射 + mask 线性探测(pow2 表)。
template<int MODE, int PROBE_RF = 1, int PROBE_HIST_ON = 0, int VAL_GMEM = 0, int LOG_HT = 0, int HT_P = 0, int PROBE2 = 0, int WM_T = -1, int TB_TAG = 0>
// PROBE2(刀D1,study/57 §十八):熔断换位——per-probe 计数器(IADD+ISETP 每探测步 +
// ++probes 归纳寄存器)→ per-k occupancy 余量检查(ocean hashSymbolic 同构:occ+本k
// B行宽 ≥ 表容 bail)+ wrap 检测(slot==起点,1 ISETP/步)作挂死保护。终止性:per-k
// 余量保证 k 扫描期间表必留空槽,read-first 探测 ≤ht 步内必遇 -1。T4b 教训对齐:
// 熔断语义保留(仅换粒度),非 T4B_PROBE 的裸删。occupancy 维护:CAS-win 时
// atomicAdd(s_occ)(每新 distinct 一次,~d/行,vs 每乘积旧形态零增量)。
// 默认 0 = 原形态逐位;HASH_PROBE2 env 控制发射点实例选择。
// PROBE_HIST_ON(spec91 ①:探测计数编译期化):1=仪器态实例(每探测环退出 atomicAdd 计数),
// 0=生产实例,热路径计数指令整体消除(此前是逐产品运行时 if(probe_hist) 分支)。
// 仪器态由 MODE=0 主发射点按 d_phist!=nullptr 分支实例化,其余发射点恒生产态。
// VAL_GMEM(spec91 刀E:值区几何,已判死保留代码):1=值区在 gmem 有界循环池块切片。
// LOG_HT(spec91 刀F 真形态):>0=表长编译期定死 1<<LOG_HT(ht_size 参数被影子覆盖)——
// mask/G 梯全部常量折叠(SASS 实证:运行时形态每乘积重算 IADD3 ht_size-1),并为后续
// 素数紧表(magic-div)与每档独立块几何(spec88 七档化合流)开路;0=运行时 ht_size(默认,
// bin0-6 甜点带 bit-for-bit 不变)。发射点仅 bins 7-9 实例化新形态(HASH_LHT=1 门)。
// __launch_bounds__(1024,2)(09-18,hash_spa 占用抬升实验):REG:40 下 1024thr 档只驻 1 块/SM(50%),
// 512thr 档 3 块(75%,crankseg 主发射 31,770 块实测 74.7%)。压 ≤32 regs → 1024 档 2 块/512 档 4 块。
// ⚠ 本 kernel 是三阵全胜主力,溢出风险高,须全套守门+电池;对照 = spgemm_lb2t。
#ifdef REGCAP
static __global__ void __launch_bounds__(HASH_BLOCK_STRUCT * 4) hash_spa_kernel(   /* REGCAP 试验:去 minBlocks(它覆盖 maxrregcount),硬帽经 NVCC -maxrregcount 传入 */
#else
static __global__ void __launch_bounds__(HASH_BLOCK_STRUCT * 4, 2) hash_spa_kernel(
#endif
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,   // 内层 j 来源:AA=A,ATT=Aᵀ(CSC)
    int upper_tri,                                                      // 0=自乘;1=ATT 只算 j≥i
    const int *bucket_rows, int n_in_bucket, int ht_size_rt,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *row_flop = nullptr,      // localLoadBalance 输入(行 flop);null=旧静态 G
    const int *row_maxbl = nullptr,     // 行 k 集内 B 行最大长;null=旧静态 G
    int llb = 0,                        // LLB=1:按行动态 G(docs/27 §4.2)
    const int *exact_rp = nullptr, int *dC_ci = nullptr, double *dC_val = nullptr,   // MODE=2
    unsigned long long *rowtime = nullptr,   // HASH_ROWTIME:逐行执行周期(docs/66;周期对 DVFS 免疫)
    const int *d5b_skip_est = nullptr,       // docs/70 §6②:MODE=1 跳过位图-合格行(交 hash_row_bcnt)
    const int *d5b_skip_span = nullptr,
    int force_logG = 0,                     // GFORCE(study/15 诊断):强制 G=2^force_logG;0=走正常梯
    int uo_extract = 0,                     // HASH_UO_XTRACT=1(口径对齐仪器):MODE=0 强制无序 extract,
                                            // 对齐 ocean ROWCYCLES 口径(彼方排序在计时窗外)。输出将无序,
                                            // 仅用于逐行对拍计时,勿入电池/默认。
    const uint4 *B16 = nullptr,             // HASH_BI(④ 交织 B):16B 单股流;null = 走双数组
    int warp_merge = 0,                     // HASH_WM(⑤ 合并 atomicAdd):同 warp 同列 lane match 认亲→
                                            // Σv 归约→leader 单探测+单原子。0=关(实验中)
    unsigned *probe_hist = nullptr,         // HASH_PROBE_HIST(①阶段链画像):探测环每退出一次计数
                                            // min(probes,32) 桶;null=生产零开销。probes=额外步数(0=首发命中)
    int rt_pre_extract = 0,                 // HASH_RT_NOEXTRACT=1:行计时截在探测循环尾(不含 extract),
                                            // 对齐 ocean ROWCYCLES 窗(用户钦定口径 A+ 修正)
    int extract2 = 0,                       // HASH_EXTRACT=2(spec 81):ht≥512 小行 extract 的 O(d²) count-sort
                                            // → 两级计数排序 O(d+1024);0 = legacy(默认)
    int vg_nbuf = 0,                        // spec91 刀E v2(VAL_GMEM=1):本 bin 循环缓冲数(池 = N_BUF×ht×8B;blockIdx%N_BUF 复用)
    double *vg_val = nullptr)               // spec91 刀E v2(VAL_GMEM=1):本 bin 值池基址;块切片 = vg_val + (blockIdx.x%vg_nbuf)×ht_size,块内自清
{
    const int ht_size = (HT_P > 0) ? HT_P                            // 刀F-主刀:素数紧表(magic-div 编译期取模)
                    : (LOG_HT > 0) ? (1 << LOG_HT) : ht_size_rt;    // LOG_HT>0 pow2 定长;=0 运行时(默认 bit-for-bit)
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];                  // 实际行号
    if (MODE == 1 && d5b_skip_est && (long long)BITMAP_DENSITY_DEN * d5b_skip_est[i] >= (long long)d5b_skip_span[i]
        && d5b_skip_span[i] <= BCNT_SPAN_MAX_BITS) return;   // 位图利用率 ≥ 1/8 ∧ 位图档装得下
    if (MODE == 2 && row_ovf[i]) return;       // count pass 溢出行已由 retry 处理(compact 末段拷)
    int tid = threadIdx.x;
    unsigned long long rt0 = rowtime ? clock64() : 0;   // 未处理行保持 0(标记区分)
    int mask = ht_size - 1;
    __shared__ int s_occ;   // 刀D1(PROBE2):occupancy 计数(per-k 熔断 + CAS-win 维护);非 PROBE2 实例 4B 死重

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                      // [ht_size]
    // spec91 刀E v2:VAL_GMEM=1 → 值区 = 有界循环缓冲池块切片(blockIdx%vg_nbuf 复用;并发块<N_BUF ⇒ 同
    // 缓冲两用者不共存),块内自清后栅栏(全块全局写可见性)。L2 驻留池 = RED 不落 DRAM(v1 全量池教训)。
    // ⚠ 累加必须双指针 + 编译期 if 特化:三元合一指针会把状态空间搅成 generic,ptxas 出 ATOMG(带返回
    // 阻塞)而非 REDG(fire-and-forget)——刀E 全部收益所在。extract 读走 val_rd(generic 读无妨,窗内无读)。
    double *sh_val = (double*)(smem + ((ht_size + 1) & ~1));                       // SMEM 形态;偶填充保 8B 对齐(素数奇表必需,pow2 不变)
    double *gv_arr = nullptr;                                                      // gmem 块切片(v2 循环缓冲)
    if (VAL_GMEM && vg_val) {
        gv_arr = vg_val + (long long)(blockIdx.x % vg_nbuf) * ht_size;
        for (int s = tid; s < ht_size; s += blockDim.x) gv_arr[s] = 0.0;          // 块内自清(替代 host memset)
    }
    double *val_rd = VAL_GMEM ? gv_arr : sh_val;                                   // extract/WM 读统一指针
#define SPA_VAL_ADD(SLOT_, V_) do { if (VAL_GMEM) atomicAdd(&gv_arr[SLOT_], V_); else atomicAdd(&sh_val[SLOT_], V_); } while (0)
    // ↑ spec91 刀E:VAL_GMEM=1 时 atomicAdd 落在可证全局指针上,ptxas 弃返回值 → RED.E.ADD.F64 一条
    for (int s = tid; s < ht_size; s += blockDim.x) { sh_col[s] = -1; if (!VAL_GMEM) sh_val[s] = 0.0f; }
    if (PROBE2 && tid == 0) s_occ = 0;
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    // G 选择:默认按 ht_size 静态梯(重行→G 大);LLB=1 改按行 (a_len, flop, max_b_len) 动态 2^k
    // (Ocean localLoadBalance 同款语义,治 Ge99 类 4.4× 的负载不均;G 可到 HASH_BLOCK=整 block/k)
    int logG;
    if (force_logG)
        logG = min(force_logG, 31 - __clz((unsigned)blockDim.x));   // GFORCE 诊断:可超 WARP(跨 warp 协作一个 k)
    else if (llb && row_flop && row_maxbl)
        logG = local_load_balance(re - rs, row_flop[i], row_maxbl[i], 2 /*G_min=4*/, 31 - __clz((unsigned)blockDim.x));
    else
        logG = hash_logG(ht_size);  // 表大小梯:表每 ×8 槽 G ×2(G ≤ WARP_SIZE 免跨 warp 同步)
    int G = 1 << logG;
    int num_groups = blockDim.x / G;
    int my_group = tid / G, my_id = tid % G;
    const int wm_eff = (WM_T >= 0) ? WM_T : warp_merge;   // 刀E3:WM_T≥0 = 编译期定死(LHT 热实例 WM×3 死兄弟枚举环整体消码,REG 压力与调度空间同收);-1 = 运行时(legacy 发射点不变)
    if (wm_eff == 4) {
        // spec95 WM v4(§3.1s 三件套):①原生 q 环索引(消 WM2 走位平方税)②MATCH.ANY 单指令
        // dup 探针,无伙伴=原路径零附加③G=8 配套(HASH_WM=4+GFORCE=8):warp 跨 4 组×连续 k,
        // 网格阵 B 行重叠 → 同 j 在 warp 内可见。warp 均匀界=__reduce_max 趟数(免碎片化死锁)。
        int lane5 = tid & 31;
        for (int p = rs + my_group; p < re; p += num_groups) {
            int k = A_col_idx[p];
            double a_ik = A_val[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            int mine = (ke - ks - 1 - my_id) / G + 1; if (mine < 0) mine = 0;   // 本 lane 趟数
            int wmax = (int)__reduce_max_sync(0xffffffffu, (unsigned)mine);      // warp 均匀界
            for (int it = 0; it < wmax; it++) {
                int q = ks + my_id + it * G;
                bool act = q < ke;
                int j = -1; double v = 0.0;
                if (act) {
                    if (B16) { uint4 e = B16[q]; j = (int)e.x; v = a_ik * __hiloint2double(e.w, e.z); }
                    else { j = B_col_idx[q]; v = a_ik * B_val[q]; }
                    if (MODE == 1) v = 0.0;
                    if (upper_tri && j < i) j = -1;   // 滤后归哨组
                }
                unsigned mm = __match_any_sync(0xffffffffu, j);   // 1 指令探针
                if (j < 0) continue;                              // 哨组零成本
                if (mm == (1u << lane5)) {                        // 无同 j 伙伴:原路径
                    unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
                    int probes = 0;
                    while (true) {
                        int entry = sh_col[slot];
                        if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; }
                        if (entry == -1) {
                            int prev = atomicCAS(&sh_col[slot], -1, j);
                            if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; }
                        }
                        slot = (slot + 1) & mask;
                        if (++probes >= ht_size) {
                            atomicExch(overflow_flag, 1);
                            if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                            break;
                        }
                    }
                } else {                                          // 同 j 伙伴:Σv 归约 → leader 单探测
                    double sv = 0.0;
                    for (unsigned m2 = mm; m2; m2 &= m2 - 1) sv += __shfl_sync(mm, v, __ffs(m2) - 1);
                    if (lane5 == (int)(__ffs(mm) - 1)) {
                        unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
                        int probes = 0;
                        while (true) {
                            int entry = sh_col[slot];
                            if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                            if (entry == -1) {
                                int prev = atomicCAS(&sh_col[slot], -1, j);
                                if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                            }
                            slot = (slot + 1) & mask;
                            if (++probes >= ht_size) {
                                atomicExch(overflow_flag, 1);
                                if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                                break;
                            }
                        }
                    }
                }
            }
        }
    } else if (wm_eff == 2) {
        // ⑤ v3 WPM 窗口化乘积合并(inno/80 终局形态):按乘积序号切 32/窗,一 lane 一乘积。
        // uniformity 平凡(每 lane 恒有槽);同 j 乘积(块结构下 k 相邻→乘积序相邻)落同窗 →
        // match 归约 → 热 j 原子 26→1-3 次 = 时间串行窗口化消灭;B 读 = 每 k 段内连续。
        int lane5 = tid & 31, nw3 = blockDim.x >> 5, warp3 = tid >> 5;
        int total = row_flop ? row_flop[i] : 0;
        int nwin = (total + 31) >> 5;
        for (int wi = warp3; wi < nwin; wi += nw3) {          // warp 领窗(窗数 warp 级均匀)
            int need = wi * 32 + lane5;                        // 本 lane 的乘积序号
            int k = rs, q = B_row_ptr[k];
            bool act = (need < total);
            if (act)
                while (need > 0) {                             // 乘积序号 → (k,q) 走位(B_rp L1 缓存)
                    int len = B_row_ptr[k + 1] - q;
                    if (need < len) { q += need; need = 0; }
                    else { need -= len; k++; q = B_row_ptr[k]; }
                }
            int j = -1; double v = 0.0;
            if (act) {
                double a = A_val[k];
                if (B16) { uint4 e = B16[q]; j = (int)e.x; v = a * __hiloint2double(e.w, e.z); }
                else { j = B_col_idx[q]; v = a * B_val[q]; }
                if (MODE == 1) v = 0.0;
                if (upper_tri && j < i) j = -1;
            }
            unsigned mm = __match_any_sync(0xffffffffu, j);
            int gs = __popc(mm);
            int gs_eff = (act && j >= 0) ? gs : 0;
            int gnm = (int)__reduce_max_sync(0xffffffffu, (unsigned)gs_eff);
            double sv = 0.0;
            unsigned m2 = mm;                                   // v2.2 已证:__ffs 逐位 gather(__fns 有语义雷)
            for (int bi = 0; bi < gnm; bi++) {
                int bl = m2 ? ( __ffs(m2) - 1) : lane5;
                double pv = __shfl_sync(0xffffffffu, v, bl);
                if (m2) { sv += pv; m2 &= m2 - 1; }
            }
            if (act && j >= 0 && lane5 == __ffs(mm) - 1) {      // leader:单探测+单原子
                unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
                int probes = 0;
                while (true) {
                    int entry = sh_col[slot];
                    if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                    if (entry == -1) {
                        int prev = atomicCAS(&sh_col[slot], -1, j);
                        if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                    }
                    slot = (slot + 1) & mask;
                    if (++probes >= ht_size) {
                        atomicExch(overflow_flag, 1);
                        if (!atomicExch(&row_ovf[i], 1)) { int pr2 = atomicAdd(ovf_cnt, 1); ovf_rows[pr2] = i; }
                        break;
                    }
                }
            }
        }
    } else
    if (wm_eff) {
        // ⑤ v2(inno/80):双层 uniform 谓词化 + full-mask match 聚合 —— 同 warp 不同 k 的并发同列乘积
        // rank-butterfly 归约,leader 单探测+单原子(dup 高时原子/探测双坍缩)。
        // v1 死因 = activemask 碎片化;此版全程 0xffffffff,无同步边界漂移。
        int lane5 = tid & 31;
        int my_k_n = 0;
        for (int p0 = rs + my_group; p0 < re; p0 += num_groups) my_k_n++;
        int ksteps = __reduce_max_sync(0xffffffffu, (unsigned)my_k_n);
        for (int ki = 0; ki < ksteps; ki++) {
            int p = rs + my_group + ki * num_groups;
            bool kact = (p < re);
            int k = kact ? A_col_idx[p] : 0;
            double a = kact ? A_val[p] : 0.0;
            int q0 = 0, qe = 0;
            if (kact) { q0 = B_row_ptr[k] + my_id; qe = B_row_ptr[k + 1]; }
            int myn = (kact && qe > q0) ? (qe - q0 + G - 1) / G : 0;
            int jsteps = __reduce_max_sync(0xffffffffu, (unsigned)myn);
            for (int qi = 0; qi < jsteps; qi++) {
                int q = q0 + qi * G;
                bool act = kact && q < qe;
                int j = -1; double v = 0.0;
                if (act) {
                    if (B16) { uint4 e = B16[q]; j = (int)e.x; v = a * __hiloint2double(e.w, e.z); }
                    else { j = B_col_idx[q]; v = a * B_val[q]; }
                    if (MODE == 1) v = 0.0;
                    if (upper_tri && j < i) j = -1;   // 过滤项入无效群(群和废弃,不发射)
                }
                unsigned mm = __match_any_sync(0xffffffffu, j);
                int gs = __popc(mm);
                int gs_eff = (act && j >= 0) ? gs : 0;                          // -1 哨兵群不计(否则其巨尺寸
                int gnm = (int)__reduce_max_sync(0xffffffffu, (unsigned)gs_eff);   // 抬高全 warp 空转上界,v2.3)
                double sv = 0.0;
                unsigned m2 = mm;                                               // __ffs 逐位版(弃 __fns,消语义风险;
                for (int bi = 0; bi < gnm; bi++) {                              // shfl 恒无条件 = mask 恒全)
                    int bl = m2 ? ( __ffs(m2) - 1) : lane5;
                    double pv = __shfl_sync(0xffffffffu, v, bl);
                    if (m2) { sv += pv; m2 &= m2 - 1; }
                }
                if (act && j >= 0 && lane5 == __ffs(mm) - 1) {   // leader:单探测+单原子
                    unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
                    int probes = 0;
                    while (true) {
                        int entry = sh_col[slot];
                        if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                        if (entry == -1) {
                            int prev = atomicCAS(&sh_col[slot], -1, j);
                            if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, sv); break; }
                        }
                        slot = (slot + 1) & mask;
                        if (++probes >= ht_size) {
                            atomicExch(overflow_flag, 1);
                            if (!atomicExch(&row_ovf[i], 1)) { int pr2 = atomicAdd(ovf_cnt, 1); ovf_rows[pr2] = i; }
                            break;
                        }
                    }
                }
            }
        }
    } else {
    // ①阶段指令瘦身刀(09-18):UPTRI/B16 从逐乘积运行时分支提为模板参,4 路派生在循环外分发
    // (出处:ncu 人口公平对照 crankseg 我 3.66 vs 彼 2.54 inst/product,+44%;逐乘积分支 =
    // 每乘积 2-4 槽纯税。编译期消分支后 B16off/upper_tri=0 路径零分支。)
    #define HASH_INNER_LOOP(UPTRI_, B16ON_)                                                    \
    for (int p = rs + my_group; p < re; p += num_groups) {   /* 并行 k(stride num_groups) */   \
        int k = A_col_idx[p];                                                                 \
        double a_ik = A_val[p];                                                               \
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];        /* B 的行 k(AA=A;ATT=Aᵀ) */      \
        for (int q = ks + my_id; q < ke; q += G) {           /* 组内 G 线程并行 j */           \
            int j; double bv;                                                                 \
            if (B16ON_) { /* ④ 交织 B:一次 128b 携带 col+val */                               \
                uint4 e = B16[q];                                                             \
                j = (int)e.x; bv = __hiloint2double(e.w, e.z);                                \
            } else { j = B_col_idx[q]; bv = B_val[q]; }                                       \
            if (UPTRI_ && j < i) continue;                  /* ATT 上三角过滤(j≥i) */          \
            double v = (MODE == 1) ? 0.0 : a_ik * bv;        /* count-only 免乘免值写 */       \
            unsigned slot = (HT_P > 0) ? ((unsigned)(j * 2654435761u) % (unsigned)HT_P)      /* 素数 magic-div */ \
                                       : (((unsigned)(j * 2654435761u)) & mask);   /* Knuth 乘法散列 pow2 */ \
            if (PROBE2) {   /* 刀D1-v2':wrap 守卫替代计数器(v1 实测 s_occ 共享原子争用 +7.5%,已剥) */ \
                unsigned s0 = slot;                                                           \
                while (true) {                                                                \
                    int entry = sh_col[slot];                                                 \
                    if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; }           \
                    if (entry == -1) {                                                        \
                        int prev = atomicCAS(&sh_col[slot], -1, j);                           \
                        if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; } \
                    }                                                                         \
                    slot = (slot + 1) & mask;                                                 \
                    if (slot == s0) {   /* 绕整圈 = 表满,语义同旧熔断(ht 步界) */           \
                        atomicExch(overflow_flag, 1);                                         \
                        if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; } \
                        break;                                                                \
                    }                                                                         \
                }                                                                             \
            } else {                                                                          \
            int probes = 0;                                                                   \
            while (true) {                                                                    \
                if (PROBE_RF) {                                                               \
                    int entry = sh_col[slot];                                                 \
                    if (entry == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; }    \
                    if (entry == -1) {                                                        \
                        int prev = atomicCAS(&sh_col[slot], -1, j);                           \
                        if (prev == -1 || prev == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; } \
                    }                                                                         \
                } else {                                                                      \
                    int old = atomicCAS(&sh_col[slot], -1, j);                                \
                    if (old == -1 || old == j) { if (MODE != 1) SPA_VAL_ADD(slot, v); break; } \
                }                                                                             \
                slot = (HT_P > 0) ? ((slot + 1 >= (unsigned)HT_P) ? 0u : slot + 1) : ((slot + 1) & mask); \
                if (++probes >= ht_size) {                                                    \
                    atomicExch(overflow_flag, 1);                                             \
                    if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; } \
                    break;                                                                    \
                }                                                                             \
            }                                                                                 \
            if (PROBE_HIST_ON) atomicAdd(&probe_hist[probes < 32 ? probes : 32], 1);          \
            }                                                                                 \
        }                                                                                     \
    }
    if (upper_tri) { if (B16) { HASH_INNER_LOOP(1, 1) } else { HASH_INNER_LOOP(1, 0) } }
    else           { if (B16) { HASH_INNER_LOOP(0, 1) } else { HASH_INNER_LOOP(0, 0) } }
    #undef HASH_INNER_LOOP
    #undef SPA_VAL_ADD
    }
    __syncthreads();
    // HASH_RT_NOEXTRACT(口径 A+ 窗对齐,用户钦定 09-18):行计时在【探测循环完成处】截断,不含 extract/输出搬运
    // ——对齐 ocean ROWCYCLES 窗(彼方 numeric kernel 不含 epilogue 提取)。此前逐行对拍我方窗多付 extract
    // (crankseg 行约 5-15% 行时间),1.355 含此不对称成分;此仪器一开即剔除。
    if (rt_pre_extract && rowtime && tid == 0) { rowtime[i] = clock64() - rt0; }
    if (MODE == 1) {   // count-only:occupancy 即精确 distinct;溢出行已在上面标记(row_nnz 留给 retry)
        int cnt = 0;
        for (int s = tid; s < ht_size; s += blockDim.x) cnt += (sh_col[s] != -1);
        for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
        __shared__ int ccnt[32];
        if ((tid & 31) == 0) ccnt[tid >> 5] = cnt;
        __syncthreads();
        if (tid == 0) { int t = 0; for (int x = 0; x < (blockDim.x >> 5); x++) t += ccnt[x]; row_nnz[i] = t; }
        if (rowtime && tid == 0) rowtime[i] = clock64() - rt0;
        return;
    }

    // extract 去重项 → tmp:小行(ht≤CSORT_HT)SMEM 内 compact+count-sort 写有序,大行无序写(交 compact_sort)
    __shared__ int cnt;
    __shared__ int cs2_seg[64];   // spec 81 两级排序 scratch(64 = 32 warp × min/max 对;Pass2 段基址复用前 32)
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    if (ht_size <= CSORT_HT && !VAL_GMEM && !(uo_extract && MODE == 0)) {   // HASH_UO_XTRACT:口径对齐仪器;刀E2:VG 实例键区独占 smem,有序 extract 走无序分支(val_rd)
        // ── 刀E5(study/57 §三一):表内 bitonic extract(C3 已验 nnz 逐位的形态移植)──
        // 消:①atomicAdd 抢位压缩(每行 ~d 次串行共享原子)②CS2 五趟 ③rank O(d²) 回退。
        // 全表 bitonic(col/val 成对交换,-1 哨兵升序沉头)+ 有序直写;45 趟 × ht/256 交换/线程。
        if (g_xbit_d && MODE == 0) {
            __shared__ int xn; __shared__ int xw[32];
            int cntl = 0;
            for (int s = tid; s < ht_size; s += blockDim.x) cntl += (sh_col[s] != -1);
            for (int off = 16; off; off >>= 1) cntl += __shfl_down_sync(0xffffffffu, cntl, off);
            if ((tid & 31) == 0) xw[tid >> 5] = cntl;
            __syncthreads();
            if (tid == 0) { int t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += xw[w]; xn = t; }
            __syncthreads();
            const int n = xn;
            for (int ksz = 2; ksz <= ht_size; ksz <<= 1)
                for (int jj = ksz >> 1; jj > 0; jj >>= 1) {
                    for (int idx = tid; idx < ht_size; idx += blockDim.x) {
                        int l = idx ^ jj;
                        if (l > idx) {
                            bool up = ((idx & ksz) == 0);
                            int a = sh_col[idx], b = sh_col[l];
                            if (a != b && (a > b) == up) {
                                sh_col[idx] = b; sh_col[l] = a;
                                double va = sh_val[idx], vb = sh_val[l];
                                sh_val[idx] = vb; sh_val[l] = va;
                            }
                        }
                    }
                    __syncthreads();
                }
            long long base2 = row_off[i];
            long long cap2 = row_off[i + 1] - row_off[i];
            for (int s = tid; s < ht_size; s += blockDim.x) {
                int c = sh_col[s];
                if (c != -1) {
                    int rank = s - (ht_size - n);
                    if (rank < (int)cap2) {
                        tmp_key[base2 + rank] = ((unsigned long long)i << 32) | (unsigned)c;
                        tmp_val[base2 + rank] = sh_val[s];
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                row_nnz[i] = (n <= (int)cap2) ? n : (int)cap2;
                if (n > (int)cap2) { atomicExch(overflow_flag, 1); if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; } }
            }
        } else {
        // compact 到前部:chunked 读进寄存器→sync→写前部(read-before-write 防 race)
        for (int it = 0; it < (ht_size + blockDim.x - 1) / blockDim.x; it++) {
            int s = it * blockDim.x + tid;
            int c = -1; double v = 0.0f;
            if (s < ht_size) { c = sh_col[s]; v = sh_val[s]; }
            __syncthreads();
            if (c >= 0) { int pos = atomicAdd(&cnt, 1); sh_col[pos] = c; sh_val[pos] = v; }
            __syncthreads();
        }
        int count = cnt;
        // ── spec 81 两级计数排序(HASH_EXTRACT=2):O(d²) rank → O(d + 1024) ──
        // Pass0 行 min/max → 逐行自适应 SH(桶数 ≤1024,聚簇行不退化);Pass1 直方图;Pass2 warp 段扫
        // 排他前缀;Pass3 桶分组重排(寄存器 staging,读全后再写);Pass4(下方写出循环)桶内 O(b²)
        // 精确位次(b 实测 ≤2-4,退化上界 = legacy d²)。门槛 ht≥512;count>4×blockDim 回退 legacy。
        int *cs2_cnt = nullptr, *cs2_start = nullptr;
        int cs2_lo = 0, cs2_SH = 0;
        if (extract2 && ht_size >= CS2_MIN_HT / 2 && count > CS2_MIN_COUNT && count > 0
            && count <= CS2_STAGE * (int)blockDim.x) {
            int *h_cnt2  = smem + 3 * ht_size;               // [1024] 桶计数(sh_col ht×4B + sh_val
                                                            // ht×8B = 12×ht B 之后;2×ht 会砸 sh_val!)
            int *h_start = h_cnt2 + 1024;                    // [1024] 排他桶起点
            int *h_fill  = h_start + 1024;                   // [1024] 落位游标
            // Pass0:行 min/max(warp 归约 + cs2_seg 跨 warp 收集)
            int lo_l = 0x7fffffff, hi_l = -1;
            for (int k = tid; k < count; k += blockDim.x) {
                int c = sh_col[k];
                lo_l = min(lo_l, c); hi_l = max(hi_l, c);
            }
            for (int off = 16; off; off >>= 1) {
                lo_l = min(lo_l, __shfl_down_sync(0xffffffff, lo_l, off));
                hi_l = max(hi_l, __shfl_down_sync(0xffffffff, hi_l, off));
            }
            if ((tid & 31) == 0) { cs2_seg[(tid >> 5) << 1] = lo_l; cs2_seg[((tid >> 5) << 1) | 1] = hi_l; }
            __syncthreads();
            if (tid == 0) {
                int L = 0x7fffffff, H = -1;
                for (int w = 0; w < (int)(blockDim.x >> 5); w++) {
                    L = min(L, cs2_seg[w << 1]); H = max(H, cs2_seg[(w << 1) | 1]);
                }
                cs2_seg[0] = L; cs2_seg[1] = H;
            }
            __syncthreads();
            int lo = cs2_seg[0], hi = cs2_seg[1];
            // Pass1:直方图 + 游标清零(cs2_seg 下次写已在两重 sync 后,无竞态)
            unsigned span = (unsigned)(hi - lo + 1);
            int sbits = 32 - __clz(span - 1);                 // ceil_log2(span);span=1 → 0(__clz(0)=32)
            int SH = sbits > 10 ? sbits - 10 : 0;             // 桶数压到 ≤1024
            int nbins = (int)((span - 1) >> SH) + 1;
            for (int b = tid; b < nbins; b += blockDim.x) { h_cnt2[b] = 0; h_fill[b] = 0; }
            __syncthreads();
            for (int k = tid; k < count; k += blockDim.x)
                atomicAdd(&h_cnt2[(sh_col[k] - lo) >> SH], 1);
            __syncthreads();
            // Pass2:排他前缀(连续所有权:线程 t 持 [t·ib,(t+1)·ib);本地扫 → warp shfl 段扫 → 段基址串行)
            // ⚠ 必须连续持有:段基址公式假设"前面线程的全部 bin 排在我的之前",跨步持有时序不成立
            //   (首版跨步 → F2 85,963 有序违规,有序守门拦截)
            int ib = (nbins + (int)blockDim.x - 1) / (int)blockDim.x;
            int b_lo = tid * ib, b_hi = min(b_lo + ib, nbins);
            int run = 0;
            for (int b = b_lo; b < b_hi; b++) { run += h_cnt2[b]; h_start[b] = run; }
            int tot_t = run, exc_w = run;
            for (int off = 16; off; off >>= 1) {
                int y = __shfl_up_sync(0xffffffff, exc_w, off);
                if ((tid & 31) >= off) exc_w += y;
            }
            if ((tid & 31) == 31) cs2_seg[tid >> 5] = exc_w;   // warp 段和
            __syncthreads();
            if (tid == 0) {
                int r = 0;
                for (int w = 0; w < (int)(blockDim.x >> 5); w++) { int t = cs2_seg[w]; cs2_seg[w] = r; r += t; }
            }
            __syncthreads();
            int wb = cs2_seg[tid >> 5];
            for (int b = b_lo; b < b_hi; b++)
                h_start[b] += wb + exc_w - tot_t - h_cnt2[b];  // 本地 inclusive → 全局排他
            __syncthreads();
            // Pass3:桶分组重排(寄存器 staging;全部读定后再写,免读-写竞态)
            int rc[CS2_STAGE]; double rv[CS2_STAGE];
            #pragma unroll
            for (int r = 0; r < CS2_STAGE; r++) {
                int k = tid + r * (int)blockDim.x;
                if (k < count) { rc[r] = sh_col[k]; rv[r] = sh_val[k]; }
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < CS2_STAGE; r++) {
                int k = tid + r * (int)blockDim.x;
                if (k < count) {
                    int b = (rc[r] - lo) >> SH;
                    int pos = h_start[b] + atomicAdd(&h_fill[b], 1);
                    sh_col[pos] = rc[r]; sh_val[pos] = rv[r];
                }
            }
            __syncthreads();
            cs2_cnt = h_cnt2; cs2_start = h_start; cs2_lo = lo; cs2_SH = SH;
        }
        if (MODE == 2) {   // 精确偏移直写(有序,免 compact;count pass 已保证无溢出)
            int base2 = exact_rp[i];
            if (cs2_cnt) {   // Pass4:桶分组已就绪,桶内 O(b²) 定位直写
                for (int k = tid; k < count; k += blockDim.x) {
                    int c = sh_col[k]; int b = (c - cs2_lo) >> cs2_SH; int s = cs2_start[b];
                    int wr = 0;
                    for (int q = s, e = s + cs2_cnt[b]; q < e; q++) wr += (sh_col[q] < c);
                    dC_ci[base2 + s + wr] = c;
                    dC_val[base2 + s + wr] = sh_val[k];
                }
            } else {
                for (int k = tid; k < count; k += blockDim.x) {
                    int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
                    for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
                    dC_ci[base2 + rank] = c;
                    dC_val[base2 + rank] = v;
                }
            }
            __syncthreads();
            if (rowtime && tid == 0) rowtime[i] = clock64() - rt0;
            return;   // row_nnz 已由 count pass 写过(精确)
        }
        // SAFETY:行槽位 cap = est(= row_off 差);欠估行(distinct>est)越界写会砸下一行 → 守卫+overflow
        long long cap = row_off[i + 1] - row_off[i];
        if (cs2_cnt) {   // Pass4:桶内 O(b²) 位次 + tmp 直写(rank 与 legacy 同集:排序后前 cap 个最小键)
            for (int k = tid; k < count; k += blockDim.x) {
                int c = sh_col[k]; int b = (c - cs2_lo) >> cs2_SH; int s = cs2_start[b];
                int wr = 0;
                for (int q = s, e = s + cs2_cnt[b]; q < e; q++) wr += (sh_col[q] < c);
                int rank = s + wr;
                if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
                tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[base + rank] = sh_val[k];
            }
        } else {
            for (int k = tid; k < count; k += blockDim.x) {      // count-sort:数 < 自己的 → rank → 落有序位
                int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
                for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
                if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
                tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[base + rank] = v;
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (count > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
            row_nnz[i] = (count <= cap) ? count : (int)cap;
        }
        }   /* 刀E5:legacy extract else 闭合 */
    } else if (MODE == 2) {
        // 大行 MODE=2:无序直写 dC(精确偏移;行内乱序 → 原地 csort 善后)
        int base2 = exact_rp[i];
        for (int s = tid; s < ht_size; s += blockDim.x) {
            int c = sh_col[s];
            if (c >= 0) {
                int pos = atomicAdd(&cnt, 1);
                dC_ci[base2 + pos] = c;
                dC_val[base2 + pos] = sh_val[s];
            }
        }
    } else {
        // 大行:无序 extract(交 compact_sort;SAFETY:cap=est 守卫同上)
        long long cap = row_off[i + 1] - row_off[i];
        for (int s = tid; s < ht_size; s += blockDim.x) {
            int c = sh_col[s];
            if (c >= 0) {
                int pos = atomicAdd(&cnt, 1);
                if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
                tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[base + pos] = val_rd[s];   // spec91 刀E:VG 行值在 gmem 块切片(val_rd 统一指针)
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (cnt > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
            row_nnz[i] = (cnt <= cap) ? cnt : (int)cap;
        }
    }
    if (rowtime && tid == 0 && !rt_pre_extract) rowtime[i] = clock64() - rt0;   // 大/小行公共尾(extract 完成;NOEXTRACT 模式已在循环尾写过)
}

// hash_spa_hkv_kernel:Hybrid Value 全 bin 版(docs/24 §4.1,治大表行低占用):
// keys 留 SMEM(4B/槽,occupancy ×3)+ values 走全局 L2 atomicAdd(Ocean HYBRID_HASHMAP 同款)。
// 仅用于 ht ≥ 4096 的 bin(ht > CSORT_HT 恒真)→ 只保留无序 extract 路径(compact 走 csort)。
static __global__ void hash_spa_hkv_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    double *gval, const long long *gval_off)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int *sh_col = smem;                        // [ht_size] 仅 keys(12B→4B/槽)
    for (int s = tid; s < ht_size; s += HASH_BLOCK) sh_col[s] = -1;
    __syncthreads();

    long long gb = gval_off[i];                // 本行 value 池基址(host 已 memset 0)
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = 1 << hash_logG(ht_size);   // 同 hash_logG 梯({4,8,16,32})
    int num_groups = HASH_BLOCK / G;
    int my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&gval[gb + slot], v); break; }
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 无序 extract(ht ≥ 4096 > CSORT_HT;compact 侧 csort 排序)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = gval[gb + s];
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (cnt > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
        row_nnz[i] = (cnt <= cap) ? cnt : (int)cap;
    }
}

// spec98 轮1:WMA 重行归并累积器(分列段多 warp 流式归并;一 block 一行,8 warp 各认一个
// 列子区间;两阶段=count→块内前缀→re-merge 直写 tmp_key 兼容 hash 下游;dup 在归并中合并)。
// 底版=merge3 v2(warp k-way)×列区间并行 − 单 warp 串行死穴;SMEM=8×num_k×12B 描述符。
static __global__ void __launch_bounds__(1024, 1) wma_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_cols,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int i = bucket_rows[blockIdx.x];
    if (i < 0) return;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    if (num_k <= 0) { if (tid == 0) row_nnz[i] = 0; return; }
    if (num_k > 512) { if (tid == 0) { *overflow_flag = 1; row_nnz[i] = 0; } return; }   // 超帽→retry 回退(hash_global 重跑,正确性保底)
    const int NW = 32;                                   // warp 数(=列子区间数;1024T)
    extern __shared__ __align__(8) int smem[];
    int    *seg_ptr = smem;                              // [NW × num_k] 各 warp 各链当前位
    int    *seg_end = smem + NW * num_k;                 // [NW × num_k] 各 warp 各链段尾
    double *weight  = (double*)(smem + 2 * NW * num_k);  // [num_k] a_{i,k}(全 warp 共享,只读)
    int    *rng_cnt = (int*)(smem + 3 * NW * num_k);     // [NW] 各区间计数(两阶段)
    // 装载权重+清计数
    for (int p = tid; p < num_k; p += blockDim.x) weight[p] = A_val[rs + p];
    for (int w = tid; w < NW; w += blockDim.x) rng_cnt[w] = 0;
    __syncthreads();
    // 各 warp:对每条链二分定位本区间 [c_lo, c_hi) 的段界(32 lane 并行扫链)
    long long c_lo = (long long)A_cols * warp / NW, c_hi = (long long)A_cols * (warp + 1) / NW;
    int *my_ptr = seg_ptr + (size_t)warp * num_k, *my_end = seg_end + (size_t)warp * num_k;
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int s = A_row_ptr[k], e = A_row_ptr[k + 1];
        int lo = s, hi = e;                              // lower_bound(c_lo)
        while (lo < hi) { int mid = (lo + hi) >> 1; if (A_col_idx[mid] < (int)c_lo) lo = mid + 1; else hi = mid; }
        int l1 = lo; hi = e;                             // lower_bound(c_hi)
        while (l1 < hi) { int mid = (l1 + hi) >> 1; if (A_col_idx[mid] < (int)c_hi) l1 = mid + 1; else hi = mid; }
        my_ptr[p] = lo; my_end[p] = lo == l1 ? lo : l1;  // 空段=lo==l1(免后续读)
    }
    __syncthreads();
    // ── 阶段A:归并计数(每 distinct 一迭代;lane 局部 min→warp 归约→match 前进)──
    // 与 v2 同构,但段界=本区间(跨区间列互不可见 ⇒ 各 warp 独立成立)
    #define WMA_MERGE(WRITE_)                                                            \
    {                                                                                    \
        long long base = row_off[i] + (WRITE_ ? (long long)rng_cnt[warp] : 0);           \
        int out_idx = 0;                                                                 \
        while (true) {                                                                   \
            int mymin = 0x7fffffff;                                                      \
            for (int p = lane; p < num_k; p += 32) {                                     \
                int pos = my_ptr[p];                                                     \
                if (pos < my_end[p]) { int c = A_col_idx[pos]; if (c < mymin) mymin = c; } \
            }                                                                            \
            for (int off = 16; off > 0; off >>= 1) {                                    \
                int v = __shfl_xor_sync(0xffffffffu, mymin, off);                        \
                if (v < mymin) mymin = v;                                                \
            }                                                                            \
            if (mymin == 0x7fffffff) break;                                              \
            double s = 0.0;                                                              \
            for (int p = lane; p < num_k; p += 32) {                                     \
                int pos = my_ptr[p];                                                     \
                if (pos < my_end[p] && A_col_idx[pos] == mymin) {                        \
                    s += weight[p] * A_val[pos];                                         \
                    my_ptr[p] = pos + 1;                                                 \
                }                                                                        \
            }                                                                            \
            for (int off = 16; off > 0; off >>= 1)                                       \
                s += __shfl_xor_sync(0xffffffffu, s, off);                               \
            if (WRITE_ && lane == 0) {                                                   \
                tmp_key[base + out_idx] = ((unsigned long long)i << 32) | (unsigned)mymin; \
                tmp_val[base + out_idx] = s;                                             \
            }                                                                            \
            out_idx++;                                                                   \
        }                                                                                \
        if (lane == 0) rng_cnt[warp] = out_idx;                                          \
    }
    WMA_MERGE(0)                                          // 阶段A:只数
    __syncthreads();
    // 块内前缀 → 各 warp 写起点;阶段B 从头 re-merge(重走一遍,确定性同序)
    if (tid == 0) {
        int acc = 0;
        for (int w = 0; w < NW; w++) { int c = rng_cnt[w]; rng_cnt[w] = acc; acc += c; }
        row_nnz[i] = acc;
        if ((long long)acc > row_off[i + 1] - row_off[i]) *overflow_flag = 1;   // 超槽(低估)
    }
    __syncthreads();
    // 重置段界(阶段B 要从头)——重新二分(与装载同码)
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int s = A_row_ptr[k], e = A_row_ptr[k + 1];
        int lo = s, hi = e;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (A_col_idx[mid] < (int)c_lo) lo = mid + 1; else hi = mid; }
        int l1 = lo; hi = e;
        while (l1 < hi) { int mid = (l1 + hi) >> 1; if (A_col_idx[mid] < (int)c_hi) l1 = mid + 1; else hi = mid; }
        my_ptr[p] = lo; my_end[p] = lo == l1 ? lo : l1;
    }
    __syncthreads();
    WMA_MERGE(1)                                           // 阶段B:直写(base 已含 rng_cnt[warp] 前缀偏移)
    #undef WMA_MERGE
}

// compact 原地化守卫:margin[j] = E_off[j] − C_rp[j](≥0 全体成立 → 前缀紧缩安全,docs/21 Fix2)
// spec90-候选2 二维梯:tier 行群 Σest 归约(工作量轴原料;与 dense_tier_block 的 span 轴解耦)
static __global__ void tier_est_stat_kernel(const int *est, const int *rows, int n, unsigned long long *sum_est) {
    unsigned long long acc = 0;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x)
        acc += (unsigned long long)est[rows[r]];
    if (acc) atomicAdd(sum_est, acc);
}

// spec92-A:bin 行群 Σa_len 归约(宽窄判据原料;单块,256 线程 atomic 累加)
static __global__ void bin_alen_stat_kernel(const int *A_row_ptr, const int *bucket_rows, int n,
                                     unsigned long long *sum_alen) {
    int s = threadIdx.x;
    unsigned long long acc = 0;
    for (int r = blockIdx.x * blockDim.x + s; r < n; r += gridDim.x * blockDim.x)
        acc += (unsigned long long)(A_row_ptr[bucket_rows[r] + 1] - A_row_ptr[bucket_rows[r]]);
    if (acc) atomicAdd(sum_alen, acc);
}

static __global__ void margin_kernel(const long long *eoff, const int *crp, int n, long long *marg)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    marg[j] = eoff[j] - (long long)crp[j];
}

// 每行 HT 梯表尺寸(与 compute_bucket 同梯;hybrid value 池的 scan 输入)
static __global__ void ladder_ht_kernel(const int *est, int A_rows, int *lht)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int target = (e <= TABLE_DOUBLING_THR) ? 2 * e : e;  // α=0.5 松弛(仅默认档内)
    int ht = WARP_SIZE;   // 最小表 = 1 warp 槽
    while (ht < target && ht < HASH_CAP) ht <<= 1;
    lht[i] = ht;
}

// hash_spa_priv_kernel:warp 私有 SPA(HASH_PRIV 门控):k 按 warp 分区→Phase A 无 atomicAdd 累加,Phase B 跨 warp merge,Phase C extract
static __global__ void hash_spa_priv_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *bucket_rows, int n_in_bucket, int ht_size, int W,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;                  // blockDim.x = W*32,warp ∈ [0,W)
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                                  // [W*ht]
    double *sh_val = (double*)(smem + (size_t)W * ht_size);// [W*ht]
    int   *my_col = sh_col + (size_t)warp * ht_size;
    double *my_val = sh_val + (size_t)warp * ht_size;

    for (int s = lane; s < ht_size; s += 32) { my_col[s] = -1; my_val[s] = 0.0; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];

    // Phase A:warp 私有累加(k 按 warp 分区;plain +=,无 atomicAdd)
    for (int p = rs + warp; p < re; p += W) {              // 本 warp 的 k(stride W)
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {         // warp 内 32 lane 并行 j
            int j = A_col_idx[q];
            double v = a_ik * A_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&my_col[slot], -1, j);
                if (old == -1 || old == j) { my_val[slot] += v; break; }   // plain +=(同 warp 时间不并发)
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }
            }
        }
    }
    __syncthreads();

    // Phase B:merge warp 1..W-1 → warp 0 表(共享,atomicAdd,fan-in ≤ W)
    if (warp > 0) {
        for (int s = lane; s < ht_size; s += 32) {
            int c = my_col[s];
            if (c >= 0) {
                double v = my_val[s];
                unsigned slot = ((unsigned)(c * 2654435761u)) & mask;
                int probes = 0;
                while (true) {
                    int old = atomicCAS(&sh_col[slot], -1, c);
                    if (old == -1 || old == c) { atomicAdd(&sh_val[slot], v); break; }
                    slot = (slot + 1) & mask;
                    if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }
                }
            }
        }
    }
    __syncthreads();

    // Phase C:extract warp 0 表 → tmp(同 flat)
    int *w0_col = sh_col; double *w0_val = sh_val;
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    if (ht_size <= CSORT_HT) {
        for (int it = 0; it < (ht_size + blockDim.x - 1) / blockDim.x; it++) {
            int s = it * blockDim.x + tid;
            int c = -1; double v = 0.0;
            if (s < ht_size) { c = w0_col[s]; v = w0_val[s]; }
            __syncthreads();
            if (c >= 0) { int pos = atomicAdd(&cnt, 1); sh_col[pos] = c; sh_val[pos] = v; }
            __syncthreads();
        }
        int count = cnt;
        for (int k = tid; k < count; k += blockDim.x) {    // count-sort:数 < 自己的 → rank → 落有序位
            int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
            for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = v;
        }
        __syncthreads();
        if (tid == 0) row_nnz[i] = count;
    } else {
        for (int s = tid; s < ht_size; s += blockDim.x) {
            int c = w0_col[s];
            if (c >= 0) {
                int pos = base + atomicAdd(&cnt, 1);
                tmp_key[pos] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[pos] = w0_val[s];
            }
        }
        __syncthreads();
        if (tid == 0) row_nnz[i] = cnt;
    }
}

// ultrasparse:est≤EST_ULTRA_THR,不建 hash,每线程一行寄存器小数组线性去重累加;CAP=32 留 2× 余量,不够→overflow_flag
static __global__ void hash_ultra_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *ultra_rows, int n_ultra,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ultra) return;
    int i = ultra_rows[idx];
    const int CAP = WARP_SIZE;  // 2× EST_ULTRA_THR(α=0.5 负载因子)                          // 2× 于 EST_ULTRA_THR,吸收估计低估
    int u_col[CAP]; double u_val[CAP]; int u_cnt = 0;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p]; double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];          // B 的行 k(AA=A;ATT=Aᵀ)
        for (int q = ks; q < ke; q++) {
            int j = B_col_idx[q]; double v = a_ik * B_val[q];
            if (upper_tri && j < i) continue;                   // ATT 上三角过滤
            int pos = -1;
            for (int t = 0; t < u_cnt; t++) if (u_col[t] == j) { pos = t; break; }
            if (pos >= 0) u_val[pos] += v;
            else {
                if (u_cnt >= CAP) {                                    // 超 CAP → 记录行交重试
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    return;
                }
                u_col[u_cnt] = j; u_val[u_cnt] = v; u_cnt++;
            }
        }
    }
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];   // SAFETY:est 槽位守卫(欠估行防越界砸下一行)
    // 插入排序(u_cnt ≤ CAP=32,单线程 ~512 次比较):输出有序 → compact 走纯 copy,
    // 免去每行一个 BlockRadixSort CTA(germany_osm 11.5M ultra 行 × csort = 62ms 教训)
    for (int t = 1; t < u_cnt; t++) {
        int c = u_col[t]; double v = u_val[t];
        int s = t - 1;
        while (s >= 0 && u_col[s] > c) { u_col[s + 1] = u_col[s]; u_val[s + 1] = u_val[s]; s--; }
        u_col[s + 1] = c; u_val[s + 1] = v;
    }
    int out = 0;
    for (int t = 0; t < u_cnt; t++) {
        if (out >= cap) {                                    // 欠估 → 记录行交重试
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
            break;
        }
        tmp_key[base + out] = ((unsigned long long)i << 32) | (unsigned int)u_col[t];
        tmp_val[base + out] = u_val[t];
        out++;
    }
    if (out < u_cnt) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
    row_nnz[i] = out;
}

// ===================== heavy 全局表路径(2026-08-25) =====================
// est > HASH_CAP 的重行不再回退 merge3(ocean337 上 67 阵 28.5× 灾难的主源),
// 改走 global-memory 开放寻址 hash 表(对标 Ocean numeric.overflow 的全局回退)。
// 每行一张 next_pow2(est) 表(≥2×HASH_CAP),表区 arena 按 scan 偏移切分;
// probe 有上界(必终止);extract 无序写 tmp → host 侧 cub 分段排序 → compact_copy。

static __global__ void heavy_prep_kernel(
    const int *rows, int n, const int *est,
    long long *row_ht, int *overflow_flag)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int e = est[rows[t]];
    if (e > GLOBAL_HT_MAX_SLOTS) {            // 超过单表上限:真溢出,走原回退语义
        row_ht[t] = 0;
        atomicExch(overflow_flag, 1);
        if (t == 0 || (t & 255) == 0)        return;
    }
    long long h = 2 * HASH_CAP;               // 最小 32768
    while (h < 2LL * e) h <<= 1;              // 2× 余量:est 低估(MinHash 误差)时不炸
    row_ht[t] = h;
}

static __global__ void hash_global_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_ht, const long long *tab_off,
    int *tab_col, double *tab_val,
    const long long *row_off, const int *est, long long tmp_total,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int ht_size = (int)row_ht[idx];
    int mask = ht_size - 1;
    int *sh_col = tab_col + tab_off[idx];      // 本行的全局表段
    double *sh_val = tab_val + tab_off[idx];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = WARP_SIZE, num_groups = HASH_BLOCK / G, my_group = tid / G, my_id = tid % G;   // G-group ≤ warp(免跨 warp 同步)
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;                    // 有界探测:必终止
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 无序 extract → tmp(host 侧 cub 分段排序后交 compact_copy)
    // SAFETY:行槽位 = est[i](gapped buffer 按 est 分配);真实 distinct > est(MinHash 低估)时
    // 越界写会砸别的行/数组尾 → 守卫:超 est 只置 overflow 标志(整阵安全回退),绝不越界。
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) {
               row_nnz[i] = cnt > cap ? cap : cnt;
    }
}

// R1c(spec85,RETRY2=1):hash_global 拆三段——原版每 block 自含 init+probe+extract,
// __syncthreads 假设单 block 独占行(R0 解剖:3 行鲸鱼 1 block/行 = GPU 尾 ~190µs 真源)。
// 拆段:init(host memsetAsync)/probe(grid.y=S 段并行,共表原子 CAS——块内并发本依赖原子,
// 跨 block 天然成立)/extract(单 block/行,流序保证 probe 全落)。仅 retry 路径调用,原 kernel 不动。
static __global__ void retry2_gprobe_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val, int upper_tri,
    const int *bucket_rows, int n_in_bucket, int S,
    const long long *row_ht, const long long *tab_off,
    int *tab_col, double *tab_val,
    int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int seg = blockIdx.y;
    int tid = threadIdx.x;
    int ht_size = (int)row_ht[idx];
    int mask = ht_size - 1;
    int *sh_col = tab_col + tab_off[idx];      // 本行的全局表段(已由 host memset 初始化)
    double *sh_val = tab_val + tab_off[idx];
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int chunk = ((re - rs) + S - 1) / S;        // k 值域连续段切分;空段(s0>=s1)自然跳过
    int s0 = rs + seg * chunk;
    int s1 = min(rs + (seg + 1) * chunk, re);
    int G = WARP_SIZE, num_groups = HASH_BLOCK / G, my_group = tid / G, my_id = tid % G;
    for (int p = s0 + my_group; p < s1; p += num_groups) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                    break;
                }
            }
        }
    }
}
static __global__ void retry2_gextract_kernel(
    const int *bucket_rows, int n_in_bucket,
    const long long *row_ht, const long long *tab_off,
    int *tab_col, double *tab_val,
    const long long *row_off, const int *rslot_cap,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int ht_size = (int)row_ht[idx];
    int *sh_col = tab_col + tab_off[idx];
    double *sh_val = tab_val + tab_off[idx];
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    int cap = rslot_cap[i];   // retry 路径实参 = d_rslot(min(flop,ncols) ≥ distinct)
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) row_nnz[i] = cnt > cap ? cap : cnt;
}

// heavy 混合版(2026-08-26,Step2):keys 进 SMEM(ht ≤ HYBRID_HT_MAX,128KB)、values 留全局
// arena —— SMEM atomicCAS 比全局快 ~4-8×,治 TSOPF 类 heavy 全局表 3.86× 的主源。
// 对标 Ocean HYBRID_HASHMAP(keys SMEM + values 全局池)。smem 每 launch 均一 →
// host 按 ht ≤ HYBRID_HT_MAX 分两批launch。
#define HYBRID_HT_MAX HYBRID_HT_MAX_STRUCT  // = HASH_CAP×2(半档留 val 池)

static __global__ void hash_global_hybrid_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_ht, const long long *tab_off,
    int *tab_col_global, double *tab_val,     // tab_val 全局;tab_col_global 仅大行用(此 kernel 不写)
    const long long *row_off, const int *est, long long tmp_total,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int ht_size = (int)row_ht[idx];
    int mask = ht_size - 1;
    double *sh_val = tab_val + tab_off[idx];               // 值:全局(与纯全局版共用 arena)
    extern __shared__ __align__(8) int hkey[];             // 键:SMEM [ht_size]
    for (int s = tid; s < ht_size; s += blockDim.x) hkey[s] = -1;
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = WARP_SIZE, num_groups = blockDim.x / G, my_group = tid / G, my_id = tid % G;   // G-group ≤ warp(免跨 warp 同步)
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&hkey[slot], -1, j);                   // SMEM CAS
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }  // 全局 val
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    int p2 = atomicAdd(ovf_cnt, 1); if (!atomicExch(&row_ovf[i], 1)) ovf_rows[p2] = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = tid; s < ht_size; s += blockDim.x) {
        int c = hkey[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (cnt > cap && !atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
        row_nnz[i] = cnt > cap ? cap : cnt;
    }
}

// heavy 拆分辅助:从全表 gather (ht, tab_off) 到子表(按子表行号)
static __global__ void heavy_gather_kernel(
    int *sub_rows, int n_sub, const int *full_rows,
    const long long *full_ht, const long long *full_off,
    long long *sub_ht, long long *sub_off)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_sub) return;
    int idx = sub_rows[t];          // heavy 表内位置
    sub_ht[t] = full_ht[idx];
    sub_off[t] = full_off[idx];
    sub_rows[t] = full_rows[idx];   // 原地换成真实行号(kernel 按此读 A/B)
}

// heavy 行分段排序的 begin/end 偏移(基于 gapped d_off + 真实 row_nnz)
static __global__ void heavy_seg_kernel(
    const int *rows, int n, const long long *row_off, const int *row_nnz,
    int *seg_beg, int *seg_end)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i = rows[t];
    seg_beg[t] = row_off[i];
    seg_end[t] = row_off[i] + row_nnz[i];
}

// ---- 行级重试 kernels(2026-08-26) ----
// 每行真实列跨度(docs/24 跨度窗口):B(CSR 有序)每行首/末元素即该行最小/最大列 →
// 行 i 的乘积列集合 ⊆ [min_k ci[rp[k]], max_k ci[rp[k+1]-1]]。O(nnz_row) 两次寻址,免费。
static __global__ void row_span_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *span_lo, int *span_len,
    int *max_b_len /*nullable:本行 k 集内 B 行最大长(localLoadBalance 输入)*/)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int lo = INT_MAX, hi = -1, mb = 0;
    for (int p = A_row_ptr[i]; p < A_row_ptr[i + 1]; p++) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            if (c0 < lo) lo = c0;
            if (c1 > hi) hi = c1;
            if (b1 - b0 > mb) mb = b1 - b0;
        }
    }
    span_lo[i]  = (lo == INT_MAX) ? 0 : lo;
    span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
    if (max_b_len) max_b_len[i] = mb;
}


struct ToLL {   // thrust scan 输入升 64 位(thrust 按输入 value_type 累加,int 输入 Σ>2^31 回绕)
    __host__ __device__ long long operator()(int x) const { return (long long)x; }
};

// prep:ht = pow2(2×min(flop, ncols))(每行 distinct ≤ min(flop,n),高 dup 行 flop 是 distinct 的
// 几千倍,裸 flop 定表 = 内存爆炸根源之一,docs/21);超全局表上限 → 不可重试(保持 overflow → 整阵回退)
// v40 SMEM 重试车配套:ovf 行的 flop 收集(host 求 max 定重试表)
static __global__ void gather_flop_kernel(const int *rows, int n, const int *flop, int *out, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long f = flop[rows[t]];
    if (f > ncols) f = ncols;
    out[t] = (int)f;
}

static __global__ void retry_prep_kernel(
    const int *rows, int n, const int *flop,
    long long *rht, int *row_ovf, int *overflow_flag, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int row = rows[t];
    long long f = flop[row];
    if (f > ncols) f = ncols;                              // distinct ≤ ncols(数学封顶)
    if (2 * f > (long long)GLOBAL_HT_MAX_SLOTS) {         // 不可重试:overflow 已置。Fix(study/16 排雷):
        rht[t] = 2LL * HASH_CAP;                          // 仍写合法表值——裸 alloc 无 memset,早退留垃圾
        return;                                           // 进 exclusive_scan → tab_off 垃圾 → hash_global 越界写
    }
    long long h = 2 * HASH_CAP;
    while (h < 2 * f) h <<= 1;
    rht[t] = h;
    row_ovf[row] = 1;   // 已在 accumulate 记录时置位;此处幂等
}
// slot:重试区按行 id 的槽位数(=min(flop, ncols);非重试行为 0,先 memset)
static __global__ void retry_slot_kernel(const int *rows, int n, const int *flop, int *rslot, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int f = flop[rows[t]];
    if (f > ncols) f = ncols;
    rslot[rows[t]] = f;
}

// R1 子集快路(spec85 RETRY2=1):重试区偏移只在溢出行子集上求,替代全 A_rows memset+inclusive_scan
// (R0 解剖:全 A_rows 扫+memset ≈195μs 是 host 编排最大项;消费者只按 ovf 行列表读 d_roff/d_rslot)
static __global__ void retry_rsub_gather_kernel(const int *rows, int n, const int *flop, long long *rsub, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    long long f = flop[rows[t]];
    if (f > ncols) f = ncols;
    rsub[t] = f;
}
static __global__ void retry_suboff_scatter_kernel(const int *rows, int n, const long long *rosub, const long long *rsub,
                                            int *rslot, long long *roff)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int row = rows[t];
    rslot[row] = (int)rsub[t];
    roff[row]  = rosub[t];
}

// R1b(spec85,RETRY2=1):单 block 融合 kernel,替代 retry 路径全部 thrust 调用——R1 v1 教训:
// thrust sort/scan 每调用临时 cudaMalloc,legacy 分配器下同步百 µs 级,是 retry 相位真凶。
// 一次完成:①d_rtab = rht 原始序 exclusive scan(全局路按 d_ovf_rows 序消费,语义不变)
// ②ovf 行稳定升序排名(SMEM,O(n²) 微秒级)③rank 序 rsub exclusive scan ④scatter d_roff/d_rslot
// ⑤尾值(r_slots/last_h/rt_prev)一次写出 → host 仅 1 次 24B D2H。n ≤ R2B_MAX_N;更大回退 R1 v1 路径。
#define R2B_MAX_N 1024
template <int TPB>
__global__ void retry_fused_off_kernel(
    const int *rows, int n, const int *flop, int ncols,
    const long long *rht, long long *rtab,
    int *rslot, long long *roff, long long *tails)
{
    using BlockScanLL = cub::BlockScan<long long, TPB>;
    __shared__ typename BlockScanLL::TempStorage ts;
    __shared__ int s_rows[R2B_MAX_N];
    __shared__ int s_rsub[R2B_MAX_N];    // rank 序 rsub
    __shared__ int s_rbyr[R2B_MAX_N];    // rank 序行号
    constexpr int IPT = (R2B_MAX_N + TPB - 1) / TPB;
    int tid = threadIdx.x;
    for (int t = tid; t < n; t += blockDim.x) s_rows[t] = rows[t];
    __syncthreads();
    // ① 稳定排名(行号升序)+ 按 rank 摆 rsub/行号
    for (int t = tid; t < n; t += blockDim.x) {
        int v = s_rows[t], r = 0;
        for (int u = 0; u < n; u++) r += (s_rows[u] < v) || (s_rows[u] == v && u < t);
        long long f = flop[v]; if (f > ncols) f = ncols;
        s_rbyr[r] = v; s_rsub[r] = (int)f;
    }
    __syncthreads();
    // ② rht 原始序 exclusive scan → rtab
    { long long a[IPT]; int k = 0;
      for (int t = tid * IPT; t < tid * IPT + IPT; t++) a[k++] = (t < n) ? rht[t] : 0;
      BlockScanLL(ts).ExclusiveSum(a, a);
      for (int t = tid * IPT, k = 0; t < tid * IPT + IPT; t++, k++) if (t < n) rtab[t] = a[k];
    }
    __syncthreads();
    // ③ rank 序 scan → roff/rslot 散射 ④ 尾值
    { long long a[IPT], tot = 0; int k = 0;
      for (int t = tid * IPT; t < tid * IPT + IPT; t++) a[k++] = (t < n) ? (long long)s_rsub[t] : 0;
      BlockScanLL(ts).ExclusiveSum(a, a, tot);
      for (int t = tid * IPT, k = 0; t < tid * IPT + IPT; t++, k++)
          if (t < n) { int row = s_rbyr[t]; rslot[row] = s_rsub[t]; roff[row] = a[k]; }
      if (tid == 0) { tails[0] = tot; tails[1] = rht[n - 1]; tails[2] = rtab[n - 1]; }
    }
}

// compact:每行 row_nnz 项从 tmp(row_off 起)拷到连续 CSR(C_row_ptr 起)
static __global__ void diter_gapped_copy_kernel(   // spec89-#4'-L2b⑤:CNT_FUSE 老段行 tmp→dC 挤压拷贝
    const int *rows, int n, const long long *gap_off, const int *row_nnz,
    const int *tmp_col, const double *tmp_val,
    const int *C_row_ptr, int *out_col, double *out_val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int row = rows[t];
    long long src = gap_off[row]; int dst = C_row_ptr[row]; int k = row_nnz[row];
    int slot = (int)(gap_off[row + 1] - src);            /* #4'-fix2:行槽宽 clamp,防越界读 */
    if (k > slot) k = slot;
    for (int j = 0; j < k; j++) { out_col[dst + j] = tmp_col[src + j]; out_val[dst + j] = tmp_val[src + j]; }
}

static __global__ void hash_compact_kernel(
    int A_rows, const long long *row_off, const int *row_nnz, const int *C_row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    unsigned long long *out_key, double *out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    long long src = row_off[i]; int dst = C_row_ptr[i], n = row_nnz[i];
    for (int t = 0; t < n; t++) {
        out_key[dst + t] = tmp_key[src + t];
        out_val[dst + t] = tmp_val[src + t];
    }
}

// 拆 key 低 32 位 → col
static __global__ void split_key_kernel(const unsigned long long *key, int *col, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) col[t] = (int)(key[t] & 0xffffffffu);
}

// per-row compact+sort:每 block 一行,BlockRadixSort 行内按 col 排→直接写 CSR;一次替掉 compact+全局 sort+split_key
template<int TPB, int IPT>
__global__ __launch_bounds__(TPB, 2048 / (TPB * 2)) void hash_compact_sort_kernel(   // 刀14c:强制限寄存器(ocean sortOutput 同款 launch_bounds;ncu 实锤 255 regs → 1块/SM=12.5%占用)
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    using BlockRadixSort = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    extern __shared__ char csort_smem[];   // 动态 shared(cap 大时 >48KB 需 opt-in,见 launch_csort)
    typename BlockRadixSort::TempStorage &temp_storage =
        *reinterpret_cast<typename BlockRadixSort::TempStorage *>(csort_smem);
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    if (skip && skip[row]) return;               // 行级重试的行:数据在重试区,主区跳过
    long long start = row_off[row];
    int n = row_nnz[row];
    int out_start = row_ptr[row];

    unsigned int col[IPT];
    double val[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;   // BLOCKED 排布(匹配 cub::BlockRadixSort 默认)
        if (idx < n) {
            col[i] = (unsigned int)(tmp_key[start + idx] & 0xffffffffu);   // key 低 32 = col
            val[i] = tmp_val[start + idx];
        } else {
            col[i] = 0xffffffffu;   // sentinel → 排到末尾(只存前 n 个真实项)
            val[i] = 0.0f;
        }
    }
    BlockRadixSort(temp_storage).Sort(col, val, 0, 32);   // 按 col 排(低 32 位)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;   // BLOCKED 排布(匹配 cub::BlockRadixSort 默认)
        if (idx < n) {
            out_col[out_start + idx] = (int)col[i];
            out_val[out_start + idx] = val[i];
        }
    }
}

// 行级表配(解耦版,study/22 新第一刀 v2):路由吃的 est 不动(expand 1.8 照旧),
// 只有【表长目标】按 sketch E=e/eref 做行级折减 → 免 est-喂-多门的耦合轮盘
// (REXP=1 耦合版教训:crankseg −8.5 实为 occgate 门槛迁移,in-2004 +58.7 实为
//  direct 写入人口翻倍 —— 都是路由副作用,不是压表红利;解耦版量纯压表收益)。
// E≤elo → g=glo(默认=eref,恒等不碰);E≥ehi → g=ghi;中间 log 插值。
static SP_HD int tbl_target_rexp(int e, int alpha_pct, double glo, double ghi,
                          int elo, int ehi, double eref) {
    int target = (e <= TABLE_DOUBLING_THR) ? (int)((long long)e * alpha_pct / 100) : e;
    if (ehi <= 0 || e <= 0) return target;
    double E = (double)e / eref;
    double t = (E <= (double)elo) ? 0.0
             : (E >= (double)ehi) ? 1.0
             : (log(E) - log((double)elo)) / (log((double)ehi) - log((double)elo));
    double g = glo + (ghi - glo) * t;
    double nt = (double)target * g / eref;
    return nt < 1.0 ? 1 : (int)nt;
}

// ── D3②:PVD 学生派发(study/36)────────────────────────────────────────────
#include "student_dispatch.cuh"   // 学生树(margin0.7 教师蒸馏;再生 scripts/d3_{train_teacher,distill_student}.py)
#include "ocean_hash_port.cuh"    // 刀E6(study/57 §三十三):ocean hashNumeric 移植替换消融(HASH_OCNK 默认关)

// maxbl 预计算:行内最长 B 链长(平方阵 B=A;LLB 同源量,学生特征 #19);O(nnz) 一发
static __global__ void row_maxbl_kernel(const int *A_rp, const int *A_ci, int A_rows, int *maxbl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int mb = 0;
    for (int p = A_rp[i]; p < A_rp[i + 1]; p++) {
        int j = A_ci[p];
        int l = A_rp[j + 1] - A_rp[j];
        if (l > mb) mb = l;
    }
    maxbl[i] = mb;
}

// GPU 端分桶(无 host 往返):每行算 bucket_id 并直方图计数(融合省 1 launch)
static __global__ void compute_bucket_kernel(
    const int *est, const int *A_row_ptr, int A_rows, int ultra_thr,
    const int *flop, int diter_thr,
    const int *span_len,   // 每行真实列跨度(docs/24:替代 n 近似)
    int *bucket_id, int *counts,
    int n_cols = 0,        // DIM2:span 护栏(DIM2=0 时传 0 关闭)
    int span_factor = SPANF_COEFF,  // G-group 摊薄系数(docs/72 推导)
    int occgate = 0,      // docs/67 占用轴(est≥span/16):Ga 族 3.6% 占用行的 O(span) 窗税实锤
    int ladder = 0,       // docs/67 §6 自制双梯:两侧实测成本律逐行比较(取代手调门的方向)
    int batch2 = 0,       // docs/67 §8:est 65-256 小行批处理门(BIN_BATCH2)
    int sroute = 0,       // docs/72 Phase 2:结构化字节路由(span×8.125B ≤ ht×12B;2=叠加 est≥2048 保底)
    int ht_l_sroute = 0,  // SROUTE 用的 hash 表大小(host 预算的表槽)
    int batch_s_max = 0,  // docs/13 v2:est≤此值 ∧ kc≤warp → BIN_BATCH_S(64 槽表;0=关)
    int largest_on = 0,   // docs/74:Ocean-D straggler 行位图窗(0=关)
    int esc_on = 0,       // docs/76 v31-④:零表 counting-sort 短行(flop≤ESC_FLOP_MAX;0=关)
    int diter_est_min = EST_DENSE_MIN / 2,  // docs/76 §16:diter est 下限(默认半默认档;Ocean dense 分流)
    int dstruct_win = 0,  // §21:span ≤ 此窗才进 diter;-1 = §22 档位梯(span ≤ DENSE_TIER_SPAN_MAX);
                          //      0 = 不限(旧行为);>0 = 手调单窗(实验兼容)
    int dim2_on = 0,     // DIM2 独立门(n_cols 传参只为 largest 的 span 护栏;误点亮 dim2 曾把
                          // bcsstk30 642 行送 BIN_DITER → +24.8%,2026-09-03 惊吓扫实证)
    int dtier_cmp = 0,   // §22b Ocean 双维档位比较(DTIER_CMP=1 且 DSTRUCT_WIN=-1 时生效):
    int tgt_alpha_pct = 200,  // 表配 α(study/10 勘误刀):target=α%×est(档内);旧 2×∧est 高估
    double tbl_glo = 0.0, double tbl_ghi = 0.0,   // 行级表配(解耦版 HASH_REXP=2):0=关;
    int tbl_elo = 0, int tbl_ehi = 0,             // 默认锚 glo=1.8(=eref 恒等)/ghi=1.4/elo=ehi=4096
    double tbl_eref = 1.8,                        // eref = 实际 est_expand(E = e/eref)
    int occ_coeff = SPANF_COEFF,
    int force_est_min = 0,   // 决赛 v2 真强制
    int had_l1 = 0, int k7_on = 0,   // k7_on:BIN_K7 路由开关(env K7,默认关)
    int d6_on = 0, const int *d6_maxbl = nullptr,          // D6EXP:PVD 学生派发(study/36;默认关)
    float d6_cr = 0.f, float d6_cs = 0.f, float d6_cm = 0.f, int d6_ncols = 0, int ovr_se = 8)          // HAD L1 路由修正(env HAD_L1,host 读取传入:est≥此值终判 BIN_DITER(覆盖全门;0=关)                  // OCCGATE 系数(默认 16;OCC_GATE_COEFF=10 已验证但停靠③阶段启用;env OCCGATE_COEFF —— REXP
                                                  // 轮盘教训:crankseg −8.5 实为本门 est×16<span 边界
                                                  // 迁移,直接调门替代 est 侧信道;变小=更多重行留 hash)
                          //   1.63× → 表≈真需求 3.3× vs Ocean 1.75×;env HASH_TGT_ALPHA(200=legacy)
                          //   档内行(span≤最大档)要求 dense 档 ≤ hash 档(窗口 vs 表,同一 SMEM
                          //   预算梯);大 span 行走老窗口内核,比较不适用。in-2004 校准通过率 95.3%
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int bid;
    int sl0 = span_len[i];
    // DIM2(docs/61 双维路由复活,保守版):Ocean 逐行双维代价比较(dense_bin ≤ hash_bin → dense)
    // 移植 —— 但 v4 门全保留,双维只【增补】est<2048 的行:hash_bin 先按梯算出(含 2× 松弛),
    // dense_bin 按 span 六档;dense_bin ≤ hash_bin ∧ flop≥4096 ∧ 16flop≥span ∧ dup<8 ∧ span≤n/4
    // → 也进 BIN_DITER。预期捕获 c-big 类(Ocean 捕 7.7% 行 dense,我们 0.03%);F2 类靠
    // flop/dup 门继续受保护。DNF 防复发:dbg 打印路由量级 + cnnz 对 scipy。
    int hash_bin = 0, ht = WARP_SIZE;                       // 最小表 = 1 warp 槽(hash_slots_pow2 同下限)
    {
        int target = tbl_target_rexp(e, tgt_alpha_pct, tbl_glo, tbl_ghi, tbl_elo, tbl_ehi, tbl_eref);
        while (ht < target && ht < HASH_CAP) { ht <<= 1; hash_bin++; }
    }
    // dense 六档 = Ocean DENSE_NUMERIC_BIN_SIZES[128KB] 公式移植:占用梯{32,16,8,4,2}÷128K
    //   → 每块预算{4,8,16,32,64}KB → ÷(val 8B+flag 1B)−4 = {451,906,1818,3641,7284}
    int dense_bin = (sl0 <= dense_numeric_bin(0)) ? 0 : (sl0 <= dense_numeric_bin(1)) ? 1
                  : (sl0 <= dense_numeric_bin(2)) ? 2
                  : (sl0 <= dense_numeric_bin(3)) ? 3 : (sl0 <= dense_numeric_bin(4)) ? 4 : 5;
    // ⚠ CPU 逐行分析(08-30):c-big 全部 straggler 行卡在 span≤n/4 护栏(其余门全过)——
    // Ocean 数据驱动窗口本就处理大 span,护栏对 DIM2-s 分支放宽到 span≤n(hash_bin 侧自然限流)。
    bool dim2 = dim2_on && (dense_bin <= hash_bin) && sl0 > 0 && n_cols > 0 && (long long)sl0 <= (long long)n_cols;
    // Ocean-D largest(docs/74):straggler 行(超 SMEM-val dense 档 ∧ 字节模型赢 hash)→ BIN_LARGEST。
    //   判据(毫字节,全具名):span×LARGEST_COL + est×LARGEST_VAL ≤ ht_eff×HASH_SLOT + est×COMPACT
    //   (dense = 位图+前缀 0.25B/列 + val 经 tmp 16B/est;hash = 表 12B/槽 + compact 24B/est)
    //   ht_eff:est ≤ HASH_CAP 用 SMEM 梯表;超过 = heavy 全局表 = 2×est 槽(α=0.5;首版误用
    //   cap 后的 ht 把 19491 条 c-big straggler 行挡在门外 —— 2026-09-03 BINS_DBG 实证修正)
    const long long ht_eff = (e <= HASH_CAP) ? (long long)ht : 2LL * e;
    const bool largest_ok = largest_on && sl0 > 0 && n_cols > 0
        && e > HASH_CAP                                   // ⚠只路由 heavy 行:est≤HASH_CAP 走 SMEM 梯表(4-20× 便宜),
                                                          // 字节模型 1:1 计价漏此差 → c-64+169/F2+62/3Dspec+25 实证
        && (long long)sl0 > (long long)dense_numeric_bin(4)
        && sl0 >= LARGEST_MIN_SPAN                        // 位图字 ≥ 2×块宽(atomicOr 摊薄;守卫电池实证)
        && (long long)LARGEST_SPARSE_DEN * e <= (long long)sl0   // 输出稀疏:每字置位 ≤ 4(c-big 1/17 vs c-64 1/2.5)
        && flop[i] >= EST_DENSE_MIN
        && flop[i] <= DUP_LARGEST_MAX * e                 // dup ≤ 8:重乘积行 2×flop 重走反劣(同实证)
        && ((long long)sl0 * LARGEST_COL_MILLI + (long long)e * LARGEST_VAL_PER_EST_MILLI
            <= ht_eff * HASH_BYTES_PER_SLOT_MILLI + (long long)e * COMPACT_BYTES_PER_EST_MILLI);
    // 门 v2(333SP +160%/F2 +13% 教训:窄跨度但 est 小的行在 batched/hash 本来就快,勿偷):
    // 极窄 span≤512:数组比任何 hash 表都小,只要 e>64 就值得;中段 512<span≤2048:须 est≥2048
    // (= v4 门的"hash 伺候不了的大表行")。两档都要求 est≥span/2(数组利用率)。
    if ((esc_on & 1) && flop[i] > 0 && flop[i] <= ESC_FLOP_MAX && flop[i] <= ESC_DUP_MAX * e) {
        // v31-④ v4 门:dup≤ESC_DUP_MAX 才进 ESC —— counting 成本 ∝ flop、表成本 ∝ est;
        // germany(dup1.4)赢/patents(dup≥2)输的分界(v3 est 门试过:germany est≤16 丢收益;
        // ultra 复活实测 8× 慢 = 死代码正确的死,勿救)
        bid = (flop[i] <= ESC_SUBW8_MAX) ? BIN_ESC8
            : (flop[i] <= ESC_SUBW16_MAX) ? BIN_ESC16 : BIN_ESC32;
    } else if ((esc_on & 2) && flop[i] > ESC_FLOP_MAX && flop[i] <= 2 * ESC_FLOP_MAX
               && flop[i] <= ESC_DUP_MAX * e) {
        bid = BIN_ESC64;   // study/50 靶1:flop∈(32,64] ∧ dup≤2 → IPT=2 零表子组(同 dup 门)
    } else if ((esc_on & 4) && flop[i] > 2 * ESC_FLOP_MAX && flop[i] <= 4 * ESC_FLOP_MAX
               && flop[i] <= ESC_DUP_MAX * e) {
        bid = BIN_ESC128;   // T3(study/53):flop∈(64,128] ∧ dup 门 → NT=128 子组秩排序
    } else if (largest_ok) {
        bid = BIN_LARGEST;
    } else if (e > BATCH_EST_MAX && sl0 > 0 && sl0 <= SMALL_SPAN_MAX && 2LL * e >= (long long)sl0) {  // 数组 < 最小表(docs/37)
        // 小跨度 dense(docs/37;门 v3=收缩到 Ocean dense0 精确槽位:span≤256):pwtk +145%/F2 +14%
        // 教训 = 高 dup 行 dense 付逐乘积原子、hash 只付逐 distinct,中窄跨度(257-2048)反复被偷;
        // 256 内数组比一切 hash 表都小且密集,是唯一稳定赢的槽位(nemeth18 全族实证)。
        bid = 14;
    }
    else if (k7_on && e >= K7_EST_MIN && sl0 > 0 && sl0 <= K7_SPAN_MAX
             && (A_row_ptr[i + 1] - A_row_ptr[i]) >= 1
             && (A_row_ptr[i + 1] - A_row_ptr[i]) <= K7_ALEN_MAX
             && (long long)flop[i] - e <= K7_FIX_MAX                       // 碰撞预算(核内精确证书复核)
             && (long long)flop[i] <= (long long)(K7_DUP_MAX * (double)e)  // dup 预算(路由侧估计)
             && (long long)span_factor * flop[i] >= (long long)sl0         // spanf 门(diter 同款:重乘积才值得窗口化;
             && (long long)span_factor * e >= (long long)sl0               //  occgate 同款:est≥span/16,位图利用率)
             && 2LL * e < (long long)sl0                                   // T5b(10-23)dense 否决:est≥span/2 的稠密行归 dense
                                                                           // (hangGlider_4/5 K7抢dense行+73/92%→修平;
                                                                           // hangGlider_3/c-43 侧小损失可接受)
             ) bid = BIN_K7;                                               // 10-06 A/B:缺此双门 → 从 hash 抢行,Si10H16 −4%/c-43 −11%   // kernel₇:dup≈1∧窄span∧短a_len(10-06;位图 0.125B/列 vs 表 12B/槽)
    else if (e <= ultra_thr) bid = BIN_ULTRA;             // ultra:线性免 hash
    // dense-iter 路由(docs/24 裁决,docs/72 结构化推导)
    // 三个门控常数均从 structural_params.cuh 的字节模型推导(docs/72 Phase 2 论文策略):
    //   est≥2048 ⟺ hash 表字节 ht×12B ≥ 48KB(SMEM 最小档;表 ≥ 档位意味着 SMEM 换出/全局表,
    //             hash 吞吐骤降 → 窗口经济的起点)——对应 "e >= 2048" 条件
    //   16flop≥span ⟺ span×8.125B ≤ flop×G/2(G-group 并行摊薄后的 dense 有效字节)
    //             在 G=HASH_BLOCK/WARP_SIZE=8 时求解 → span ≤ 16×flop/8.125×G ≈ 16flop
    //   dup<8/64 ⟺ 争用字节 (flop−est)×B_cont ≤ compact 字节 est×24B 的两个工作点:
    //             flop<8est ⟺ B_cont<24B(dup<8 时争用可忽略,G-group 摊薄)
    //             flop≥16384 ∧ <64est ⟺ B_cont<24B×est/(flop−est) 的重行域(docs/37 实证)
    const long long fl64 = (long long)flop[i];
    // dup_ok(争用 vs compact 推导)+ HEAVY 候选行放宽:
    //   当 est>HASH_CAP 时,备选方案从 SMEM hash 变为全局内存 hash——全局原子比 SMEM 慢
    //   一个量级,dense(逐乘积)在高 dup 下仍优于全局 hash(逐 distinct×全局 CAS)。
    //   因此 HEAVY 候选行用 DUP_HEAVY 门(与 flop≥HASH_CAP 同一物理推导)。
    //   SiO2 定谳:3521ms accumulate(全局表)vs 8.4ms dense_direct = 419× 差距。
    bool dup_ok = (fl64 < DUP_LIGHT * e) || ((fl64 >= HASH_CAP || e > HASH_CAP) && fl64 < DUP_HEAVY * e);
    // ── HAD L1 三刀(④路由;had_l1 参数从 host 传入,默认 0)──
    if (had_l1) {
        // 刀1(nd24k): 高 dup × 高 est → 强制 hash(DUP_HEAVY=64 重域条款太宽松;
        //   四臂实测 allhash 107 vs default 143 = hash 快 25%)
        if (fl64 >= HASH_CAP && fl64 >= (long long)had_l1 * e && e >= 2048) dup_ok = false;   // 阈值=HAD_L1 env 值(1=旧30 兼容;09-30 判别式:c-64 q75=6.2/in-2004=1.0 全保护,crankseg 中位16/nd24k 34.8 进网→8 为候选)
        // 刀2(pre2): 近满密行(span/est<1.5 ∧ est≥256)→ 松 flop 地板走 dense_struct
        //   (DITER_FLOP_TIER=2048 拦死 pre2 型行[flop中位 698];dense_struct 同档 1.3-2× 快)
        // (实施在 v4 门段,此处标记)
        // 刀3(crankseg): 低 a_len × 长B行 → 不进 cursor(hash 快 25%)
        // ( crankseg 行被 DITER 的宽span路径捕获,但 hash 更优;由刀1的高dup覆盖大部分)
    }
    // §22d Ocean 双维档位比较:只作用于【档内】行(span ≤ DENSE_TIER_SPAN_MAX);
    // 超档行走老窗口 cursor 内核(多窗数据驱动,经济学已由 v28-v32 全库验证 —— est 地板
    // 维持结构化默认 EST_DENSE_MIN/2;档内行地板可松到 diter_est_min,§22 电池验证)。
    // v33b/c 两版电池定谳:豁免版(est 地板松到老路径)→ c-64 +54%;全域比较版 → 老内核
    // 赢家被误驱逐。六档梯把 in-2004 怪物(est≥4096∧span 7.3-18.9K)档内化是正解。
    const bool tier_fit = (long long)sl0 <= (long long)DENSE_TIER_SPAN_MAX;
    const int est_floor = tier_fit ? diter_est_min : (EST_DENSE_MIN / 2);
    // §22d 老路径的 flop 地板同样回结构化默认(c-64 定谳:flop 2048-4095 ∧ span 50K 的半产物
    // 全幅行,窗口扫描成本不变产物减半 → 摊销劣化 +2.8ms;档内行受益于松地板,分而治之)
    int flop_floor = tier_fit ? diter_thr : (diter_thr > EST_DENSE_MIN ? diter_thr : EST_DENSE_MIN);
    // HAD L1 刀2(pre2 v2): 近满密行(span<1.5×est ∧ est≥256 ∧ dup_ok ∧ tier_fit)松 flop 地板
    // v1(flap_floor=0 无条件)判负:pre2 1.93× 回退(松太激进把不该进 diter 的也放了)
    // v2:只在 dup_ok && tier_fit 时松到 diter_est_min(不全归零)
    // 判负入档(v1/v2 均败): pre2 行走 diter 后 dense_count=25.5ms(精确计数相位)
    // > 整条 hash 路径(17.4+8.3=25.7ms)——架构差(diter 需精确计数 vs 彼 HLL 估计免计)
    // 非 kernel 差(§4d 同档 numeric 1.3-2× 快被计数开销吃掉);修复需架构改非路由改
    // if (had_l1 && dup_ok && tier_fit && sl0 > 0 && e >= 256 && ...) { ... }  ← 禁用
    const bool tier_cmp_ok = !dtier_cmp || dstruct_win >= 0 || !tier_fit
        || dense_tier_of((long long)sl0) <= hash_tier_of((long long)ht);
    // ↑ dup<8:争用字节≈0(G-group 摊薄);dup<64∧flop≥16384:重行域的争用可容忍(docs/37)
    // docs/67 §6 自制双梯(LADDER=1):diter 候选行按两侧【实测成本律】裁决,取代手调门的方向。
    //   t_dense ≈ 30109·⌈span/5400⌉ + 2.55·flop       (120k 行 × 7 阵,R²=0.896)
    //   t_hash  ≈ 8.1·ht + 1.89·flop + 1038·kc − 50225 (大表行 R²=0.926;kc=输入链数,每链 1038 周期)
    // oracle(69k 配对行):86% diter 行 hash 更快;OCCGATE 一致率 69%,残留 1.94e9 周期(est 8k/占用 16% 类)。
    bool ladder_dense_ok = true;
    if (ladder) {
        // 窗数按真实窗宽 PB2_W 取整(5399/5400 时代与 PB2_W=5500 不一致,2026-09-03 单源化)
        int W = (int)(((long long)sl0 + PB2_W - 1) / PB2_W);
        long long td = TD_WIN_NS * W + TD_FLOP_CENTI * fl64 / 100;
        int tgt = tbl_target_rexp(e, tgt_alpha_pct, tbl_glo, tbl_ghi, tbl_elo, tbl_ehi, tbl_eref);
        int ht_l = WARP_SIZE; while (ht_l < tgt && ht_l < HASH_CAP) ht_l <<= 1;
        long long kc_n = (long long)(A_row_ptr[i + 1] - A_row_ptr[i]);
        long long th = TH_HT_CENTI * ht_l / 100 + TH_KC_NS * kc_n + TH_FLOP_CENTI * fl64 / 100 + TH_BASE_NS;
        ladder_dense_ok = (td <= th);   // dense 便宜才进窗;否则留 hash 梯
    }
    // docs/72 Phase 2 路由字节比较(SROUTE=1/3/4):结构化成本比较。
    //   SROUTE=1(纯表):span×8.125B ≤ ht_row×12B
    //   SROUTE=3(含 compact):span×8.125B ≤ ht_row×12B + est×24B
    //   SROUTE=4(含 compact+争用):span×8.125B + (flop−est)×争用B ≤ ht×12B + est×24B
    // 争用 B 推导:H100 SMEM 同地址 atomicAdd 吞吐 ≈ 1/4 cycles vs 不同地址 ≈ 1/cycle
    //   → 每"多余乘积"(flop>est 部分)多付 3 cycles ≈ 3×4B = 12B 争用字节
    //   (Ga41 dup=38× → 37×est×12B ≈ 巨额 → 正确回 hash;F2 dup<3 → 小额 → 留 dense)
    if (bid == BIN_LARGEST || (bid >= BIN_ESC8 && bid <= BIN_ESC128) || bid == BIN_K7) {}   // k7 同 largest/esc:已定案不被后续 diter 链覆写(10-06 bug:漏豁免致全数被 diter 抢走)
    // ↑ largest/esc 已定案,后续链(小跨度/ultra/v4/heavy)不覆写;原 if(false) 占位
    // SROUTE v3(docs/72 Phase 2):结构化字节路由,替代 v4 拟合门(sroute>0 时)
    else if (sroute > 0 && diter_thr > 0 && dup_ok) {
        int tgt_sr = tbl_target_rexp(e, tgt_alpha_pct, tbl_glo, tbl_ghi, tbl_elo, tbl_ehi, tbl_eref);
        int ht_row = WARP_SIZE; while (ht_row < tgt_sr && ht_row < HASH_CAP) ht_row <<= 1;
        long long hash_bytes = (long long)ht_row * HASH_BYTES_PER_SLOT_MILLI
            + (sroute >= 3 ? (long long)e * COMPACT_BYTES_PER_EST_MILLI : 0);
        long long dense_bytes = (long long)sl0 * DENSE_BYTES_PER_COL_MILLI
            + (sroute >= 4 ? (fl64 - e) * CONTENTION_BYTES_PER_DUP_MILLI / 1000 : 0);
        if (dense_bytes <= hash_bytes && flop[i] >= diter_thr &&
            (sroute < 2 || e >= EST_DENSE_MIN/2)) bid = BIN_DITER;
        else if (e > HASH_CAP) bid = BIN_HEAVY;
        else { int bi_sr = 0, ht_sr = WARP_SIZE; int t_sr = (e <= TABLE_DOUBLING_THR) ? (int)((long long)e * tgt_alpha_pct / 100) : e;
               while (ht_sr < t_sr && ht_sr < HASH_CAP) { ht_sr <<= 1; bi_sr++; }
               if (e <= BATCH_EST_MAX) { int rk = A_row_ptr[i+1]-A_row_ptr[i]; bid = (rk <= WARP_SIZE) ? ((batch_s_max && e <= batch_s_max) ? BIN_BATCH_S : 0) : bi_sr; }
               else bid = bi_sr; }
    }
    // v4 拟合门(默认路径;常数附 docs/72 结构推导注释 4194b45)
    else if (diter_thr > 0 && dup_ok &&
             ((flop[i] >= flop_floor && e >= est_floor &&           // 表字节门(docs/72;§22d 档内=env 松地板/超档=结构化 4096/2048)
               (dstruct_win <= 0 || sl0 <= dstruct_win) &&   // §21 span 门:>0 手调单窗;0/-1 不限
                                                          // §22b(09-04 定谳):-1 = 档位梯【数值侧】分流
                                                          // (≤最大档 → 档位梯数值;>最大档 → 老窗口 cursor 内核),
                                                          // 路由不再驱逐 —— c-64 19442 行 span 7K-50K 被 §22
                                                          // 驱逐回 hash → +95% 回归(老内核本就是它们的赢家)
               (long long)span_factor * flop[i] >= (long long)span_len[i])
              || (dim2 && flop[i] >= EST_DENSE_MIN && e <= HASH_CAP))
             && (!occgate || (long long)occ_coeff * e >= (long long)sl0)  // OCCGATE:est×coeff ≥ span(默认 16=同 spanf 门系数;OCCGATE_COEFF 可调)
             && tier_cmp_ok
             && ladder_dense_ok) bid = BIN_DITER;
    // HEAVY 候选行(e>CAP)无条件改判 DITER:全局内存原子(CAS ~600ns)比任何 dense
    // 窗扫描都贵——dup 门的"防 dense 浪费"逻辑在备选方案为 HEAVY 时不再适用。
    // SiO2:2373 行×1.74GB 全局表=3521ms;dense_direct 同人口 8.4ms=419× 差距。
    // study/54 §11(10-21)补经济学门:无条件改判对宽 span 行是灾难——para 族
    // span/est=8.0(窗扫 8× 浪费,dense 7.25ms vs heavy 2.2ms);c-64=2.5/SiO2 小(窗经济,
    // 关 diter 则 +8144%/+42469%)。门=span≤G×est(默认 G=4,实测分离 2.5 vs 8.0;
    // DITER_OVR_SPANEST=0 回旧无条件行为)。
    else if (diter_thr > 0 && e > HASH_CAP
             && (ovr_se <= 0 || (long long)span_len[i] <= (long long)ovr_se * e)) bid = BIN_DITER;
    else if (e > HASH_CAP) bid = BIN_HEAVY;          // heavy:全局表(2026-08-25,不再回退 merge)
    else {
        int bi = 0, ht = WARP_SIZE;
        // 小行表 2× 松弛(est=flop 紧界 → 满载原子争用,333SP accumulate +8ms 教训;槽位不变只放大表)
        int target = tbl_target_rexp(e, tgt_alpha_pct, tbl_glo, tbl_ghi, tbl_elo, tbl_ehi, tbl_eref);
        while (ht < target && ht < HASH_CAP) { ht <<= 1; bi++; }
        // 批量 kernel 门(2026-08-26):est≤64 且行长≤32(k 短,warp 串行 k 才划算;
        // 3Dspectralwave2 的 est 小但 k 数百的长链行回归教训)→ bin0(BATCH_HT=128 覆盖 est≤64);
        // 否则走原 per-row 梯。333SP 教训:门须独立于 2× 表目标的梯子(否则 est~36 被推到 bi2)
        if (false) {  // subwarp8 否决(germany_osm 49→55ms,warp idle 不是瓶颈;kernel 保留待查;dispatch 分支已删,勿再加在 BIN_HEAVY 前——曾挡死 heavy 导致 DNF)
            bid = BIN_DITER;
        } else if (e <= BATCH_EST_MAX) {
            int rk = A_row_ptr[i + 1] - A_row_ptr[i];
            bid = (rk <= WARP_SIZE) ? ((batch_s_max && e <= batch_s_max) ? BIN_BATCH_S : 0) : bi;
        } else if (e <= SMALL_SPAN_MAX && batch2) {   // docs/67 §8:est 65-256 小行批处理(web 图 22k cyc/行块地板税)
            int rk = A_row_ptr[i + 1] - A_row_ptr[i];
            bid = (rk <= WARP_SIZE) ? BIN_BATCH2 : bi;
        } else {
            bid = bi;
        }
    }
    // ── D3② Round2:PVD 学生派发保守安全域(study/36 §六)──
    //    Round1 教训:逐行成本+摊账捕获不了集体相位效应(pre2 2.48/c-64 14.9 崩盘)。
    //    处方 A:学生只在静态门【已判 diter/k7】的人口内改判(三路);纯 hash 行(梯 0-10)
    //    只允许 k7 抢走(L0 规则+学生),**永不进 BIN_DITER**——消灭集体 count 相位爆发源。
    //    micro bin(ESC/batch/ultra/sspan/largest/heavy)原路放行。
    if (d6_on && (bid <= 10 || bid == BIN_DITER || bid == BIN_K7)) {
        const int al = A_row_ptr[i + 1] - A_row_ptr[i];
        const int fl = flop[i], sp2 = span_len[i], e2 = est[i];
        bool k7_legal = (al >= 1 && al <= 32 && sp2 > 0 && sp2 <= 32768 && e2 >= 256
                         && (fl - e2) <= 512 && (double)fl <= 1.67 * (double)e2
                         && (long long)span_factor * fl >= (long long)sp2
                         && 16LL * e2 >= (long long)sp2);
        if (k7_legal && d6_ncols > 0 && (double)e2 >= 0.9 * (double)d6_ncols) {
            bid = BIN_K7;                                    // L0 全稠密规则
        } else {
            float x[21];                                     // 特征序=d3_distill_student.py FEATNAMES
            int t = (int)((double)e2 * 1.5); int s = 32; while (s < t && s < 65536) s <<= 1;
            x[0] = (float)al; x[1] = (float)fl; x[2] = (float)e2; x[3] = (float)sp2; x[4] = (float)s;
            x[5] = (float)((sp2 > 0) ? (sp2 + 5399) / 5400 : 1);   // W=ceil(span/PB2_W)
            x[6] = logf((float)al + 1.f); x[7] = logf((float)fl + 1.f);
            x[8] = logf((float)e2 + 1.f); x[9] = logf((float)sp2 + 1.f);
            x[10] = d6_cr; x[11] = d6_cs; x[12] = d6_cm;
            x[13] = (float)fl / fmaxf((float)e2, 1.f); x[14] = (float)e2 / fmaxf((float)sp2, 1.f);
            x[15] = (float)fl / fmaxf((float)sp2, 1.f);
            x[16] = (float)fl / fmaxf((float)al, 1.f); x[17] = logf(x[16] + 1.f);
            float mb = d6_maxbl ? (float)d6_maxbl[i] : 0.f;
            x[18] = mb; x[19] = logf(mb + 1.f); x[20] = (float)((fl > e2) ? fl - e2 : 0);
            int node = 0;
            while (stu_leaf[node] < 0)
                node = (x[stu_feat[node]] <= stu_thr[node]) ? stu_cl[node] : stu_cr[node];
            int pick = stu_leaf[node];                       // 0=hash 1=dense 2=k7
            // D4 出厂:R2+宽行护栏 v2(阈值=7277=DENSE_TIER_SPAN_MAX,物理出处)
            //   hash→k7 ✓(L0/学生)  hash→dense ✗(pre2 灭雷)
            //   diter→k7 ✓            diter→hash 仅 span≤7277(c-64 宽行护栏)
            if (pick == 2 && k7_legal) {
                bid = BIN_K7;
            } else if (pick == 1 && (bid == BIN_DITER || bid == BIN_K7)
                       && (sp2 <= 7277 || x[16] >= 64.f)) {
                bid = BIN_DITER;                             // 学生判 dense,静态已在 dense 基础设施
            } else if (pick == 0 && (bid == BIN_DITER || bid == BIN_K7)
                       && sp2 <= 7277) {
                // 学生判 hash:只允许 span≤7277(档位梯界)回梯;宽行保持 diter(c-64 护栏)
                if (e2 > HASH_CAP) bid = BIN_HEAVY;
                else {
                    int bi2 = 0, ht2 = WARP_SIZE;
                    int tgt2 = tbl_target_rexp(e2, tgt_alpha_pct, tbl_glo, tbl_ghi, tbl_elo, tbl_ehi, tbl_eref);
                    while (ht2 < tgt2 && ht2 < HASH_CAP) { ht2 <<= 1; bi2++; }
                    if (e2 <= BATCH_EST_MAX && al <= WARP_SIZE && batch_s_max && e2 <= batch_s_max) bid = BIN_BATCH_S;
                    else bid = bi2;
                }
            }
            // else:pick 0/1 但静态已在 hash(梯 0-10)→ 保持;micro bin 不改判
        }
    }
    // P0-1 修复(10-14,study/43 审计):强制改判必须在 bucket_id 写入【之前】——
    // 原序 counts 按 BIN_DITER 计数而 scatter 按原 bid 落位,两套口径错位 → 段溢出。
    if (force_est_min > 0 && est[i] >= force_est_min) bid = BIN_DITER;   // 真强制(决赛 v2):终判处覆盖一切门与 BATCH/HEAVY 拦截层
    bucket_id[i] = bid;
    // warp 聚合:同 warp 内同 bin 的行由 leader 一次 atomicAdd(3.7M 行×12 计数器争用 → 333SP 2.9ms 教训)
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, (unsigned)bid);
    int leader = __ffs(peers) - 1;
    if ((int)(threadIdx.x & 31) == leader) atomicAdd(&counts[bid], __popc(peers));
}

// scatter:把行号按 bucket 写到预分配大 buffer 的对应位置
// 1c-②a(10-14,审计弹药①安全面):binning 链双计数器(d_cnt/d_pos)清零合一
// (原为两次 cudaMemset,每次一个 driver 往返;N_BINS=16×2 个 int,单线程一发足够)。
static __global__ void binning_zero2_kernel(int *cnt, int *pos, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { cnt[i] = 0; pos[i] = 0; }
}

static __global__ void scatter_rows_kernel(
    const int *bucket_id, int A_rows,
    const int *offsets, int *pos, int *sorted_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int bid = bucket_id[i];
    // warp 聚合取槽:leader 一次 atomicAdd(popc),成员按 warp 内 rank 分配(0.65→聚合后原子减 ~32×)
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, (unsigned)bid);
    int leader = __ffs(peers) - 1;
    int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == leader) base = atomicAdd(&pos[bid], __popc(peers));
    base = __shfl_sync(mask, base, leader);
    int rank = __popc(peers & ((1u << lane) - 1));    // 同 bid 组内低于自己的 lane 数
    sorted_rows[offsets[bid] + base + rank] = i;
}

// 单 block inclusive prefix-sum(Hillis-Steele);小阵专用,替 thrust::inclusive_scan
// T = int(cnnz)/ long long(est:ocean337 Σest 超 2^31)
template <typename T>
__global__ void scan_inclusive_kernel(const int *x, T *y, int n) {
    extern __shared__ unsigned char scan_smem_raw[];
    T *ss = (T*)scan_smem_raw;
    int tid = threadIdx.x;
    int N = blockDim.x;
    ss[tid] = (tid < n) ? (T)x[tid] : (T)0;
    __syncthreads();
    for (int off = 1; off < N; off <<= 1) {
        T v = (tid >= off) ? ss[tid - off] : (T)0;
        __syncthreads();
        ss[tid] += v;
        __syncthreads();
    }
    if (tid < n) y[tid] = ss[tid];
}

// 小行专用:accumulate 已在 SMEM 内 count-sort 写有序,这里只把 tmp(gapped)→ CSR(packed) 纯 copy。
static __global__ void hash_compact_copy_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    if (skip && skip[row]) return;               // 行级重试的行:数据在重试区,主区跳过
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// warp-per-row 批量 copy(小行,2026-08-26):bin0 的 ~3M 行×19 项,1CTA/行 launch-bound → 8 行/CTA
__device__ __forceinline__ int largest_lower_bound(const int *a, int lo, int hi, int v) {
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
}

// Ocean-D largest 数值内核(docs/74):straggler 行(密度 1/16~1/80,SMEM-val dense 与 hash 都亏)的
// 第三形态 —— 位图+前缀驻 SMEM(0.25B/列,单窗 span 上限 ≈ 800K),val 经 tmp 全局流过(atomicAdd)。
// 输出 = tmp_key(低 32 列)/tmp_val + row_nnz,与 hash 行同消费(compact 零改动);
// est 欠估 → out > est 预算 → 推 ovf_rows 走标准 retry(全局 hash)。
static __global__ void dense_largest_numeric_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols, const int *est,
    const long long *row_off, unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int lo0 = span_lo[i], span = span_len[i];
    int rs = A_rp[i], re = A_rp[i + 1];
    int a_len = re - rs;
    // G-group(Phase 1b 同推导):两遍乘积(OR 与 补值)共用组布局
    int G = WARP_SIZE;
    {
        int est_flop = span > 0 ? a_len * (span / 32 > 0 ? span / 32 : 1) : a_len;
        int avg_B = a_len > 0 ? est_flop / a_len : 1;
        if (avg_B > 1) {
            int logG = 31 - __clz((unsigned)avg_B);
            if (logG < 1) logG = 1;
            if (logG > ilog2c(WARP_SIZE)) logG = ilog2c(WARP_SIZE);
            G = 1 << logG;
        }
    }
    int num_groups = blockDim.x / G, my_group = tid / G, my_id = tid % G;
    extern __shared__ __align__(8) unsigned int lsmem[];   // [words] 位图 | [words] 前缀
    unsigned int *bmp = lsmem;
    unsigned int *pref = lsmem + ((win_cols + 31) >> 5);
    long long base = row_off[i];
    long long budget = (long long)est[i];
    long long out = 0;
    bool ovf = false;
    int nw_win = (span + win_cols - 1) / win_cols;
    for (int wi = 0; wi < nw_win && !ovf; wi++) {
        int c0 = lo0 + wi * win_cols;
        int c1 = min(c0 + win_cols, lo0 + span);
        int w = c1 - c0;
        int words = (w + 31) >> 5;
        for (int j4 = tid; j4 < words; j4 += blockDim.x) bmp[j4] = 0;
        __syncthreads();
        for (int p = rs + my_group; p < re; p += num_groups) {   // pass1:位图 OR
            int k = A_ci[p];
            int ks = B_rp[k], ke = B_rp[k + 1];
            int from = (nw_win == 1) ? ks : largest_lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : largest_lower_bound(B_ci, from, ke, c1);
            for (int q = from + my_id; q < to; q += G) {
                int col = B_ci[q];
                if (upper_tri && col < i) continue;
                int j = col - c0;
                atomicOr(&bmp[j >> 5], 1u << (j & 31));
            }
        }
        __syncthreads();
        __shared__ int w_tot;
        if (warp == 0) {   // pass2:字级 popcount 前缀(Phase 1b 同款)
            int seglen = (words + 31) / 32;
            int lo = lane * seglen, hi = min(lo + seglen, words);
            int run = 0;
            for (int j = lo; j < hi; j++) { pref[j] = run; run += __popc(bmp[j]); }
            int excl = run;
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, excl, off);
                if (lane >= off) excl += v;
            }
            for (int j = lo; j < hi; j++) pref[j] += excl - run;
            if (lane == 31) w_tot = excl;
        }
        __syncthreads();
        if (out + w_tot > budget) {   // est 欠估:tmp 区会越界 → 交 retry(全局 hash)
            ovf = true;
            break;
        }
        for (int p = rs + my_group; p < re; p += num_groups) {   // pass3:补值 → 精确偏移直写 tmp
            int k = A_ci[p];
            double a = A_val[p];
            int ks = B_rp[k], ke = B_rp[k + 1];
            int from = (nw_win == 1) ? ks : largest_lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : largest_lower_bound(B_ci, from, ke, c1);
            for (int q = from + my_id; q < to; q += G) {
                int col = B_ci[q];
                if (upper_tri && col < i) continue;
                int j = col - c0;
                int wo = j >> 5, b = j & 31;
                unsigned below = b ? (bmp[wo] & ((1u << b) - 1)) : 0u;
                long long off = (long long)pref[wo] + __popc(below);
                tmp_key[base + out + off] = (unsigned long long)(unsigned)(c0 + j);   // 低 32 位 = 列(compact 同消费)
                atomicAdd(&tmp_val[base + out + off], a * B_val[q]);
            }
        }
        out += w_tot;
        __syncthreads();
    }
    if (tid == 0) {
        if (ovf) {
            if (!atomicExch(&row_ovf[i], 1)) { int q3 = atomicAdd(ovf_cnt, 1); ovf_rows[q3] = i; }
            atomicExch(overflow_flag, 1);
        } else {
            row_nnz[i] = (int)out;
        }
    }
}

// v31-④ ESC 零表短行内核(docs/76;Ocean usparse 同位,机制 = 我们 csort 的位次计数前移):
// flop ≤ ESC_FLOP_MAX 的行,每 subwarp(=WARP_SIZE)一行、每线程 ≤1 乘积;
// 位次 = popc(比我小的 key),同 key 只 leader 写全和 → 零表零清零,输出天然有序
// (compact 免 csort)。乘积定位 = 逐链计数 O(rk)(空 B 链自然跳过)。溢出守卫同 largest。
template<int SUBW>
__global__ void hash_esc_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int sw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / SUBW);   // subwarp = SUBW(v2 按行宽分档)
    int t = threadIdx.x & (SUBW - 1);
    if (sw >= n_in_bucket) return;
    int i = bucket_rows[sw];
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];

    // ① 每线程定位第 t 个乘积(逐链计数;哨兵 = 无乘积)
    int my_key = 0x7fffffff; double my_val = 0.0;
    {
        int cnt = 0;
        for (int p = rs; p < re && my_key == 0x7fffffff; p++) {
            int k = A_col_idx[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            int len = ke - ks;
            if (cnt <= t && t < cnt + len) {
                int q = ks + (t - cnt);
                int col = B_col_idx[q];
                if (!(upper_tri && col < i)) {
                    my_key = col; my_val = A_val[p] * B_val[q];
                } else { my_key = -1; my_val = 0.0; }   // 过滤项:哨兵(不参与)
                break;
            }
            cnt += len;
        }
        // 注:t≥flop 或被过滤的线程保持哨兵
    }
    __syncwarp();

    // ② counting-sort fused(同 csort/Ocean countSortCompactFused):
    //    32 轮 shfl 直扫:数"比我小的有效 key"= 位次;同 key 只最小 lane 当 leader 持全和
    double sum = my_val;
    unsigned smaller = 0u; bool leader = true;
    // ⚠ 子组 mask 必须按 lane 基址对齐(v2 首版 0xFF 全组共用 = lane8-15 非法 → road/web-G nnz 红旗)
    const unsigned full_mask = ((SUBW == WARP_SIZE) ? 0xffffffffu : ((1u << SUBW) - 1u))
                               << (threadIdx.x & 31 & ~(SUBW - 1));
    const int sub_base = threadIdx.x & 31 & ~(SUBW - 1);   // ⚠ shfl 的源 lane 是【绝对】编号:
    #pragma unroll                                                      // 子组 8-15 曾读 0-7 = 别人的行 → 跨行键污染超计
    for (int k = 0; k < SUBW; k++) {                                    // (ESC32 的 k=0..31 恰=绝对 → bcsstk30 掩护)
        int ok = __shfl_sync(full_mask, my_key, sub_base + k);
        double ov = __shfl_sync(full_mask, my_val, sub_base + k);
        if (ok == 0x7fffffff || ok == -1) continue;          // 哨兵/过滤项
        if (ok < my_key) smaller |= (1u << k);
        else if (ok == my_key) { if (k != t) sum += ov; if (k < t) leader = false; }   // k==t 是自己:sum 已含 my_val,再 += = 自加倍(v31-④ 起全量值 2×/1.5× 雷,study/18;单键 2.0×,双键 v1≈v2 时 1.5×)
    }
    __syncwarp();
    unsigned leaders = __ballot_sync(full_mask, leader && my_key != 0x7fffffff && my_key != -1);
    // ⚠ ballot 返回全局 lane 位,smaller 是子组局部位 —— 必须移位对齐(ESC8 的 2/3/4 号
    //   子组曾因此位空间错位 AND 出垃圾位 → road 超计;ESC32 全 warp 恰重合所以 bcsstk30 对)
    leaders >>= sub_base;
    smaller &= leaders;   // 只数"更小且是 leader"的 lane(Ocean countSortCompactFused 的 bitmap &= leadings_bitmap)
    int n_out = __popc(leaders);
    if (n_out > est[i]) {   // est 欠估 → 标准 retry 兜底
        if (t == 0) {
            if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
            atomicExch(overflow_flag, 1);
        }
        return;
    }
    if (leader && my_key != 0x7fffffff && my_key != -1) {
        int pos = __popc(smaller);   // 有效中比我小的个数 = 精确位次(天然有序)
        long long base = row_off[i];
        tmp_key[base + pos] = (unsigned long long)(unsigned)my_key;   // 低 32 = 列(compact 同消费)
        tmp_val[base + pos] = sum;                                    // leader 持全和,免 atomicAdd
    }
    if (t == 0) row_nnz[i] = n_out;
}


// ESC64(study/50 靶1,10-20):flop∈(32,64] 的行——1 warp/行、每线程 2 乘积(IPT=2)。
// 机理同 hash_esc_kernel 的零表 counting-sort,差别:64 虚拟 lane → smaller 位图 64bit(ull),
// 每线程两套 (key,sum,leader) 状态;A 槽(虚拟 t)恒先于 B 槽(虚拟 t+32),同键时 A 领导。
static __global__ void hash_esc64_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int w = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int t = threadIdx.x & 31;
    if (w >= n_in_bucket) return;
    int i = bucket_rows[w];
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];

    // ① 每线程装第 t(A 槽)与 t+32(B 槽)个乘积;哨兵 0x7fffffff,过滤项 -1
    int kA = 0x7fffffff, kB = 0x7fffffff; double vA = 0.0, vB = 0.0;
    {
        int needA = t, needB = t + 32, cnt = 0;
        for (int p = rs; p < re && (needA != INT_SENTINEL || needB != INT_SENTINEL); p++) {
            int k = A_col_idx[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            int len = ke - ks;
            if (needA != INT_SENTINEL && needA >= cnt && needA < cnt + len) {
                int col = B_col_idx[ks + (needA - cnt)];
                if (upper_tri && col < i) { kA = -1; } else { kA = col; vA = A_val[p] * B_val[ks + (needA - cnt)]; }
                needA = INT_SENTINEL;
            }
            if (needB != INT_SENTINEL && needB >= cnt && needB < cnt + len) {
                int col = B_col_idx[ks + (needB - cnt)];
                if (upper_tri && col < i) { kB = -1; } else { kB = col; vB = A_val[p] * B_val[ks + (needB - cnt)]; }
                needB = INT_SENTINEL;
            }
            cnt += len;
        }
    }
    __syncwarp();

    // ② counting-sort fused:32 轮 shfl,每轮广播 2 键(A 槽=虚拟 k,B 槽=虚拟 k+32)
    double sumA = vA, sumB = vB;
    unsigned long long smallerA = 0ull, smallerB = 0ull;
    bool leadA = true, leadB = true;
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int okA = __shfl_sync(0xffffffffu, kA, k);
        double ovA = __shfl_sync(0xffffffffu, vA, k);
        int okB = __shfl_sync(0xffffffffu, kB, k);
        double ovB = __shfl_sync(0xffffffffu, vB, k);
        // 守卫=【我自己】的键有效(n23 超计根因:旧版用广播方 lane k 的 B 有效性(vB_)做外层门,
        // 当 k 无 B 槽时整块被跳过——连本该触发的 A-B 夺领导也跳过 → A/B 同键双双当 leader,超计 ~4/行;
        // 独立复现件 /tmp/esc64_fix.cu 定谎,22/22 对齐)
        const bool myA = kA != 0x7fffffff && kA != -1, myB = kB != 0x7fffffff && kB != -1;
        // —— 对我的 A 键(虚拟 t;B 槽同键恒不夺领导[t+32>t],但其值要被领导者吸收)——
        if (myA) {
            if (okA < kA) smallerA |= (1ull << k);
            else if (okA == kA && k != t) { sumA += ovA; if (k < t) leadA = false; }
            if (okB == kA) sumA += ovB;   // 哨兵恒不等,无需 vB_ 守卫
        }
        // —— 对我的 B 键(虚拟 t+32;A 槽同键恒夺领导[k<t+32];广播 A 的哨兵 0x7fffffff 恒不等于 kB)——
        if (myB) {
            if (okA < kB) smallerB |= (1ull << k);
            if (okA == kB) { sumB += ovA; leadB = false; }
            if (okB < kB) smallerB |= (1ull << (k + 32));
            else if (okB == kB && k != t) { sumB += ovB; if (k < t) leadB = false; }   // B-B 同键:低 lane 夺领导
        }
    }
    __syncwarp();
    unsigned lA = __ballot_sync(0xffffffffu, leadA && kA != 0x7fffffff && kA != -1);
    unsigned lB = __ballot_sync(0xffffffffu, leadB && kB != 0x7fffffff && kB != -1);
    unsigned long long leaders64 = (unsigned long long)lA | ((unsigned long long)lB << 32);
    smallerA &= leaders64;   // 只数"更小且是 leader"的虚拟 lane(同 32 档语义)
    smallerB &= leaders64;
    int n_out = __popcll(leaders64);
    if (n_out > est[i]) {   // est 欠估 → 标准 retry 兜底(同 32 档)
        if (t == 0) {
            if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
            atomicExch(overflow_flag, 1);
        }
        return;
    }
    long long base = row_off[i];
    if (leadA && kA != 0x7fffffff && kA != -1) {
        int pos = __popcll(smallerA);
        tmp_key[base + pos] = (unsigned long long)(unsigned)kA;
        tmp_val[base + pos] = sumA;
    }
    if (leadB && kB != 0x7fffffff && kB != -1) {
        int pos = __popcll(smallerB);
        tmp_key[base + pos] = (unsigned long long)(unsigned)kB;
        tmp_val[base + pos] = sumB;
    }
    if (t == 0) row_nnz[i] = n_out;
}

// ── T3(study/53 §T3 设计简报,10-21):NT=64/128 子组秩排序 ──────────────────────
// 机理 = ocean countSortCompactFused(AccumulatorESC.cuh:71-131)的 smem 广播形态:
// 每线程 1 乘积 → smem 暂存 → O(N²) 全对扫描(同键聚值+首现 leader+更小键计秩)
// → leader 按秩直写有序输出。零原子零表零探测。
// 与 hash_esc_kernel(shfl 版,SUBW≤32 硬限)的差异:全对扫描走 smem 广播读
// (LDS broadcast 每槽 N 读廉价),leader 用"∃j<t 同键"判定(免跨 warp ballot);
// 与 hash_esc64(shfl 双键,study/50 证伪)的差异:无 64bit shfl。
template<int NT>
__global__ void esc_rank_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    constexpr int SGRP = HASH_BLOCK_STRUCT / NT;          // 子组/块(64→4,128→2)
    const int sg = threadIdx.x / NT, t = threadIdx.x % NT;
    constexpr int NTP = NT + 4;   // +4 垫:64 槽=2 圈银行,跨子组同 k 步冲突
    __shared__ int s_key[SGRP][NTP];
    __shared__ double s_val[SGRP][NTP];
    __shared__ unsigned char s_dup[SGRP][NTP];
    __shared__ int s_nout[SGRP];
    const int stride = gridDim.x * SGRP;                   // 持久化行循环(ocean dispatcher 形态:
    for (int row = blockIdx.x * SGRP + sg; row < n_in_bucket; row += stride) {   // 块数~2×SM,行循环复用缓存/摊调度)
    const int i = bucket_rows[row];

    int my_key = 0x7fffffff; double my_val = 0.0;
    {
        int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
        int cnt = 0;                                       // ① 链式定位第 t 个乘积(同 hash_esc ①)
        for (int p = rs; p < re && my_key == 0x7fffffff; p++) {
            int k = A_col_idx[p];
            int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
            int len = ke - ks;
            if (cnt <= t && t < cnt + len) {
                int q = ks + (t - cnt);
                int col = B_col_idx[q];
                if (!(upper_tri && col < i)) { my_key = col; my_val = A_val[p] * B_val[q]; }
                else my_key = -1;
                break;
            }
            cnt += len;
        }
    }
    s_key[sg][t] = my_key; s_val[sg][t] = my_val;
    __syncthreads();

    bool leader = my_key != 0x7fffffff && my_key != -1;
    {   // ② 首现判定:∃j<t 同键则非 leader。⚠ 统一步数(k<NT 全员同趟)+谓词(k<t)——
        // 首版 k<t 步数 = warp 内每人不同 trip = 大分歧(8.09ms 2× 慢的根因)
        #pragma unroll 4
        for (int k = 0; k < NT; k++)
            if (k < t && s_key[sg][k] == my_key) leader = false;
    }
    s_dup[sg][t] = leader ? 0 : 1;
    __syncthreads();

    if (t == 0) {                                          // ③ n_out(leader 数)+ est 欠估 retry 兜底
        int n = 0;
        for (int k = 0; k < NT; k++) if (!s_dup[sg][k]) n++;
        s_nout[sg] = n;
        if (n > est[i]) {
            if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
            atomicExch(overflow_flag, 1);
        }
    }
    __syncthreads();
    if (!s_dup[sg][t]) {                            // ④ leader:聚值+计秩+直写
        double sum = 0.0; int rank = 0;
        #pragma unroll 4
        for (int k = 0; k < NT; k++) {
            int ok = s_key[sg][k];
            if (ok == 0x7fffffff || ok == -1) continue;
            if (ok == my_key) sum += s_val[sg][k];
            else if (ok < my_key && !s_dup[sg][k]) rank++;
        }
        long long base = row_off[i];
        tmp_key[base + rank] = (unsigned long long)(unsigned)my_key;   // 低 32 = 列(compact 同消费)
        tmp_val[base + rank] = sum;
        if (t == 0) row_nnz[i] = s_nout[sg];
    }
    __syncthreads();   // 行循环边界:下一行的 scratch 写前,本轮读全退(数据依赖栅)
    }   // 行循环结束
}

// compact 小行线程版(docs/13 v2):行内串行拷贝零闲 lane(warp 版 5-nnz 行 27/32 lane 闲;
// road 族 compact 10.6ms/24M 行 = 440ns/行,CTA 数 3M → 94K)
static __global__ void hash_compact_copy_thread_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_rows) return;
    int row = rows[t];
    if (skip && skip[row]) return;
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    for (int i = 0; i < n; i++) {
        out_col[dst + i] = (int)(tmp_key[src + i] & 0xffffffffu);
        out_val[dst + i] = tmp_val[src + i];
    }
}

static __global__ void hash_compact_copy_warp_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    int row = rows[warp];
    if (skip && skip[row]) return;
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    for (int t = lane; t < n; t += 32) {
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// 刀C1(study/57 tick6):4 路独立装载展开(每 lane 在飞 8 载 vs 2)。
// nsys 实测 cage15 copy_warp 17.1ms/迭代 = 1.20TB/s(峰值 43%)——延迟遮蔽不足,非带宽墙。
// 语义零变(同序同值);COMPACT_COPY_V2=1 启用,默认 0 = 原核逐位。
static __global__ void hash_compact_copy_warp_v2_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    int row = rows[warp];
    if (skip && skip[row]) return;
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    int t = lane;
    for (; t + 128 <= n; t += 128) {          // 4 段 × 32 lane,装载全部独立发射后再写
        unsigned long long k0 = tmp_key[src + t];
        unsigned long long k1 = tmp_key[src + t + 32];
        unsigned long long k2 = tmp_key[src + t + 64];
        unsigned long long k3 = tmp_key[src + t + 96];
        double v0 = tmp_val[src + t];
        double v1 = tmp_val[src + t + 32];
        double v2 = tmp_val[src + t + 64];
        double v3 = tmp_val[src + t + 96];
        out_col[dst + t] = (int)(k0 & 0xffffffffu);
        out_col[dst + t + 32] = (int)(k1 & 0xffffffffu);
        out_col[dst + t + 64] = (int)(k2 & 0xffffffffu);
        out_col[dst + t + 96] = (int)(k3 & 0xffffffffu);
        out_val[dst + t] = v0;
        out_val[dst + t + 32] = v1;
        out_val[dst + t + 64] = v2;
        out_val[dst + t + 96] = v3;
    }
    for (; t < n; t += 32) {                  // 尾段(<128)回标量
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// ===== P1 G1(study/58 附录A.1 v2):compact/csort host 助手区 = TU-90 独占 =====
// TU-61 编译这些发射器会实例化未打签 csort 模板 → 探针P2 弱合并雷(静默错 codegen)。
// 区内全为 compact 侧家族,TU-61 accumulate 梯零引用(已核函数地图 4449-4880)。
#if HASH_TB != 61
// 刀C1 发射 helper:5 个调用点统一走此门(env 读一次;默认 0 = 原核逐位)。
static void launch_copy_warp(int n_rows, const int *rows_ptr,
                             const long long *d_off, const int *d_row_nnz, const int *dC_rp,
                             const unsigned long long *d_tmp_key, const double *d_tmp_val,
                             int *dC_ci, double *d_val, const int *d_row_ovf)
{
    static int g_ccv2 = -1;
    if (g_ccv2 < 0) { const char *e = getenv("COMPACT_COPY_V2"); g_ccv2 = (e && *e) ? atoi(e) : 0; }
    int grid = (n_rows + WPR_COPY_ROWS - 1) / WPR_COPY_ROWS;
    if (g_ccv2)
        hash_compact_copy_warp_v2_kernel<<<grid, HASH_BLOCK_STRUCT>>>(
            rows_ptr, n_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
    else
        hash_compact_copy_warp_kernel<<<grid, HASH_BLOCK_STRUCT>>>(
            rows_ptr, n_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
    CHECK_CUDA(cudaGetLastError());
}

// COMPACT_FLAT 条目覆盖计数(wb-edu 教训 10-19:行数比例门漏了大 C_nnz 低覆盖阵——
// 行过半但条目占少数时全量条目线程空转二分,wb-edu +109%/nd24k +56% 实测;改按条目比例放行)
static __global__ void k_count_copy_entries(const int *bkid, const int *row_nnz, int n, int *out, int flatb = 0)
{
    int c = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int b = __ldg(bkid + i);
        if (b == 0 || b == 19 || (b >= 21 && b <= 24)) c += __ldg(row_nnz + i);
    else if (flatb && b >= 2 && b <= 6) c += __ldg(row_nnz + i);   // T4d(FLATB)
    }
    __shared__ int red[256];
    red[threadIdx.x] = c; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, red[0]);
}
static __global__ void hash_compact_copy_flat_kernel(
    int total_entries, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip, const int *bkid, int flatb = 0)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total_entries) return;
    // 二分:最大 row_ptr[e] == e 的行(闭开区间上界)
    int lo = 0, hi = n_rows;              // 不变式:row_ptr[lo] <= e < row_ptr[hi](row_ptr[n_rows]=total)
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(row_ptr + mid) <= e) lo = mid; else hi = mid;
    }
    int row = lo;
    if (skip && __ldg(skip + row)) return;
    if (bkid) {   // 仅拷贝桶:bin0 / BIN_BATCH_S(19) / ESC8-32(21-23),与循环显式拷贝分支一一对应;
                  // FLATB=1 扩 bins2-6(T4d:这些行 accumulate 已 count-sort 出有序 tmp,compact 只剩拷贝)
        int b = __ldg(bkid + row);
        bool ok = (b == 0 || b == 19 || (b >= 21 && b <= 24));
        if (!ok && flatb) ok = (b >= 2 && b <= 6);   // T4d(FLATB)
        if (!ok) return;
    }
    int idx = e - __ldg(row_ptr + row);
    long long src = __ldg(row_off + row) + idx;
    out_col[e] = (int)(__ldg(tmp_key + src) & 0xffffffffu);
    out_val[e] = __ldg(tmp_val + src);
}

// D5H 原地行排序(MODE=2 大行:无序直写 dC 后,行内 BlockRadixSort col/val in==out)
template<int TPB, int IPT>
__global__ void csort_inplace_kernel(const int *rows, const int *rp, int *col, double *val,
                                    const int *skip = nullptr) {
    int r = rows[blockIdx.x];
    if (skip && skip[r]) return;   // 刀14:溢出行数据在 retry 区,跳过
    int s = rp[r], n = rp[r + 1] - rp[r];
    if (n <= 1) return;
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT, double>;
    extern __shared__ char cip_smem[];
    unsigned kc[IPT]; double kv[IPT];
    using BTmp = typename BRS::TempStorage;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        kc[i] = (idx < n) ? (unsigned)col[s + idx] : 0xffffffffu;
        kv[i] = (idx < n) ? val[s + idx] : 0.0;
    }
    __syncthreads();
    BRS(reinterpret_cast<BTmp&>(*cip_smem)).Sort(kc, kv);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n) { col[s + idx] = (int)kc[i]; val[s + idx] = kv[i]; }
    }
}
template<int TPB, int IPT>
static void launch_csort_ip(int n, const int *rows_ptr, const int *d_rp, int *d_col, double *d_val,
                            const int *d_skip = nullptr) {
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT, double>;
    size_t smem = sizeof(typename BRS::TempStorage);
    if (smem > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)csort_inplace_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    csort_inplace_kernel<TPB, IPT><<<n, TPB, smem>>>(rows_ptr, d_rp, d_col, d_val, d_skip);
    CHECK_CUDA(cudaGetLastError());
}

// 融合键排序(docs/38,Ocean sortOutputFused):键 = (src_pos << valid_bits) | col,radix 只扫
// col 位;值不进排序流 —— 先灌 SMEM,排序后按键内 src_pos gather。省一半排序交通。
// 约束:n < 2^valid_bits 且 CAP=TPB×IPT ≤ 2^(32−valid_bits)(大 n 回退旧双数组内核)。
template<int TPB, int IPT>
__global__ void csort_fused_kernel(
    const int *rows, const long long *d_off, const int *d_row_nnz, const int *dC_rp,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *dC_ci, double *dC_val, int valid_bits)
{
    constexpr int CAP = TPB * IPT;
    int r = rows[blockIdx.x];
    long long s = d_off[r];
    int n_el = d_row_nnz[r];
    int out = dC_rp[r];
    if (n_el == 0) return;
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT>;
    extern __shared__ __align__(16) char cfs[];
    union CFU { typename BRS::TempStorage sort; double vals[CAP]; };
    CFU &u = reinterpret_cast<CFU&>(*cfs);
    unsigned k[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) {
            unsigned col = (unsigned)(tmp_key[s + idx] & 0xffffffffu);
            k[i] = ((unsigned)idx << valid_bits) | col;  // src_pos 高位免费搭车
        } else {
            k[i] = 0xffffffffu;
        }
    }
    __syncthreads();
    BRS(u.sort).Sort(k, /*begin_bit=*/0, /*end_bit=*/valid_bits);   // 只扫 col 位
    __syncthreads();
    // 值在排序【后】装 SMEM(union 区排序时被 TempStorage 占用,Ocean 同款时序)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) u.vals[idx] = tmp_val[s + idx];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) {
            unsigned key = k[i];
            unsigned col = key & ((1u << valid_bits) - 1u);
            int src = (int)(key >> valid_bits);
            dC_ci[out + idx] = (int)col;
            dC_val[out + idx] = u.vals[src];           // 值按源位 gather
        }
    }
}
template<int TPB, int IPT>
static void launch_csort_fused(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                               const int *dC_rp, const unsigned long long *k, const double *v,
                               int *dC_ci, double *d_val, int valid_bits) {
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT>;
    constexpr int CAP = TPB * IPT;
    size_t smem = std::max(sizeof(typename BRS::TempStorage), (size_t)CAP * sizeof(double));
    if (smem > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)csort_fused_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    csort_fused_kernel<TPB, IPT><<<n, TPB, smem>>>(rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, d_val, valid_bits);
    CHECK_CUDA(cudaGetLastError());
}

// ── 队列②(counting-vs-radix 交叉实测):bin6 csort 两级计数排序变体,CSORT_CNT=1 启用(默认关)──
// 靶:F2 compact+sort 2.48ms 主载体 = bin6 47.9K 行 per-row cub radix(3-4 pass 扫 col 位)→
// counting 1 pass + 直方图 + 桶内 O(b²)。算法 = spec 81 extract2 守门验证同款(含两条尸检修正:
// SMEM 区段字节核算 / 前缀连续所有权)。SMEM = sc 8K + sv 16K + hist 12K = 36KB(<48K 免 opt-in)。
// I/O 契约同 csort_fused:溢出行只排前 CAP 写废(retry 善后覆写);键 distinct(hash 去重后)→ rank 唯一。
// 差异:无需 valid_bits/A_cols 约束(无融合键位打包)。
#define CSC_TPB 256
#define CSC_CAP 2048
template<int TPB, int CAP, int NB = 1024>   // 刀16c:NB=桶数上限(旧 1024;Pass4 Σb² 随 d² 增长 → 大 d 行升桶打平)
__global__ void csort_counting_kernel(
    const int *rows, const long long *d_off, const int *d_row_nnz, const int *dC_rp,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *dC_ci, double *dC_val)
{
    constexpr int STG = CAP / TPB;         // 寄存器 staging 槽/线程(m ≤ CAP = STG×TPB 恒覆盖)
    int r = rows[blockIdx.x];
    long long s = d_off[r];
    int n_el = d_row_nnz[r];
    int out = dC_rp[r];
    if (n_el == 0) return;
    int m = min(n_el, CAP);
    int tid = threadIdx.x;
    extern __shared__ __align__(8) int ccsm[];
    int *sc = ccsm;                        // [CAP] 列键
    double *sv = (double*)(ccsm + CAP);    // [CAP] 值(CAP 偶 → 8B 对齐;占 2×CAP int 槽)
    int *h_cnt2  = ccsm + 3 * CAP;         // [NB] 桶计数(⚠ 3×CAP = sc+sv 字节界,2×CAP 会砸 sv)
    int *h_start = h_cnt2 + NB;            // [NB] 排他桶起点
    int *h_fill  = h_start + NB;           // [NB] 落位游标
    __shared__ int csc_seg[64];            // min/max 对 + 段扫 scratch(≤32 warp)
    for (int k = tid; k < m; k += TPB) {
        sc[k] = (int)(tmp_key[s + k] & 0xffffffffu);
        sv[k] = tmp_val[s + k];
    }
    __syncthreads();
    // Pass0:行 min/max(warp 归约 + 跨 warp 收集)
    int lo_l = 0x7fffffff, hi_l = -1;
    for (int k = tid; k < m; k += TPB) { lo_l = min(lo_l, sc[k]); hi_l = max(hi_l, sc[k]); }
    for (int off = 16; off; off >>= 1) {
        lo_l = min(lo_l, __shfl_down_sync(0xffffffff, lo_l, off));
        hi_l = max(hi_l, __shfl_down_sync(0xffffffff, hi_l, off));
    }
    if ((tid & 31) == 0) { csc_seg[(tid >> 5) << 1] = lo_l; csc_seg[((tid >> 5) << 1) | 1] = hi_l; }
    __syncthreads();
    if (tid == 0) {
        int L = 0x7fffffff, H = -1;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) {
            L = min(L, csc_seg[w << 1]); H = max(H, csc_seg[(w << 1) | 1]);
        }
        csc_seg[0] = L; csc_seg[1] = H;
    }
    __syncthreads();
    int lo = csc_seg[0], hi = csc_seg[1];
    // Pass1:直方图(桶 (c-lo)>>SH,SH 逐行自适应 → 桶数 ≤NB;聚簇行不退化)
    constexpr int LOGB = 31 - __builtin_clz(NB);   // NB 为 2 的幂
    unsigned span = (unsigned)(hi - lo + 1);
    int sbits = 32 - __clz(span - 1);      // ceil_log2(span);span=1 → 0(__clz(0)=32)
    int SH = sbits > LOGB ? sbits - LOGB : 0;
    int nbins = (int)((span - 1) >> SH) + 1;
    for (int b = tid; b < nbins; b += TPB) { h_cnt2[b] = 0; h_fill[b] = 0; }
    __syncthreads();
    for (int k = tid; k < m; k += TPB) atomicAdd(&h_cnt2[(sc[k] - lo) >> SH], 1);
    __syncthreads();
    // Pass2:排他前缀(连续所有权!跨步持有配段基址公式不成立——spec81 尸检第二雷)
    int ib = (nbins + TPB - 1) / TPB;
    int b_lo = tid * ib, b_hi = min(b_lo + ib, nbins);
    int run = 0;
    for (int b = b_lo; b < b_hi; b++) { run += h_cnt2[b]; h_start[b] = run; }
    int tot_t = run, exc_w = run;
    for (int off = 16; off; off >>= 1) {
        int y = __shfl_up_sync(0xffffffff, exc_w, off);
        if ((tid & 31) >= off) exc_w += y;
    }
    if ((tid & 31) == 31) csc_seg[tid >> 5] = exc_w;   // warp 段和
    __syncthreads();
    if (tid == 0) {
        int r2 = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) { int t = csc_seg[w]; csc_seg[w] = r2; r2 += t; }
    }
    __syncthreads();
    int wb = csc_seg[tid >> 5];
    for (int b = b_lo; b < b_hi; b++) h_start[b] += wb + exc_w - tot_t - h_cnt2[b];
    __syncthreads();
    // Pass3:桶分组重排(sc+sv 寄存器 staging;全部读定后再写,免读-写竞态)
    int rc[STG]; double rv[STG];
    #pragma unroll
    for (int i2 = 0; i2 < STG; i2++) {
        int k = tid + i2 * TPB;
        if (k < m) { rc[i2] = sc[k]; rv[i2] = sv[k]; }
    }
    __syncthreads();
    #pragma unroll
    for (int i2 = 0; i2 < STG; i2++) {
        int k = tid + i2 * TPB;
        if (k < m) {
            int b = (rc[i2] - lo) >> SH;
            int pos = h_start[b] + atomicAdd(&h_fill[b], 1);
            sc[pos] = rc[i2]; sv[pos] = rv[i2];
        }
    }
    __syncthreads();
    // Pass4:桶内 O(b²) 位次 + 直写 CSR(同 csort_fused 输出契约)
    for (int k = tid; k < m; k += TPB) {
        int c = sc[k]; int b = (c - lo) >> SH; int s0 = h_start[b];
        int wr = 0;
        for (int q = s0, e = s0 + h_cnt2[b]; q < e; q++) wr += (sc[q] < c);
        dC_ci[out + s0 + wr] = c;
        dC_val[out + s0 + wr] = sv[k];
    }
}
template<int TPB, int CAP, int NB = 1024>
static void launch_csort_counting(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                                  const int *dC_rp, const unsigned long long *k, const double *v,
                                  int *dC_ci, double *dC_val) {
    size_t smem = (size_t)CAP * (sizeof(int) + sizeof(double)) + 3 * (size_t)NB * sizeof(int);
    if (smem > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)csort_counting_kernel<TPB, CAP, NB>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    csort_counting_kernel<TPB, CAP, NB><<<n, TPB, smem>>>(rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val);
    CHECK_CUDA(cudaGetLastError());
}

// 刀16a:CAP 8192 档的 TPB 分派(env CS1024;ncu 实锤 <512,8192> SMEM 108KB→1块/SM=25%占用,
// barrier 全 SM 干等;<1024,8192> 同 SMEM 1块/SM=1024线程=50%占用,STG 16→8 减寄存器)
static void launch_cs8192(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                          const int *dC_rp, const unsigned long long *k, const double *v,
                          int *dC_ci, double *dC_val) {
    static int g_cs1024 = -1;
    if (g_cs1024 < 0) { const char *e = getenv("CS1024"); g_cs1024 = (e && *e && atoi(e) > 0) ? 1 : 0; }
    if (g_cs1024) launch_csort_counting<1024, 8192>(n, rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val);
    else          launch_csort_counting<512, 8192>(n, rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val);
}

// 刀16b:bin8 按真 d 劈双 CAP(d_row_nnz 排序时已知)——d≤4096 走 <512,4096>(60KB→3块/SM=75%占用,
// bin7 健康几何;ncu:现 <512,8192> 1块/SM=25%)。劈分 kernel:两计数器原子分桶(18K 行,µs 级)。
static __global__ void cs_part_kernel(const int *rows, int n, const int *d_row_nnz, int thr,
                               int *cnt_lo, int *cnt_hi, int *list_lo, int *list_hi) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int r = rows[j];
    if (d_row_nnz[r] <= thr) list_lo[atomicAdd(cnt_lo, 1)] = r;
    else                     list_hi[atomicAdd(cnt_hi, 1)] = r;
}
// env CS16B(默认关):=1 开 bin8 双 CAP 劈分
static void launch_cs8192_part(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                               const int *dC_rp, const unsigned long long *k, const double *v,
                               int *dC_ci, double *dC_val, int *scratch2, int thr) {
    static int g_cs16b = -1;
    if (g_cs16b < 0) { const char *e = getenv("CS16B"); g_cs16b = (e && *e && atoi(e) == 0) ? 0 : 1; }   // 刀16 翻默认 ON
    if (!g_cs16b || n < 8192) { launch_cs8192(n, rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val); return; }   // 规模门:小 bin8 人口免劈分(host 同步固定开销>F2 +3.6% 相位代价源)
    int *cnt_lo = scratch2, *cnt_hi = scratch2 + 1;
    int *list_lo = scratch2 + 2, *list_hi = scratch2 + 2 + n;
    CHECK_CUDA(cudaMemsetAsync(scratch2, 0, 2 * sizeof(int)));
    cs_part_kernel<<<(n + 255) / 256, 256>>>(rows_ptr, n, d_row_nnz, thr, cnt_lo, cnt_hi, list_lo, list_hi);
    int h2[2];
    CHECK_CUDA(cudaMemcpyAsync(h2, scratch2, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    static int g_cs16c = -1;
    if (g_cs16c < 0) { const char *e2 = getenv("CS16C"); g_cs16c = (e2 && *e2 && atoi(e2) == 0) ? 0 : 1; }   // 刀16 翻默认 ON
    if (h2[0] > 0) { if (g_cs16c) launch_csort_counting<512, 4096, 2048>(h2[0], list_lo, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val);   // NB=2048
                     else         launch_csort_counting<512, 4096>(h2[0], list_lo, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val); }
    if (h2[1] > 0) { if (g_cs16c) launch_csort_counting<512, 8192, 4096>(h2[1], list_hi, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val);   // NB=4096
                     else         launch_cs8192(h2[1], list_hi, d_off, d_row_nnz, dC_rp, k, v, dC_ci, dC_val); }
}

// v31-①'(docs/76 §10 靶纸,SEGRTS=1):bin≥7 大行 csort O(n²/block) 排序税换 cub 分段基数排序。
//   pre2 实证:bin6/7 380K 行(1-4K distinct)csort = 26.2ms(57% 时长)vs Ocean epilogue.sort 2.5ms。
static __global__ void bin_gather_kernel(
    const int *rows, int n, const long long *row_off, const int *row_nnz,
    const long long *bin_off,
    const unsigned long long *src_key, const double *src_val,
    unsigned long long *dst_key, double *dst_val)
{
    int j = blockIdx.x;
    if (j >= n) return;
    int i = rows[j];
    long long so = row_off[i], m = row_nnz[i], d = bin_off[j];
    for (int t = threadIdx.x; t < m; t += blockDim.x) {
        dst_key[d + t] = src_key[so + t];
        dst_val[d + t] = src_val[so + t];
    }
}
static __global__ void bin_scatter_kernel(
    const int *rows, int n, const int *row_ptr, const int *row_nnz,
    const long long *bin_off,
    const unsigned long long *src_key, const double *src_val,
    int *out_col, double *out_val, const int *skip)
{
    int j = blockIdx.x;
    if (j >= n) return;
    int i = rows[j];
    if (skip && skip[i]) return;
    long long d = bin_off[j];
    int dst = row_ptr[i], m = row_nnz[i];
    for (int t = threadIdx.x; t < m; t += blockDim.x) {
        out_col[dst + t] = (int)(src_key[d + t] & 0xffffffffu);
        out_val[dst + t] = src_val[d + t];
    }
}
static __global__ void bin_seg_kernel(const int *rows, int n, const long long *bin_off,
                               const int *row_nnz, int *seg_beg, int *seg_end)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    seg_beg[j] = (int)bin_off[j];                      // 按 bin 内序(heavy_seg_kernel 按行号,勿复用)
    seg_end[j] = (int)(bin_off[j] + row_nnz[rows[j]]);
}
struct RowNnzPicker {
    const int *row_nnz;
    __host__ __device__ int operator()(int r) const { return row_nnz[r]; }
};
static void launch_bin_segrts(int n, const int *rows_ptr,
                              const long long *d_off, const int *d_row_nnz,
                              const int *dC_rp, int *dC_ci, double *d_val,
                              unsigned long long *d_tmp_key, double *d_tmp_val,
                              const int *skip)
{
    if (n <= 0) return;
    int  *d_bnnz = (int*)dev_alloc((size_t)n * sizeof(int));
    long long *d_boff = (long long*)dev_alloc((size_t)(n + 1) * sizeof(long long));
    RowNnzPicker pick{d_row_nnz};
    thrust::transform(thrust::device_ptr<const int>(rows_ptr),
                      thrust::device_ptr<const int>(rows_ptr + n),
                      thrust::device_ptr<int>(d_bnnz), pick);
    CHECK_CUDA(cudaMemset(d_boff, 0, sizeof(long long)));            // [0]=0
    thrust::inclusive_scan(thrust::device_ptr<int>(d_bnnz),          // [1..n] = 前缀和(exclusive_scan
                           thrust::device_ptr<int>(d_bnnz + n),      // 只写 [0..n-1],读 [n] 曾得垃圾
                           thrust::device_ptr<long long>(d_boff + 1));   // → 15PB cudaMalloc,c-64 红旗根因
    long long btot = 0;
    CHECK_CUDA(cudaMemcpy(&btot, d_boff + n, sizeof(long long), cudaMemcpyDeviceToHost));
    if (btot <= 0) return;
    unsigned long long *gk = (unsigned long long*)dev_alloc((size_t)btot * sizeof(unsigned long long));
    double            *gv = (double*)dev_alloc((size_t)btot * sizeof(double));
    unsigned long long *ok = (unsigned long long*)dev_alloc((size_t)btot * sizeof(unsigned long long));
    double            *ov = (double*)dev_alloc((size_t)btot * sizeof(double));
    bin_gather_kernel<<<n, HASH_BLOCK_STRUCT>>>(rows_ptr, n, d_off, d_row_nnz, d_boff,
                                                d_tmp_key, d_tmp_val, gk, gv);
    int *d_sgb = (int*)dev_alloc((size_t)n * sizeof(int));
    int *d_sge = (int*)dev_alloc((size_t)n * sizeof(int));
    bin_seg_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
        rows_ptr, n, d_boff, d_row_nnz, d_sgb, d_sge);
    size_t cb = 0;
    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, cb, gk, ok, gv, ov, (int)btot, n, d_sgb, d_sge));
    void *ct = dev_alloc(cb);
    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
        ct, cb, gk, ok, gv, ov, (int)btot, n, d_sgb, d_sge));
    bin_scatter_kernel<<<n, HASH_BLOCK_STRUCT>>>(rows_ptr, n, dC_rp, d_row_nnz, d_boff,
                                                 ok, ov, dC_ci, d_val, skip);
}

// launch helper:按 config 查 BlockRadixSort TempStorage 大小,>48KB 自动 opt-in 动态 shared(H100 可 ~228KB)。
template<int TPB, int IPT>
static void launch_csort(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                         const int *dC_rp, const unsigned long long *d_tmp_key, const double *d_tmp_val,
                         int *dC_ci, double *d_val, const int *skip) {
    using BRS = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    size_t smem = sizeof(typename BRS::TempStorage);
    if (smem > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)hash_compact_sort_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    if (getenv("CSORT_DBG")) fprintf(stderr, "[csort-dbg] TPB=%d IPT=%d n=%d smem=%zu\n", TPB, IPT, n, smem);
    hash_compact_sort_kernel<TPB, IPT><<<n, TPB, smem>>>(
        rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, skip);
}

// DBG 校验:输出 CSR 每行 col 严格升序(验证 compact_sort 排序正确)。
#endif   // P1 G1 结束(compact/csort host 助手区;hash_check_sorted 起 TU-61 可见——死码无害)

static __global__ void hash_check_sorted_kernel(const int *row_ptr, const int *col, int A_rows, int *violations) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= A_rows) return;
    int s = row_ptr[r], e = row_ptr[r + 1];
    for (int i = s; i < e - 1; i++)
        if (col[i] >= col[i + 1]) atomicAdd(violations, 1);
}

// ═════ spec99 刀1:重行同构复刻 kernel(2026-09-29 开工,study/26 §5 蓝图/inno/99)══════
// 对照形态非创新(用户批准):机理吸收 ocean hashNumeric 五要素,代码自研——
// ①素数贴合档:HT_P 模板参复用 spec91 资产(magic-div 编译期取模+8B 偶对齐,1746 式)
// ②占用驱动块几何:launch_bounds 线程×块积=2048/SM(2719↔512T×4 / 5449↔1024T×2)
// ③组自适应并行:localLoadBalance 机理(avg B 行长定组+双向 2× 容差重平衡,G∈[2,块宽])
// ④极简栅栏:全程 3 个(init/收尾/仪器),主环零栅栏零分支(对照 hash_spa 18 栅栏)
// ⑤大行无序写交下游 compact+sort(hash_spa 大行同款;有序化由 compact_sort_kernel 保)
// 保留我方资产:Knuth×2654435761 探测(雪崩优于彼 id*11,A/B 实验后定去留);溢出走
// ovf_rows→retry 机具(hash_spa 同款);rowtime[i]=探测环完成处(同彼 ROWCYCLES 窗,不含提取)。
// SPEC99 env 门(默认关=零生产影响);bins7-8 接管,bin9+ 仍走原路径。
// 刀2:B16ON 模板参(交织 B 装载,每乘积 1 次 uint4 vs 2 次分离读——hash_spa 同款,编译期消分支)
template<int LOG_NTHR, int HT_P, int B16ON = 0, int OCH = 0>   // OCH=1:彼 id*11 哈希 A/B(刀10:错峰机理直测)
__global__ __launch_bounds__((1 << LOG_NTHR), (2048 >> LOG_NTHR)) void hash_heavy_kernel(
    const int *__restrict__ A_rp, const int *__restrict__ A_ci, const double *__restrict__ A_val,
    const int *__restrict__ B_rp, const int *__restrict__ B_ci, const double *__restrict__ B_val,
    const uint4 *__restrict__ B16,
    const int *row_ids, int nrows, const long long *d_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz,
    int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *d_flop, const int *d_maxbl, unsigned long long *rowtime,
    const int *d_est, int est_lo, int est_hi, const int rot_on)   // 刀3过滤+刀11旋转总闸(SPEC99_ROT,host 读入)
{
    (void)nrows;
    extern __shared__ int smem_hh[];                        // [HT_P]=sh_col(int);对齐后 sh_val(double)
    const int tid = threadIdx.x;
    const unsigned long long rt0 = rowtime ? clock64() : 0;
    // 刀Gv3(study/57 §五十):hash_heavy 表内有序 rank 直写——env 经已有的 rt_pre_extract
    // 参数位传入不便,新增参数侵入大;改用 __device__ 全局符号 host 装载(同 g_xbit_d 模式)
    int *sh_col = smem_hh;
    double *sh_val = (double *)(smem_hh + ((HT_P + 1) & ~1));   // 8B 对齐(素数奇表必需,spec91 修复)
    for (int s = tid; s < HT_P; s += blockDim.x) { sh_col[s] = -1; sh_val[s] = 0.0; }   // 清表(ids+vals 同遍,彼同构)
    __syncthreads();                                       // 栅栏 1/3: init

    const int i = row_ids[blockIdx.x];
    if (est_hi > 0) {                       // 刀3:raw est 域过滤(还 d_est 的 1.8× 膨胀;ncu 实证彼 2719 档 issue 77%)
        const int raw_est = (int)(((long long)d_est[i] * 5) / 9);
        if (raw_est < est_lo || raw_est > est_hi) return;
    }
    const int rs = A_rp[i], re = A_rp[i + 1];
    const long long base = d_off[i];
    if (tid == 0) row_nnz[i] = 0;                          // 溢出行会被 retry 覆盖,与 hash_spa 同语义

    // 组自适应(机理吸收自研):avg B 行长(剔除最长)定初始 lg → 最长 B 行剩余轮次 vs A 迭代数
    // 双向 2× 容差重平衡 → G∈[2, 2^LOG_NTHR](治我方 G 帽 32 的跨 warp 壁垒,crankseg 型主受益)
    {
        const int a_len = re - rs;
        const long long prods = d_flop[i];
        const int max_bl = d_maxbl[i] + 1;
        long long avg = (prods - max_bl) / (a_len > 1 ? a_len - 1 : 1);
        if (avg < 1) avg = 1;
        int lg = 31 - __clz((unsigned)avg);
        if (lg < 2) lg = 2;
        if (lg > LOG_NTHR) lg = LOG_NTHR;
        long long max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
        long long iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        while (max_sub > iters * 2 && lg < LOG_NTHR) {      // 组太小:最长 B 行拖尾
            lg++; max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
            iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        }
        while (iters > max_sub * 2 && lg > 2) {             // 组太大:A 侧并行度浪费
            lg--; max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
            iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        }
        const int group_size = 1 << lg;
        const int num_groups = blockDim.x >> lg;
        const int my_group = tid >> lg;
        const int my_id = tid & (group_size - 1);

        // 刀11:k 序旋转错峰(直击 PC 采样的到达压缩:近邻 k 对共享 j 最密,原序下它们恰好同时刻
        // 在不同 warp 被处理=正面对撞;每 warp 旋转 own k 序列,近邻对的到达时间被拉开。
        // 每 k ~3 条指令(非每乘积),乘积集合与表操作完全不变=零正确性风险,仅原子到达次序变化)
        {
            int cnt = 0;
            for (int p = rs + my_group; p < re; p += num_groups) cnt++;
            // 行几何门 v2:cnt≥10 且 B 行短(maxbl≤1024)才旋转——错峰收益在短 B 行 nd24k 型
            // (maxbl≈520);长 B 行 crankseg 型(≈2000+)错峰无用反赔 k 序 L2 局部性(×3 实证)
            // 刀11 门 v3(dup 判据,原理性):旋转收益∝dup 浓度(到达压缩源),伤害∝k 序局部性;
            // dup=flop/est≥20 才开——nd24k(62×)吃收益,crankseg bin7-8 型(低 dup)免疫(maxbl 门实证拦不住)
            const bool rot_ok = rot_on && (cnt >= 10) && ((long long)d_flop[i] >= 20LL * d_est[i]);
            const int rot = rot_ok ? (int)(((unsigned)(tid >> 5) * 2654435761u) % (unsigned)cnt) : 0;
            // K3A(study/31 错峰族):affine 置换 idx=(s*stride+rot)%cnt——stride=warp 素数(7/11/13...)
            // 比刀11 纯旋转(相邻 k 仍相邻,只整体平移)错峰更彻底:近邻 k 对到达时间差 ∝ stride
            const int stride = rot_ok ? (7 + 4 * (int)((tid >> 5) & 3)) : 1;
            for (int s = 0; s < cnt; s++) {
            const int idx = (s * stride + rot) % cnt;
            const int p = rs + my_group + idx * num_groups;
            const int k = A_ci[p];
            const double a_ik = A_val[p];
            const int ks = B_rp[k], ke = B_rp[k + 1];
            for (int q = ks + my_id; q < ke; q += group_size) {
                int j; double bv;
                if (B16ON) {                       // 交织 B:一次 128b 携带 col+val(hash_spa 同款)
                    const uint4 e = B16[q];
                    j = (int)e.x; bv = __hiloint2double(e.w, e.z);
                } else { j = B_ci[q]; bv = B_val[q]; }
                const double v = a_ik * bv;
                // 刀6:编译期 pow2 分派(pow2 表→mask 低位 1 条;素数表→magic-div)——刀8 判负回滚:
                // multiply-shift 高位取槽 87.8(+8.8% vs mask 80.75),Knuth 低位已够混,高位假设证伪
                constexpr unsigned HH_MSK = (unsigned)(HT_P - 1);
                constexpr bool HH_P2 = ((HT_P & (int)HH_MSK) == 0);   // HT_P 为 2^n?
                // 刀10:OCH=1 换彼 id*11 哈希(混合差→聚类多→链长→dup 到达错峰→自旋少;PC 采样假说直测)
                unsigned slot = HH_P2 ? (((unsigned)(OCH ? (j * 11u) : (j * 2654435761u))) & HH_MSK)
                                      : ((unsigned)(OCH ? (j * 11u) : (j * 2654435761u)) % (unsigned)HT_P);
                int probes = 0;                             // 刀4 判负回滚:start_slot 版抑制 ptxas 跨乘积展开(5层→2层,+12%);计数器版让展开摊销循环控制,更优
                while (true) {
                    const int entry = sh_col[slot];
                    if (entry == j) { atomicAdd(&sh_val[slot], v); break; }
                    if (entry == -1) {
                        const int prev = atomicCAS(&sh_col[slot], -1, j);
                        if (prev == -1 || prev == j) { atomicAdd(&sh_val[slot], v); break; }
                    }
                    slot = HH_P2 ? ((slot + 1) & HH_MSK) : ((slot + 1 >= (unsigned)HT_P) ? 0u : slot + 1);
                    if (++probes >= HT_P) {                 // 满表:retry 机具(hash_spa 同款)
                        atomicExch(overflow_flag, 1);
                        if (!atomicExch(&row_ovf[i], 1)) { const int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                        break;
                    }
                }
            }
            }
        }
    }
    __syncthreads();                                       // 栅栏 2/3: 收尾
    // 公平口径修正:rowtime 移到提取后(= 完整 kernel 时间,含 extraction)
    // 之前在探测环后测 = 不含提取,跟彼方 ROWCYCLES(含提取)不同窗 → 偏向我们
    // 现在两边都是"从 kernel 开始到 kernel 结束"的完整时间

    // 提取:刀Gv3v2(study/57 §六十)有序直写——v1 三死因逐一拆:
    // ①段认领=seg=warp序+k×nw(每段 32 连续槽,全体 warp 进环;v1 的 tid*32 步进=仅 1/4 线程);
    // ②段内位移 rank=5 轮 __shfl_xor 广播同段键累"有效且小于我"位掩码,popc(v1 逐槽串行比较);
    // ③跨段前缀按表序入 smem 段数组(段序=表序,v1 的 warp 前缀在 warp 段跨表序散布=次序错);
    // ④cap=d_off 行槽宽,rank≥cap 走既有 retry 熔断语义(v1 cap2=INT_MAX 裸写)。
    // 表内键天然无重 → rank 直写=行内有序,下游免 counting sort 载体趟(刀G载体悖论正解)。
    if (g_hh_sorted_d) {
        const int lane5 = tid & 31, wid = tid >> 5, nw = (int)(blockDim.x >> 5);
        const int nseg = (HT_P + 31) >> 5;
        __shared__ int seg_pre[(HT_P + 31) >> 5 > 1 ? (HT_P + 31) >> 5 : 1];   // 段 exclusive 前缀(HT_P=8192 → 256 int=1KB)
        // Pass A:段计数——ballot 必须全体 lane 到场(§六十一:首版把 ballot 写进
        // if(lane5==0) = 仅 lane0 等全掩码 = 必挂;结果由 lane0 落账)
        for (int seg = wid; seg < nseg; seg += nw) {
            const int lo = (seg << 5) + lane5;
            const int c = (lo < HT_P) ? sh_col[lo] : -1;
            const unsigned valid = __ballot_sync(0xffffffffu, c != -1);
            if (lane5 == 0) seg_pre[seg] = __popc(valid);
        }
        __syncthreads();
        if (tid == 0) {   // exclusive 前缀(段序=表序;单线程 nseg≤256 串行,亚 μs)
            int acc = 0;
            for (int s = 0; s < nseg; s++) { const int t = seg_pre[s]; seg_pre[s] = acc; acc += t; }
            row_nnz[i] = acc;
        }
        __syncthreads();
        const int cap = (int)(d_off[i + 1] - d_off[i]);   // 行 est 槽宽
        // Pass B:重算段内 rank(shfl 现算,免 smem 位次表)+ 直写
        for (int seg = wid; seg < nseg; seg += nw) {
            const int lo = (seg << 5) + lane5;
            const int c = (lo < HT_P) ? sh_col[lo] : -1;
            unsigned lt = 0;   // 段内"有效且小于我"位掩码
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int other = __shfl_xor_sync(0xffffffffu, c, off);
                if (other != -1 && other < c) lt |= (1u << (lane5 ^ off));
            }
            if (c != -1) {
                const int r = seg_pre[seg] + __popc(lt);
                if (r < cap) {
                    tmp_key[base + r] = ((unsigned long long)i << 32) | (unsigned int)c;
                    tmp_val[base + r] = sh_val[lo];
                } else {   // 超帽:既有 retry 熔断语义(探测环满表同款)
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { const int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                }
            }
        }
    } else
    // 提取(legacy):无序 compact 写(hash_spa 大行同款;行内有序由下游 compact_sort_kernel 保证)
    for (int s = tid; s < HT_P; s += blockDim.x) {
        if (sh_col[s] != -1) {
            const int pos = atomicAdd(&row_nnz[i], 1);
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)sh_col[s];
            tmp_val[base + pos] = sh_val[s];
        }
    }
    // 公平口径:完整 kernel 时间(含提取),对齐彼方 ROWCYCLES 窗
    __syncthreads();
    if (rowtime && tid == 0) rowtime[i] = clock64() - rt0;
}

// ===== P1 TU-61 出口(study/58 附录A.1 v2)=====
// cutover 段将在此移入 bi≥4 accumulate 梯(HashAccumCtx + 名字别名,移动体文本不动)
// + HH_SORTED loader(本 TU cubin 的 g_hh_sorted_d 副本,探针P3)+ TB_TAG 版 SPA_LAUNCH0。
// B1 骨架:仅证 TU-61 符号层(device 模板区在 compute_61 下可编、门内零 host 发射)。
// P1 cutover(study/58 附录A.1):accumulate 上下文——TU-90 构造,TU-61 消费。
// 字段=移动体(generic-else 内部原文)引用的全部外层名(编译器驱动核对)。
struct HashAccumCtx {
    const int *dA_rp; const int *dA_ci; const double *dA_val;
    const int *dB_rp; const int *dB_ci; const double *dB_val;
    const int *rows_ptr; int n;
    long long *d_off; unsigned long long *d_tmp_key; double *d_tmp_val;
    int *d_row_nnz; int *d_overflow; int *d_ovf_rows; int *d_ovf_cnt; int *d_row_ovf;
    int *d_flop; int *d_maxbl; unsigned long long *d_rt; int *d_est;
    unsigned *d_phist; uint4 *dB16;
    double **d_gv_bins;              // 刀E v2 per-bin VG 池(hash_product 本体数组,TU-61 侧填充分配)
    cudaStream_t cur_s; const char *tag;
    int upper_tri, A_rows, A_cols, g_llb, g_hyb_tier, g_priv, PRIV_W;
    bool att;
};
void hash_accum_c61(HashAccumCtx &c, int bi);   // TU-61(spgemm_kernel_hash_c61.o)提供
#if HASH_TB == 61
// TU-61 大表/托付形态发射器:generic-else 内部原文移入(仅删 bi2/3 分支保实例集不相交)
void hash_accum_c61(HashAccumCtx &c, int bi) {
    // §六十六:E3 同类雷的 TU-61 版——tagged 实例独立符号,原 opt-in 块(TU-90 untagged)
    // 不覆盖 → bin9 大表(196KB)launch invalid argument(Ga10As10H30 实证,40 阵批量空值)。
    // 镜像 TU-90 块(hash_product 6080-6112)为 TB_TAG=61 全家:一次性 opt-in。
    {
        static bool done = false;
        if (!done) {
            const size_t maxsm = (size_t)HASH_CAP * (sizeof(int) + sizeof(double));
            const size_t maxsm_vg = (size_t)HASH_CAP * sizeof(int);
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,7,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,7,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,8,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,8,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,10,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,10,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,11,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,11,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,12,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,12,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,13,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,13,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,14,0,0,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,14,0,1,0,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,0,0,0,0,0,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,0,1,0,0,0,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,0,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,0,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,1,0,0,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm_vg));   // VG ids-only 表
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,2719,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2719*12)));   // 素数档(PHT)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,5449,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(5449*12)));   // 素数档(PHT)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,16253,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(16253*12)));   // 素数档(PHT)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,2719,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2719*12)));   // 素数档(PHT)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,5449,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(5449*12)));   // 素数档(PHT)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,16253,0,-1,HASH_TB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(16253*12)));   // 素数档(PHT)
            done = true;
        }
    }
    auto &dA_rp = c.dA_rp; auto &dA_ci = c.dA_ci; auto &dA_val = c.dA_val;
    auto &dB_rp = c.dB_rp; auto &dB_ci = c.dB_ci; auto &dB_val = c.dB_val;
    auto &rows_ptr = c.rows_ptr; auto &n = c.n;
    auto &d_off = c.d_off; auto &d_tmp_key = c.d_tmp_key; auto &d_tmp_val = c.d_tmp_val;
    auto &d_row_nnz = c.d_row_nnz; auto &d_overflow = c.d_overflow;
    auto &d_ovf_rows = c.d_ovf_rows; auto &d_ovf_cnt = c.d_ovf_cnt; auto &d_row_ovf = c.d_row_ovf;
    auto &d_flop = c.d_flop; auto &d_maxbl = c.d_maxbl; auto &d_rt = c.d_rt; auto &d_est = c.d_est;
    auto &d_phist = c.d_phist; auto &dB16 = c.dB16;
    auto &d_gv_bins = c.d_gv_bins; double *d_gv = nullptr;   // decltype 锚(VG 池分配)
    auto &cur_s = c.cur_s; auto &tag = c.tag;
    auto &upper_tri = c.upper_tri; auto &A_rows = c.A_rows; auto &A_cols = c.A_cols;
    auto &g_llb = c.g_llb; auto &g_hyb_tier = c.g_hyb_tier; auto &g_priv = c.g_priv; auto &PRIV_W = c.PRIV_W;
    auto &att = c.att;
    static int g_hhs_host = -1;
    if (g_hhs_host < 0) {
        const char *e = getenv("HH_SORTED");
        g_hhs_host = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1;   // Gv3v2 转正默认 1(§六十一四阵+13阵体检全绿);HH_SORTED=0 逃生门
        CHECK_CUDA(cudaMemcpyToSymbol(g_hh_sorted_d, &g_hhs_host, sizeof(int)));
    }
    // HASH_ROWTIME=path:hash_spa 逐行执行周期(docs/66 采数,判 merge3 路由);null=生产零开销

                static int g_ht_shift = -1;   // HASH_HT_SHIFT(study/54 LF 弹性探针):bins≤6 表松弛位;默认 0=旧行为零变更
                if (g_ht_shift < 0) { const char *e = getenv("HASH_HT_SHIFT"); g_ht_shift = (e && *e) ? atoi(e) : 0; }
                int ht = (WARP_SIZE << bi) << ((bi <= 5) ? g_ht_shift : 0);
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                // spec 81 两级排序 hist 预算:3×1024×4B=12KB(仅 512≤ht≤1024 档——kernel 侧分支同界;
                // bin5 表 12KB+12KB=24KB,256 线程 8 块 ×24KB=192KB < 228KB 不压 occupancy。
                // ⚠ ht>CSORT_HT 档绝不能加:bin9 表 196KB+12KB=208KB > opt-in(HASH_CAP×12)→ 发射失败)
                static int g_extract2 = -1;   // B'(09-24 用户授权默认全局开):两级计数排序;HASH_EXTRACT=0 逃生门
                // (bin 级 dup 条件版试作判负:stat 核在 F2/msc 触发未定位 fault,且条件仅护 msc +0.06ms 绝对值——不值)
                if (g_extract2 < 0) { const char *e = getenv("HASH_EXTRACT"); g_extract2 = (e && atoi(e) == 0) ? 0 : 1; }
                if (g_extract2 && ht >= CS2_MIN_HT && ht <= CSORT_HT) smem_flat += 3 * 1024 * sizeof(int);
                size_t smem_priv = (size_t)PRIV_W * ht * (sizeof(int) + sizeof(double));
                // spec91 刀E:VG 发射所需 env statics + 块梯(原在下游 else 内,上提共用)
                static int g_force_logG = -1;   // GFORCE(study/15 诊断):强制 G;0=正常梯
                if (g_force_logG < 0) { const char *e = getenv("GFORCE"); g_force_logG = (e && *e) ? __builtin_ctz(atoi(e)) : 0; }
                static int g_uo_x = -1;
                if (g_uo_x < 0) { const char *e = getenv("HASH_UO_XTRACT"); g_uo_x = (e && *e && atoi(e) > 0) ? 1 : 0; }
                static int g_rt_nox = -1;
                if (g_rt_nox < 0) { const char *e = getenv("HASH_RT_NOEXTRACT"); g_rt_nox = (e && *e && atoi(e) > 0) ? 1 : 0; }
                static int g_wm = -1;
                if (g_wm < 0) { const char *e = getenv("HASH_WM"); g_wm = (e && *e && atoi(e) > 0) ? 1 : 0; }
                // docs/44 per-bin block 梯(Ocean BLOCK_SIZES 同哲学)+ v37 修正梯(study/06 §7 F2 对拍):
                //   默认 = 旧三档 64/256/512 + bi≥8→1024T(修 occupancy 缺陷:ht≥8192 表 ≥96KB ×512T
                //   只驻 2 CTA = 1024 线程;1024T 恢复满格 2048。F2/电池已证)。
                //   HASH_BLK_LADDER=legacy 纯旧三档;=6:384,7:768 逐档覆写(32 倍数,免重编)。
                // (spec91 刀E:梯计算上提——VG 发射同吃 hblk;旧 hkv 固定 256T 是 8月否决的
                //  病因之一:同 bin SMEM 路径 512/1024T,VG 256T = 行车道少 2-4× → 行延迟 4-5×)
                static int g_blk_lad[HASH_LADDER_MAX_BIN + 1];
                static int g_blk_init = -1;
                if (g_blk_init < 0) {
                    for (int b = 0; b <= HASH_LADDER_MAX_BIN; b++)
                        g_blk_lad[b] = (b <= ilog2c(SMALL_SPAN_MAX / WARP_SIZE)) ? HASH_BLOCK_STRUCT / 4
                                         : (b <= ilog2c(CSORT_HT / WARP_SIZE)) ? HASH_BLOCK_STRUCT : DENSE_BLK;
                    const char *e = getenv("HASH_BLK_LADDER");
                    if (e && !strcmp(e, "legacy")) {
                        // 纯旧梯,不加默认覆写
                    } else if (e && *e) {
                        // 用户逐档覆写全权(不再叠加默认)
                        for (char *s = strdup(e), *tok = strtok(s, ","); tok; tok = strtok(nullptr, ","))
                        {
                            int b, t;
                            if (sscanf(tok, "%d:%d", &b, &t) == 2 && b >= 0 && b <= HASH_LADDER_MAX_BIN &&
                                t > 0 && t <= 1024 && (t & 31) == 0)
                                g_blk_lad[b] = t;
                        }
                    } else {
                        for (int b = 8; b <= HASH_LADDER_MAX_BIN; b++) g_blk_lad[b] = 1024;   // 默认(×5 电池已证)
                    }
                    g_blk_init = 1;
                }
                int hblk = g_blk_lad[bi <= HASH_LADDER_MAX_BIN ? bi : HASH_LADDER_MAX_BIN];
                static int g_vg45 = -1;   // 刀E2(study/57 §廿六):bin4→VG 空格补测——VAL_GMEM(L2 原生 RED.E.ADD.F64,免 smem CAST.SPIN 模拟税)×128T 窄块 = ocean <7,673> 耦合配置;默认关
                if (g_vg45 < 0) { const char *e = getenv("HASH_VG45"); g_vg45 = (e && *e) ? atoi(e) : 0; }
                if (g_hyb_tier && !att && (ht >= TABLE_DOUBLING_THR || (g_vg45 && bi == 4)) && probe_rf_on()) {
                    // spec91 刀E:大表 bin 值区 gmem 池 + 主力内核模板实例(VAL_GMEM=1)。
                    // 取代 hash_spa_hkv_kernel(2026-08-27 否决的残缺形态:无 read-first 每乘积必
                    // CAS/无 B16/固定块宽;刀E 保留探测全套现代机构,仅值区几何换 gmem——累加
                    // atomicAdd 弃返回值 → RED.E.ADD.F64 原生归约一条/乘积,SASS 对拍见 inno/91 §1.5)。
                    // probe_hist 仪器走 SMEM 形态(VG 行不采探测链);PRF 逃生门开着则回 SMEM 路径。
                    // v2 有界循环缓冲池:N_BUF ≥ 最大并发块(REG32 占用上限 × SM 数)+20% 裕量,
                    // 且 N_BUF×ht×8B ≤ L2 预算(48MB)——RED 全程 L2 命中(v1 全量池 1.1GB 落 DRAM 的教训)
                    static int g_sm_count = -1;
                    if (g_sm_count < 0) { cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0)); g_sm_count = prop.multiProcessorCount; }
                    int bpsm = max(1, 65536 / (hblk * 32));                       // REG:32 上限(占用恒等式;launch_bounds(1024,2) 一致)
                    long long nbuf = (long long)g_sm_count * bpsm * 6 / 5;
                    const long long l2_budget = 48LL << 20;                       // HYB_L2_BUDGET:H100 L2 50MB 内留 2MB
                    if (nbuf * (long long)ht * 8 > l2_budget) nbuf = l2_budget / ((long long)ht * 8);
                    if (nbuf < 1) nbuf = 1;
                    if (!d_gv_bins[bi])
                        d_gv_bins[bi] = decltype(d_gv)(dev_alloc((size_t)nbuf * ht * sizeof(double)));
                    dbg("[%s] VG bin %d: ht=%d hblk=%d nbuf=%lld pool=%.1fMB\n", tag, bi, ht, hblk, nbuf, nbuf * ht * 8.0 / (1 << 20));
                    hash_spa_kernel<0,1,0,1,0,0,0,-1,HASH_TB><<<n, hblk, (size_t)ht * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr,
                        g_force_logG, g_uo_x, dB16, g_wm, nullptr, g_rt_nox,
                        (ht < TABLE_DOUBLING_THR) ? 0 : g_extract2,   // 刀E2:bin4 VG 无 CS2 hist smem → legacy rank extract(T4a:d≤384 rank 优)
                        (int)nbuf, d_gv_bins[bi]);
                    CHECK_CUDA(cudaGetLastError());   // 发射即时检查(opt-in/表大小问题即抓)
                } else if (!att && g_priv && smem_priv <= (size_t)PRIV_SMEM_MAX) {
                    hash_spa_priv_kernel<<<n, PRIV_W * WARP_SIZE, smem_priv>>>(
                        dA_rp, dA_ci, dA_val, rows_ptr, n, ht, PRIV_W, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
                } else {
                        // docs/44 per-bin block 梯(Ocean BLOCK_SIZES 同哲学)+ v37 修正梯(study/06 §7 F2 对拍):
                    // (spec91 刀E:梯计算上提共用——VG 发射同吃 hblk;旧 hkv 固定 256T 是 8月否决的
                    //  病因之一:同 bin SMEM 路径 512/1024T,VG 256T = 行车道 2-4× 少 → 行延迟 4-5×)
                    int hblk = g_blk_lad[bi <= HASH_LADDER_MAX_BIN ? bi : HASH_LADDER_MAX_BIN];
                    // spec92-A:行群 a_len 选宽(结构条件禁阵身份)——bins6-7(512T 自然档)行群平均
                    // a_len ≥ WSEL_ALEN_THR(默认 48=1024T/32 组的裕量)时升 1024T:大 a_len 行
                    // (crankseg 209)块宽=行并行度;小 a_len 行(in-2004 24<32 组)升宽=空组纯亏。
                    static int g_wsel = -1, g_wsel_thr = 48;
                    if (g_wsel < 0) {
                        const char *e = getenv("HASH_WSEL");
                        g_wsel = (e && *e && atoi(e) > 0) ? 1 : 0;
                        const char *t = getenv("WSEL_ALEN_THR");
                        if (t && *t) g_wsel_thr = atoi(t);
                    }
                    if (g_wsel && bi >= 6 && bi <= 7 && hblk < 1024) {
                        static unsigned long long *d_wsum = nullptr;
                        if (!d_wsum) { d_wsum = decltype(d_wsum)(dev_alloc(sizeof(unsigned long long)));
                                       CHECK_CUDA(cudaMemset(d_wsum, 0, sizeof(unsigned long long))); }
                        CHECK_CUDA(cudaMemsetAsync(d_wsum, 0, sizeof(unsigned long long), cur_s));
                        bin_alen_stat_kernel<<<(n + 255) / 256, 256, 0, cur_s>>>(dA_rp, rows_ptr, n, d_wsum);
                        unsigned long long sum_al = 0;
                        CHECK_CUDA(cudaMemcpyAsync(&sum_al, d_wsum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, cur_s));
                        CHECK_CUDA(cudaStreamSynchronize(cur_s));
                        int mean_al = n > 0 ? (int)(sum_al / n) : 0;
                        if (mean_al >= g_wsel_thr) {
                            hblk = 1024;
                            dbg("[%s] WSEL bin %d: mean_a_len=%d ≥ %d → 1024T\n", tag, bi, mean_al, g_wsel_thr);
                        }
                    }
                    // read-first 探测已并入 hash_spa_kernel 本体(PROBE_RF 模板参;逃生门 HASH_PROBE_RF=0)
                    // (g_uo_x/g_rt_nox/g_wm/g_force_logG 已上提至 VG 分支前共用,spec91 刀E)
                    // spec91 ①:仪器态(d_phist≠null)单独实例化 PROBE_HIST_ON=1,生产路径零探测计数指令
#define SPA_LAUNCH0(PRF_) do { \
    if (d_phist) hash_spa_kernel<0,(PRF_),1,0,0,0,0,-1,HASH_TB><<<n, hblk, smem_flat, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, ht, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, d_phist, g_rt_nox, g_extract2); \
    else         hash_spa_kernel<0,(PRF_),0,0,0,0,0,-1,HASH_TB><<<n, hblk, smem_flat, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, ht, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, d_phist, g_rt_nox, g_extract2); \
} while (0)
                    // spec91 刀F:LOG_HT 编译期定长实例(bins 7-9 = ht 4096/8192/16384 损失带;
                    // HASH_LHT=1 门,默认 0 走下方运行时形态 bit-for-bit)。仪器(probe_hist)/WM/PRF
                    // 逃生门开着时回运行时形态。
                    // spec98 轮1:WMA 归并累积器发射(WMA=1 劫持 bins8-9 重行;est 槽=row_off 间隙,
                    // 输出 tmp_key 兼容下游;SAFETY:num_k≤2048(a_len 帽)否则回落哈希)
                    static int g_wma = -1;
                    if (g_wma < 0) { const char *e = getenv("WMA"); g_wma = (e && *e && atoi(e) > 0) ? 1 : 0; }
                    static int g_pht = -1;   // (上提:声明不可悬于 else 后)
                    if (g_pht < 0) { const char *e = getenv("HASH_PHT"); g_pht = (e && *e && atoi(e) > 0) ? 1 : 0; }
                    // spec99 刀1:重行同构复刻发射(SPEC99=1 门默认关;bins7-8 接管=2719↔512T×4/
                    // 5449↔1024T×2;占 hash_spa 之外独立 kernel,行计时段同彼窗)
                    static int g_s99 = -1;
                    if (g_s99 < 0) { const char *e = getenv("SPEC99"); g_s99 = (e && *e && atoi(e) == 0) ? 0 : 1; }   // T1 翻默认 ON(10-04:行级 0.726/100%/n53万);=0 逃生门
                    static int g_s99cap = -1;   // 刀3/3b 分裂帽:全线判负(1950/2100/2311=106.1/109.2/116.9 全劣于 100.9)——2719 档红利不付账,默认 0=关;env 留作后续复验
                    if (g_s99cap < 0) { const char *e = getenv("SPEC99_CAP"); g_s99cap = (e && *e) ? atoi(e) : 0; }
                    static int g_s99och = -1;   // 刀10:id*11 哈希 A/B(错峰机理直测;0=Knuth 默认)
                    if (g_s99och < 0) { const char *e = getenv("SPEC99_OCH"); g_s99och = (e && *e) ? atoi(e) : 0; }
                    static int g_s99rot = -1;   // 刀11 旋转总闸:默认 0 关(nd24k −1.7%但 crankseg +1.7%,三行级门(cnt/maxbl/dup)均无区分力=人口整体行为;SPEC99_ROT=1 开)
                    if (g_s99rot < 0) { const char *e = getenv("SPEC99_ROT"); g_s99rot = (e && *e) ? atoi(e) : 0; }
                    static int g_h45 = -1;   // 刀E1 试点(study/57 §廿五):bin4→hash_heavy 瘦核窄块(ocean <7,673> 对位 128T);默认关=原 LHT 路径逐位
                    if (g_h45 < 0) { const char *e = getenv("HASH_H45"); g_h45 = (e && *e) ? atoi(e) : 1; }
                    static int g_h55 = -1;   // 刀G(study/57 §四十九):bin5→hash_heavy<10,2048> 1024T(compute_61 复活后扩覆盖;bin5 est≤1024→2048 素数梯);默认关
                    if (g_h55 < 0) { const char *e = getenv("HASH_H55"); g_h55 = (e && *e) ? atoi(e) : 1; }   // Gv3v2 转正默认 1;HASH_H55=0 逃生门
                    static int g_h69 = -1;   // 刀①'(study/59):bin9→HH(16253) 默认 1(五阵定谳 §刀①');bin6→HH(2719) 需 H69=2(未验证人口不默认翻);HASH_H69=0 逃生门
                    if (g_h69 < 0) { const char *e = getenv("HASH_H69"); g_h69 = (e && *e) ? atoi(e) : 1; }
                    if (g_s99 && ((bi >= 7 && bi <= 8) || (g_h45 && bi == 4) || (g_h55 && bi == 5) || (g_h69 && bi == 9) || (g_h69 >= 2 && bi == 6)) && !g_wm && !g_pht) {
#define HH_LAUNCH2(LNT_, P_, OCH_, LO_, HI_, ROT_) do { \
    const size_t hhsm = (size_t)(((P_) + 1) & ~1) * sizeof(int) + (size_t)(P_) * sizeof(double); \
    CHECK_CUDA(cudaFuncSetAttribute(hash_heavy_kernel<(LNT_), (P_), 1, (OCH_)>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hhsm)); \
    CHECK_CUDA(cudaFuncSetAttribute(hash_heavy_kernel<(LNT_), (P_), 0, (OCH_)>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hhsm)); \
    if (dB16) hash_heavy_kernel<(LNT_), (P_), 1, (OCH_)><<<n, (1 << (LNT_)), hhsm, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, dB16, rows_ptr, n, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, d_rt, d_est, (LO_), (HI_), (ROT_)); \
    else      hash_heavy_kernel<(LNT_), (P_), 0, (OCH_)><<<n, (1 << (LNT_)), hhsm, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, nullptr, rows_ptr, n, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, d_rt, d_est, (LO_), (HI_), (ROT_)); } while (0)
#define HH_LAUNCH(LNT_, P_) do { if (g_s99och) HH_LAUNCH2((LNT_), (P_), 1, 0, 0, g_s99rot); else HH_LAUNCH2((LNT_), (P_), 0, 0, 0, g_s99rot); } while (0)
                        // 刀6:pow2 档终极 A/B(4096/8192=hash_spa 同表经济,mask 1 条 vs magic-div 6 条)——
                        // 隔离 3 栅栏结构 vs 18 栅栏的净价值;刀5 素数版(4111/8209)留 git 史
                        if (bi == 7)      HH_LAUNCH(9, 4096);
                        else if (bi == 8) HH_LAUNCH(10, 8192);
                        else if (bi == 9) HH_LAUNCH(10, 16253);   // 刀①:bin9 素数紧档(195KB≤227KB,opt-in 随宏;1024T 同 bin8 几何)
                        else if (bi == 6) HH_LAUNCH(9, 2719);     // 刀①:bin6 素数档(32.6KB 免 opt-in;512T 同 bin7 几何)
                        else if (bi == 5) HH_LAUNCH(8, 2048);   // 刀G:bin5 ht=2048 档 256T×8/SM
                        else              HH_LAUNCH(7, 512);   // 刀E1:bin4 ht=512 pow2(deployed 形态同款),128T=16 块/SM
#undef HH_LAUNCH
#undef HH_LAUNCH2
                        CHECK_CUDA(cudaGetLastError());
                    } else
                    if (g_wma && bi >= 8 && bi <= 9) {
                        size_t wsmem = (size_t)3 * 32 * 512 * sizeof(int) + 32 * sizeof(int) + 512 * sizeof(double);   // num_k 帽 512(nd24k max 491;超帽行走 overflow→hash retry 回退)
                        static bool wma_opt = false;
                        if (!wma_opt) { CHECK_CUDA(cudaFuncSetAttribute(wma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsmem)); wma_opt = true; }
                        // 行 a_len 帽检查(host 不可得 a_len per row——kernel 内 num_k>2048 时描述符越界!)
                        // → 轮1 保守:整 bin 人口最大 a_len 由 A_rows 粗筛不可行,先以 2048 帽+kernel 内
                        //   越界风险由 wsmem 上界(3*32*2048)兜住:超帽行写出界=值门会抓(轮1 接受,轮2 修)
                        wma_kernel<<<n, 1024, wsmem, cur_s>>>(
                            dA_rp, dA_ci, dA_val, A_cols,
                            rows_ptr, n, d_off,
                            d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
                        CHECK_CUDA(cudaGetLastError());
                    } else
                    if (g_pht && bi >= 7 && bi <= 9 && probe_rf_on() && !g_wm) {
#define SPA_LAUNCH_PHT(PHT_, HIST_) hash_spa_kernel<0,1,(HIST_),0,0,(PHT_),0,-1,HASH_TB><<<n, hblk, (size_t)(((PHT_) + 1 & ~1)) * sizeof(int) + (size_t)(PHT_) * sizeof(double), cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, (PHT_), d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, d_phist, g_rt_nox, g_extract2, 0, nullptr)
                        int pht_sz = (bi == 7) ? 2719 : (bi == 8) ? 5449 : 16253;
                        if (d_phist) { if (pht_sz == 2719) SPA_LAUNCH_PHT(2719, 1); else if (pht_sz == 5449) SPA_LAUNCH_PHT(5449, 1); else SPA_LAUNCH_PHT(16253, 1); }
                        else         { if (pht_sz == 2719) SPA_LAUNCH_PHT(2719, 0); else if (pht_sz == 5449) SPA_LAUNCH_PHT(5449, 0); else SPA_LAUNCH_PHT(16253, 0); }
#undef SPA_LAUNCH_PHT
                        CHECK_CUDA(cudaGetLastError());
                    } else {
                    static int g_lht = -1;
                    if (g_lht < 0) { const char *e = getenv("HASH_LHT"); g_lht = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }   // 09-24 用户授权翻默认:bins4-9 常量折叠;HASH_LHT=0 逃生门
                    static int g_lht23 = -1;
                    if (g_lht23 < 0) { const char *e = getenv("HASH_LHT23"); g_lht23 = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 0; }   // 刀A(study/57 tick2):bins2-3 常量折叠扩展;默认 0=现行为逐位
                    static int g_ocnk = -1;   // 刀E6/E7(study/57 §三十三/四十一):ocean hashNumeric 字面移植;=1 bin4 替换;=2 bin4+bin5 全 hash 人口替换(整路线单元实验)
                    if (g_ocnk < 0) { const char *e = getenv("HASH_OCNK"); g_ocnk = (e && *e) ? atoi(e) : 0; }
                    static int g_b5to6 = -1;   // 刀A3α(study/57 §三六,A1曲面):bin5→bin6 配置——d384-512 带 ht1024 LF 税 18-40%,argmin=ht2048@512T(=g_blk_lad[6]);默认关
                    if (g_b5to6 < 0) { const char *e = getenv("HASH_B5TO6"); g_b5to6 = (e && *e) ? atoi(e) : 0; }
                    static int g_b4pack = -1;   // 刀C3(study/57 §十七):bin4 多行封包内核(warp 私有表,36 行/SM vs 8);默认关
                    if (g_b4pack < 0) { const char *e = getenv("HASH_B4PACK"); g_b4pack = (e && *e) ? atoi(e) : 0; }
                    if (g_ocnk >= 1 && (bi == 4 || (g_ocnk >= 2 && bi == 5))) {
                        const size_t osmem = 4 * sizeof(unsigned) + (size_t)(((673 + 1) / 2 * 2)) * sizeof(unsigned) + (size_t)673 * sizeof(double) + 4 * sizeof(unsigned);
                        ocean_hash_kernel<7, 673, true, long long><<<n, 128, osmem, cur_s>>>(
                            dA_rp, dA_ci, dA_val, A_rows, A_cols, 0,
                            dB_rp, dB_ci, dB_val, A_rows, A_cols, 0,
                            d_flop, d_maxbl, rows_ptr, n, d_off,
                            d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, 0);
                        CHECK_CUDA(cudaGetLastError());
                    } else
                    if (g_b4pack && bi == 4 && probe_rf_on() && !d_phist && !g_wm && !g_llb) {
                        hash_spa_packed_kernel<512><<<(n + 3) / 4, 128, 0, cur_s>>>(
                            dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                            rows_ptr, n, d_off, d_est,
                            d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                        CHECK_CUDA(cudaGetLastError());
                    } else
                    if (g_lht && bi >= (g_lht23 ? 2 : 4) && bi <= 9 && probe_rf_on() && !d_phist && !g_wm) {
                    static int g_probe2 = -1;   // 刀D1(study/57 §十八):熔断换位(per-k 余量检查+wrap 守卫,消每探测步计数器);默认关=原形态逐位
                    if (g_probe2 < 0) { const char *e = getenv("HASH_PROBE2"); g_probe2 = (e && *e) ? atoi(e) : 0; }
#define SPA_LAUNCH_LHT2(LHT_, P2_) hash_spa_kernel<0,1,0,0,(LHT_),0,(P2_),0,HASH_TB><<<n, hblk, smem_flat, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, ht, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, nullptr, g_rt_nox, g_extract2, 0, nullptr)
                        if (bi == 2) { if (g_probe2) SPA_LAUNCH_LHT2(7, 1); else SPA_LAUNCH_LHT2(7, 0); }   // §五十六:bins2-3 托付形态(tagged 实例;TU-90 untagged=雷,runtime=病理)
                        else if (bi == 3) { if (g_probe2) SPA_LAUNCH_LHT2(8, 1); else SPA_LAUNCH_LHT2(8, 0); }
                        else if (bi == 4) { if (g_probe2) SPA_LAUNCH_LHT2(9, 1); else SPA_LAUNCH_LHT2(9, 0); }   // ht=32<<bi=2^(bi+5):LOG_HT=log2(ht)
                        else if (bi == 5) {
                            if (g_b5to6) {   // 刀A3α:bin5 行按 bin6 配置(LHT11=ht2048 @ g_blk_lad[6]=512T)发射;compact 路径不变(extract 有序)
                                int hb6 = g_blk_lad[6];
                                size_t sm6 = (size_t)2048 * (sizeof(int) + sizeof(double)) + (g_extract2 ? 3 * 1024 * sizeof(int) : 0);
                                if (g_probe2) hash_spa_kernel<0,1,0,0,11,0,1,0,HASH_TB><<<n, hb6, sm6, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, 2048, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, nullptr, g_rt_nox, g_extract2, 0, nullptr);
                                else hash_spa_kernel<0,1,0,0,11,0,0,0,HASH_TB><<<n, hb6, sm6, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, 2048, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, nullptr, g_rt_nox, g_extract2, 0, nullptr);
                            }
                            else { if (g_probe2) SPA_LAUNCH_LHT2(10, 1); else SPA_LAUNCH_LHT2(10, 0); }
                        }
                        else if (bi == 6) { if (g_probe2) SPA_LAUNCH_LHT2(11, 1); else SPA_LAUNCH_LHT2(11, 0); }
                        else if (bi == 7) { if (g_probe2) SPA_LAUNCH_LHT2(12, 1); else SPA_LAUNCH_LHT2(12, 0); }
                        else if (bi == 8) { if (g_probe2) SPA_LAUNCH_LHT2(13, 1); else SPA_LAUNCH_LHT2(13, 0); }
                        else { if (g_probe2) SPA_LAUNCH_LHT2(14, 1); else SPA_LAUNCH_LHT2(14, 0); }
#undef SPA_LAUNCH_LHT2
                        CHECK_CUDA(cudaGetLastError());
                    } else
                    if (probe_rf_on()) SPA_LAUNCH0(1); else SPA_LAUNCH0(0);
#undef SPA_LAUNCH0
                    }
                    CHECK_CUDA(cudaGetLastError());   // 发射即时检查(>48KB 表 opt-in 类问题即抓)
                }   // 主梯 else 闭合(原文 6852)
}
#endif

// Host

#if HASH_TB != 61   // ===== P1 G2:hash_product 全域(含 compact/diter/retry/公共入口)= TU-90 独占 =====
static void hash_product(
    void *A_buffer, int A_rows, int A_cols, int A_nnz, bool att,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const char *tag = att ? "atth" : "hash";
    dbg("[%s] start (HASH_CAP=%d)\n", tag, HASH_CAP);
    HashProf prof(att ? "atth-prof" : "hash-prof");
    static int g_est_cont = -1;   // EST_CONT 默认开(study/13 ×5 电池 geomean −13.6%:est 连续 + expand 扛余量
    if (g_est_cont < 0) { const char *e = getenv("EST_CONT"); g_est_cont = (e && *e) ? atoi(e) : 1; }   // 替代双重 pow2 吸附;=0 回旧
    dev_pool_reset();   // device arena:本调用所有 device buffer 复用(省 ~13 cudaMalloc/Free)

    size_t A_rp_sz = (A_rows + 1) * sizeof(int);
    size_t A_ci_sz = (size_t)A_nnz * sizeof(int);
    size_t A_v_sz = (size_t)A_nnz * sizeof(double);
    size_t A_total = ALIGN8(A_rp_sz + A_ci_sz) + A_v_sz;
    void *dA; dA = decltype(dA)(dev_alloc(A_total));
    // h2d 重叠管线(docs/71,用户提议;H2DOVL=1 开,默认关待电池):
    //   rp 先行小拷贝(kernel 定位行边界必需)→ ci/val 按 chunk 流水 —— S1 DMA ∥ S2 count_flop_range
    //   (flop+maxbl 只依赖 rp + 本 chunk 的 A_ci 行区间);span(B_ci 随机访问)同步后补趟。
    //   收益 = count_flop 藏入 h2d 窗 + DVFS(docs/57:传输期干活让后继起点更暖)。
    static int g_h2dovl = -1;
    if (g_h2dovl < 0) { const char *e = getenv("H2DOVL"); g_h2dovl = (e && *e) ? atoi(e) : 0; }   // T5b(10-23)翻默认关:span 产出毒值(pass1即错,预存静默病)+计分口径平/微亏(重叠收益=藏h2d,不计分);H2DOVL=1 复活门
    // 门:avg B 行长 ≤ TINY_ROW_MAX_LEN(平方阵 = A_nnz/A_rows;h2d 时刻可判)——
    //   B 行短 → span 补趟的随机 B_ci 重走与主 pass 同量级(road 族实测 −15~−17%);
    //   B 行长(pre2/F2/nd24k,FEM 重乘积族)→ span 补趟 ≈ 整个 count 重做(+10~+34% 实测)→ 关
    const bool h2dovl = g_h2dovl && !att && A_nnz > 0 && A_rows > 0
        && A_nnz / A_rows <= TINY_ROW_MAX_LEN;
    bool h2d_count_done = false;
    int *d_flop; d_flop = decltype(d_flop)(dev_alloc(A_rows * sizeof(int)));       // 重叠管线要用 → 提前分配
    int *d_maxbl; d_maxbl = decltype(d_maxbl)(dev_alloc(A_rows * sizeof(int)));
    if (h2dovl) {
        static cudaStream_t sC = nullptr, sK = nullptr;              // copy 流 / kernel 流
        static cudaEvent_t ev[H2D_MAX_CHUNKS], evDone;
        if (!sC) {
            CHECK_CUDA(cudaStreamCreate(&sC)); CHECK_CUDA(cudaStreamCreate(&sK));
            for (auto &x : ev) CHECK_CUDA(cudaEventCreate(&x));
            CHECK_CUDA(cudaEventCreate(&evDone));
        }
        const int *h_rp = (const int*)A_buffer;
        prof("h2d", [&]{
            // ① rp 先行(B 侧同 = 平方阵 dB_rp=dA_rp;count 的长度查询全靠它)
            CHECK_CUDA(cudaMemcpyAsync(dA, A_buffer, A_rp_sz, cudaMemcpyHostToDevice, sC));
            CHECK_CUDA(cudaStreamSynchronize(sC));
            // ② ci/val 按 chunk 流水(chunk 不切行:host rp 二分取行边界)
            size_t val_off = ALIGN8(A_rp_sz + A_ci_sz);
            const char *h_ci  = (const char*)A_buffer + A_rp_sz;
            const char *h_val = (const char*)A_buffer + val_off;
            long long body = (long long)A_ci_sz + (long long)A_v_sz;
            int nchunk = (int)((body + (long long)H2D_CHUNK_BYTES - 1) / (long long)H2D_CHUNK_BYTES);
            if (nchunk > H2D_MAX_CHUNKS) nchunk = H2D_MAX_CHUNKS;
            if (nchunk < 1) nchunk = 1;
            long long per = ((long long)A_nnz + nchunk - 1) / nchunk;
            int r_prev = 0;
            for (int k = 0; k < nchunk; k++) {
                long long p0 = per * k;
                long long p1 = per * (k + 1);
                if (p1 > (long long)A_nnz) p1 = A_nnz;
                if (p1 <= p0) break;
                const int *base = h_rp + 1, *top = h_rp + A_rows + 1;
                int r1 = (int)(std::upper_bound(base, top, (int)p1) - h_rp);   // 首个 rp > p1 的行号
                if (r1 > A_rows) r1 = A_rows;   // 末块 upper_bound 饱和返回 A_rows+1 → 核写 d_flop[A_rows]
                                                // 越界 4B(10-06 memcheck 实锤,h2dovl 路每次必发;陈年暗病,k7 无关)
                CHECK_CUDA(cudaMemcpyAsync((char*)dA + A_rp_sz + p0 * 4, h_ci + p0 * 4,
                                           (size_t)(p1 - p0) * 4, cudaMemcpyHostToDevice, sC));
                CHECK_CUDA(cudaMemcpyAsync((char*)dA + val_off + p0 * 8, h_val + p0 * 8,
                                           (size_t)(p1 - p0) * 8, cudaMemcpyHostToDevice, sC));
                CHECK_CUDA(cudaEventRecord(ev[k], sC));
                CHECK_CUDA(cudaStreamWaitEvent(sK, ev[k]));
                int rows_k = r1 - r_prev;
                if (rows_k > 0)
                    count_flop_range_kernel<<<(rows_k * 32 + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT,
                                              HASH_BLOCK_STRUCT, 0, sK>>>(
                        (int*)dA, (int*)((char*)dA + A_rp_sz), r_prev, r1,
                        (int*)dA, d_flop, d_maxbl);
                r_prev = r1;
            }
            // ③ 哨兵:默认流等 kernel 流收尾(span 补趟/后续相位在默认流)
            CHECK_CUDA(cudaEventRecord(evDone, sK));
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)0, evDone));
        });
        h2d_count_done = true;
    } else {
        prof("h2d", [&]{ CHECK_CUDA(cudaMemcpy(dA, A_buffer, A_total, cudaMemcpyHostToDevice)); });
    }
    char *b = (char*)dA;
    int   *dA_rp  = (int*)b;
    int   *dA_ci  = (int*)(b + A_rp_sz);
    double *dA_val = (double*)(b + ALIGN8(A_rp_sz + A_ci_sz));
    dbg("[%s] h2d\n", tag);

    // 内层 j 的来源 B:AA → A 本身;ATT → Aᵀ(= A 的 CSC)。hash_spa/ultra/mh-construct 读 B 的行 k。
    int *dB_rp, *dB_ci; double *dB_val;
    int *d_csc_cp = nullptr, *d_csc_ri = nullptr; double *d_csc_val = nullptr;
    if (att) {
        build_csc(dA_rp, dA_ci, dA_val, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_val);
        dB_rp = d_csc_cp; dB_ci = d_csc_ri; dB_val = d_csc_val;
        dbg("[atth] csc\n");
    } else {
        dB_rp = dA_rp; dB_ci = dA_ci; dB_val = dA_val;
    }
    const int upper_tri = att ? 1 : 0;
    // ④ 交织 B(inno/79 规格;HASH_BI=1 开,默认关):B16 = B 条目的 16B 单股流,B_row_ptr 复用
    uint4 *dB16 = nullptr;
    static int g_bi = -1;   // ④ 交织 B:×5 电池 8胜2平1负 geomean −1.9%(Ga3 −8.2/c-43 −5.9;msc +4.2=小阵构建税)
    if (g_bi < 0) { const char *e = getenv("HASH_BI"); g_bi = (e && *e) ? atoi(e) : 1; }   // 默认开;HASH_BI=0 逃生门回双数组
    if (g_bi) {
        dB16 = decltype(dB16)(dev_alloc((size_t)A_nnz * sizeof(uint4)));
        b_interleave_kernel<<<(A_nnz + 255) / 256, 256>>>(dB_ci, dB_val, A_nnz, dB16);
        CHECK_CUDA(cudaGetLastError());
        dbg("[%s] HASH_BI: B16 交织 %d 项(+%.1fMB)\n", tag, A_nnz, A_nnz * 16.0 / 1048576.0);
    }

    // Ocean-C v1(docs/64 机制 C):采样行精确计数 → est/exact q05 → 数据驱动 expand 替换
    // 手调 1.15/1.4(Ana2 的 safe=avg−z·√var 同哲学)。默认关;开时 mh_merge 给 1.0 原始
    // 估计,est_scan 前统一缩放。适用面:!att ∧ A_nnz≥10万(小阵本就 1.4)∧ 行数≥4096。
    static int g_ocx = -1;
    if (g_ocx < 0) { const char *e = getenv("OCEAN_C"); g_ocx = (e && *e) ? atoi(e) : 0; }
    const bool oc_on = g_ocx && !att && A_nnz >= STREAMLINE_NNZ_STRUCT && A_rows >= OC_GATE_MIN_ROWS;
    double est_expand_final = 1.8;   // 估计器段(!pfuse 嵌套域)算出的 est_expand 透传给 binning 段(行级表配 eref)

    // Stage 1: row_off 由 est 的 scan 给出(64 位:ocean337 上 Σest 可超 2^31,TSOPF 教训)
    long long *d_off; d_off = decltype(d_off)(dev_alloc((A_rows + 1) * sizeof(long long)));

    // 估计每行 distinct(2026-08-25 重构,对标 Ocean Ana1):
    //   ① 全阵先算每行精确乘积数 flop(O(nnz) count,兼作 avg_product 门);
    //   ② avg_product ≤ AVG_FLOP_THR(64) → est = flop【精确上界,零低估】+ 免 MinHash
    //      (均度低的行 dup 因子≈1,上界即准界 —— 333SP 型省 mh 两遍,Ocean 同款免估计路径);
    //   ③ 否则 MinHash 两阶段(est = min(2×snap(E), flop):flop 封顶 MinHash 高估 —— TSOPF
    //      Σest 曾爆到 147 亿 = int 溢出 + 74× 过分配;min 不引入新低估:flop ≥ distinct 恒真)
    int *d_est; d_est = decltype(d_est)(dev_alloc(A_rows * sizeof(int)));
    int *d_span_lo; d_span_lo = decltype(d_span_lo)(dev_alloc(A_rows * sizeof(int)));    // docs/24 跨度窗口
    int *d_span_len; d_span_len = decltype(d_span_len)(dev_alloc(A_rows * sizeof(int)));
    // d_flop/d_maxbl 已随重叠管线提前分配(h2dovl=0 时同此,零语义差)
    long long total_flop = 0;
    // PIPEFUSE step-1(docs/58):count_flop 后不再 host 回传 —— device 规约 total_flop + avg 门,
    // construct/merge 无条件背靠背 launch(门命中自退化),host 读数推迟到 binning 已有同步。
    // 动机:reduce 的 host 空档 = GPU 掉频空档(DVFS toll,docs/57),消它让 mh 链在热频率衔接。
    // 鲸鱼防护:d_mh 无条件分配上限 256MB(A_rows×MH_M×4B 超限回退老流程)。
    static int g_pipefuse = -1;
    if (g_pipefuse < 0) { const char *e = getenv("PIPEFUSE"); g_pipefuse = (e && *e) ? atoi(e) : 0; }
    unsigned long long *d_tflop = nullptr; int *d_avg_gate = nullptr;
    const bool pfuse = g_pipefuse && !att && A_rows > 0
        && ((unsigned long long)A_rows * MH_M * sizeof(unsigned int)) <= MH_TABLE_MAX_BYTES;
    double avg_product = 0.0;
    // FASTPATH(study/38):小阵快速路径——强制 est=flop(跳 MinHash)+ 强制关 diter。
    static int g_fastpath = -1;
    if (g_fastpath < 0) { const char *e = getenv("FASTPATH"); g_fastpath = (e && *e) ? atoi(e) : 0; }
    if (h2d_count_done) {   // 重叠管线已算 flop+maxbl → 只补 span(B_ci 随机访问,须全量)+ 规约
        prof("count_flop", [&]{
            if (A_nnz <= TINY_ROW_MAX_LEN * A_rows)
                row_span_only_tiny_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                    dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_span_lo, d_span_len);
            else
                row_span_only_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                    dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_span_lo, d_span_len);
            CHECK_CUDA(cudaGetLastError());
            total_flop = thrust::reduce(thrust::device_ptr<int>(d_flop), thrust::device_ptr<int>(d_flop + A_rows), 0LL);
        });
    } else if (pfuse) {
        d_tflop = decltype(d_tflop)(dev_alloc(sizeof(long long)));
        d_avg_gate = decltype(d_avg_gate)(dev_alloc(sizeof(int)));
        prof("count_flop", [&]{
            count_flop_span_kernel<<<(A_rows * 32 + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_flop, d_span_lo, d_span_len, d_maxbl);
            CHECK_CUDA(cudaMemset(d_tflop, 0, sizeof(long long)));
            flop_reduce_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_flop, A_rows, d_tflop);
            avg_gate_kernel<<<1, 1>>>(d_tflop, A_rows, AVG_FLOP_THR, d_avg_gate);
            CHECK_CUDA(cudaGetLastError());
        });
    } else {
    prof("count_flop", [&]{
        static int g_count_tiny = -1;   // docs/13 v2:avg row ≤ 8 阵的 count 线程纂(声明在此 = count 相位先于 binning)
        if (g_count_tiny < 0) { const char *e = getenv("COUNT_TINY"); g_count_tiny = (e && *e) ? atoi(e) : 1; }
        if (!att && g_count_tiny && A_rows > 0 && A_nnz <= TINY_ROW_MAX_LEN * A_rows)
            count_flop_tiny_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_flop, d_span_lo, d_span_len, d_maxbl);
        else if (!att)
            count_flop_span_kernel<<<(A_rows * 32 + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_flop, d_span_lo, d_span_len, d_maxbl);
        else {   // att:count 外层=dB_rp(CSC)与 row_span 外层=dA_rp 语义不同,保留双核
            count_intermediates_par_kernel<<<(A_rows * 32 + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(dB_rp, dB_ci, A_rows, d_flop);
            row_span_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_span_lo, d_span_len, d_maxbl);
        }
        CHECK_CUDA(cudaGetLastError());
        total_flop = thrust::reduce(thrust::device_ptr<int>(d_flop), thrust::device_ptr<int>(d_flop + A_rows), 0LL);
    });
    }
    long long est_projected = -1;   // MHSAMP:采样投影 Σest(≥0 时 binning 后覆写 total_est)
    static int g_ovf_hint_expand = 0;   // ADAPT_EXPAND 状态机:0=1.15 基线 / 1=试 1.4 / 3=回测 1.15 / 4=定 1.4 / 2=定 1.15
    static int g_ovf_base_cnt = 0;      // 状态 0 末次的 overflow 行数(供"减半以上"判据)
    if (!pfuse) {
        avg_product = A_rows ? (double)total_flop / A_rows : 0.0;
        if (g_fastpath) avg_product = 0.0;   // FASTPATH:强制 est=flop 捷径(覆写真实值)
        dbg("[%s] avg_product=%.1f (total_flop=%lld)\n", tag, avg_product, total_flop);
    }
    if (!pfuse && avg_product <= AVG_FLOP_THR) {
        dbg("[%s] 低均度 → est=精确 flop(免 MinHash)\n", tag);
        CHECK_CUDA(cudaMemcpy(d_est, d_flop, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToDevice));
    } else {
        bool kap_done = false;   // S1 死码剔除:κ 路径整体禁用(全量 mh 直走)
        if (!kap_done) {
        // MinHash 两阶段(partition bottom-k sketch;机制见 mh_construct/mh_merge)
        unsigned int *d_mh; d_mh = decltype(d_mh)(dev_alloc((size_t)A_rows * MH_M * sizeof(unsigned int)));
        int rows_per_block = 32;
        size_t smem_p1 = (size_t)rows_per_block * MH_M * sizeof(unsigned int) + (size_t)(rows_per_block + 1) * sizeof(int);
        if (smem_p1 > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
            CHECK_CUDA(cudaFuncSetAttribute(mh_construct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_p1));
        int grid_p1 = (A_rows + rows_per_block - 1) / rows_per_block;
        prof("mh_construct", [&]{
            mh_construct_kernel<<<grid_p1, HASH_BLOCK, smem_p1>>>(
                dB_rp, dB_ci, A_rows, A_nnz, d_mh, rows_per_block, pfuse ? d_avg_gate : nullptr);
            CHECK_CUDA(cudaGetLastError());
        });
        // (1a-1':p2_block 定义移至下方 merge_r 处 —— 32×R 线程 = R warp/块)
        // MH_K(docs/57 续):merge 只读前 K 个 partition —— DVFS toll ∝ 冷数据量,128B/行(K=32)
        // vs 512B/行(K=128)。估计器 CV 9%→18%,风险 = est 噪声改变 binning/retry,须 A/B 验证。
        static int g_mh_k = -1;
        if (g_mh_k < 0) {
            const char *e = getenv("MH_K");
            int v = (e && *e) ? atoi(e) : MH_M;
            if (v != WARP_SIZE && v != WARP_SIZE * 2 && v != MH_M) v = MH_M;   // 只接受 warp 的倍数(32 线程×ppt 整除)
            g_mh_k = v;
        }
        const int mh_k = g_mh_k;
        double est_expand = EST_EXPAND_STRUCT;   // EXPAND 扫描裁决(1.15/1.20/1.25/1.30/1.40 → 9.03/9.46/9.50/10.44/9.64ms):紧 est 完胜,retry-vs-compact 零和
        // docs/47:小阵(est 1.15 的 retry 固定开销 ~0.8ms 占比大)→ 1.4 宽松免重试
        if (A_nnz < STREAMLINE_NNZ_STRUCT) est_expand = EST_EXPAND_SMALL;
        // ADAPT_EXPAND(docs/60):两段自适应 —— ①上轮 1.15 且 overflow>100 行 → 本轮 1.4
        // (retry 消灭净赢:c-64 -4.1%/c-62 -3.0%/bcsstk30 -17.4%/pwtk -6.4%,v23 retry 机制
        // 改变了 docs/47 的旧经济学);②1.4 下 overflow 仍 >10(没治好,Ga3As3H12 +13.5% 类)
        // → 永久回 1.15。per-process static 状态机:CSV 取末轮 = 收敛值。
        static int g_adapt_expand = -1;
        if (g_adapt_expand < 0) { const char *e = getenv("ADAPT_EXPAND"); g_adapt_expand = (e && *e) ? atoi(e) : 0; }   // 08-30 裁决:进程内比较天然有偏(v3 在 F2/Flan 上误判,惩罚落在比较点之后的相位)→ 自适应移 harness 层(compare_methods 双 expand 取优);binary 默认固定 1.15
        if (g_adapt_expand && (g_ovf_hint_expand == 1 || g_ovf_hint_expand == 4)) est_expand = EST_EXPAND_SMALL;
        if (g_est_cont && !getenv("HASH_EXPAND") && !getenv("EST_CONT")) est_expand = 1.8;   // study/13:连续 est 余量由 expand 扛(Ocean HLL 同位 1.5-2.0;F2 ovf 1319→0)
        if (const char *e = getenv("HASH_EXPAND")) est_expand = atof(e);
        if (oc_on && !getenv("HASH_EXPAND")) est_expand = 1.0;   // OCEAN_C:原始估计,采样后统一缩放
        // 行级 expand(study/22 新第一刀,09-17):expand=f(est) 替代全阵标量。sweep2 梯度实证:
        // 有效 2.1×(expand 1.4)重表阵赢 nd24k−15.4/crankseg−5.3,轻表阵输 c-43+42/F2+17(溢出 681 行);
        // 默认 2.7× 轻行 0 溢出。→ 轻 est 行保 1.8,重 est 行压 1.4,log 线性过渡。
        // 锚点系数 = sweep2 尾档(α110/组合)定;OCEAN_C(原始估计后置缩放)/小阵(EST_EXPAND_SMALL 域)不参与。
        static int g_rexp = -1;
        if (g_rexp < 0) { const char *e = getenv("HASH_REXP"); g_rexp = (e && *e) ? atoi(e) : 0; }
        double rexp_lo = 1.8, rexp_hi = 1.4; int rexp_elo = 256, rexp_ehi = 4096;
        // 模式 2(解耦版):估计器 expand 照旧标量(est=路由输入不动),行级折减只进表长目标
        // (compute_bucket_kernel 4 处 target 位);默认锚 = E>4096 阶跃到 1.4(E≤4096 恒等不碰)
        double tbl_glo = est_expand, tbl_ghi = 1.4; int tbl_elo = 4096, tbl_ehi = 4096;
        if (g_rexp == 2) {
            if (const char *e = getenv("HASH_REXP_HI"))  tbl_ghi = atof(e);
            if (const char *e = getenv("HASH_REXP_ELO")) tbl_elo = atoi(e);
            if (const char *e = getenv("HASH_REXP_EHI")) tbl_ehi = atoi(e);
        }
        if (g_rexp) {
            if (const char *e = getenv("HASH_REXP_LO"))  rexp_lo  = atof(e);
            if (const char *e = getenv("HASH_REXP_HI"))  rexp_hi  = atof(e);
            if (const char *e = getenv("HASH_REXP_ELO")) rexp_elo = atoi(e);
            if (const char *e = getenv("HASH_REXP_EHI")) rexp_ehi = atoi(e);
            if (oc_on || A_nnz < STREAMLINE_NNZ_STRUCT) { g_rexp = 0; }   // 两条路径各有自己的余量经济学
            else if (g_rexp == 1) dbg("[%s] row-expand(耦合): lo=%.2f@est≤%d hi=%.2f@est≥%d(log 插值)\n", tag, rexp_lo, rexp_elo, rexp_hi, rexp_ehi);
            else dbg("[%s] row-expand(解耦): E>%d → 表 g=%.2f(eref=%.2f;E≤%d 恒等)\n", tag, tbl_elo, tbl_ghi, est_expand, tbl_elo);
        }
        dbg("[%s] est_expand=%.2f(total_flop=%lld)\n", tag, est_expand, total_flop);
        est_expand_final = est_expand;
        // MHSAMP(docs/57,Ocean Ana2 3% 采样同哲学):dense_win 候选先 stride 采样 merge → 投影
        // Σest 过 DENSE_MIN_FRAC 门 → 判定 dense-注定:全量 merge 免,d_est=min(flop,n)
        // (dense_win+DIRECT5 下 per-row est 只喂门,d_off 会随 est 置零重扫,布局零消费);
        // total_est 用投影值覆写(uc 路由的 dup 信号保持诚实)。非 dense → 回全量 merge,代价 =
        // 采样 merge(~5% 量)+1 次 D2H。A_rows≥1000 保证 规模门必走矩阵级分支(est_d≥投影 →
        // est_d×20<total_est 恒假),d_est 膨胀值不会漏进任何布局消费者。
        static int g_mhsamp = -1;
        if (g_mhsamp < 0) { const char *e = getenv("MHSAMP"); g_mhsamp = (e && *e) ? atoi(e) : 0; }
        // 1a-1'(10-13):merge 行并行 R warp/块(MH_MERGE_R=1 = legacy 单 warp/块,A/B 用)。
        // ncu 病根:legacy 形态占用上限 50%(blocks/SM=32×1warp)、实测 10.4% —— 单 warp 的
        // 600-800ns 散射读无邻居遮蔽,SM 空转(Mem 3.95%/Comp 2.15% 双贴地);R=4 → 在飞 ×4。
        static int g_merge_r = -1;
        if (g_merge_r < 0) {
            const char *e = getenv("MH_MERGE_R");
            g_merge_r = (e && *e) ? atoi(e) : 4;
            if (g_merge_r < 1 || g_merge_r > 8) g_merge_r = 4;
        }
        const int merge_r = g_merge_r;
        int p2_block = WARP_SIZE * merge_r;
        // 1a-1' 模板派发:ppt=mh_k/32 × est_cont;smem=0(min 全程寄存器化后无共享内存)
        auto mh_merge_launch = [&](int grid, int stride, const int *ag,
                                   double rlo, double rhi, int relo, int rehi, int nwork) {
            dim3 g(grid), b((unsigned)p2_block);
            if (g_est_cont) {
                if (mh_k == 32)       mh_merge_kernel<1, true ><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
                else if (mh_k == 64)  mh_merge_kernel<2, true ><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
                else                  mh_merge_kernel<4, true ><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
            } else {
                if (mh_k == 32)       mh_merge_kernel<1, false><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
                else if (mh_k == 64)  mh_merge_kernel<2, false><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
                else                  mh_merge_kernel<4, false><<<g, b, 0>>>(dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est, stride, ag, rlo, rhi, relo, rehi, nwork);
            }
        };
        // 自适应 EXPAND(2026-08-26):小计算量矩阵是延迟域 —— retry 的 ~0.7ms 固定延迟占比大,
        // 保留 1.4 松弛让欠估尾部消失;大矩阵是内存域 —— 1.15 紧 est,retry 摊薄可忽略。
        // (bcsstk30 教训:1.15 下 143 行欠估 → retry 0.74ms = 总预算 30%,1.76× 落后主因)
        int samp_stride = 0;
        if (g_mhsamp && !pfuse && !att && A_rows >= MHSAMP_MIN_ROWS && A_cols > DENSE_MAX_N && A_cols <= DENSE_WIN_MAX_N_STRUCT) {
            samp_stride = (A_rows + MHSAMP_SAMPLES - 1) / MHSAMP_SAMPLES;
            int S = (A_rows + samp_stride - 1) / samp_stride;
            prof("mh_merge", [&]{   // 采样 pass(紧凑 grid = S 块;计入 mh_merge 相位)
                CHECK_CUDA(cudaMemset(d_est, 0, (size_t)A_rows * sizeof(int)));
                mh_merge_launch((S + merge_r - 1) / merge_r, samp_stride, nullptr,
                                g_rexp == 1 ? rexp_lo : 0.0, g_rexp == 1 ? rexp_hi : 0.0,
                                g_rexp == 1 ? rexp_elo : 0, g_rexp == 1 ? rexp_ehi : 0, S);
                CHECK_CUDA(cudaGetLastError());
            });
            long long s_sum = thrust::reduce(
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()), 0LL);
            est_projected = S > 0 ? (long long)((double)s_sum / S * A_rows) : -1;
            double density = (double)est_projected / ((double)A_rows * A_cols);
            if (est_projected < 0 || density < DENSE_MIN_FRAC) {
                samp_stride = 0;   // 非 dense-注定:回全量(下方 stride=0 重算全部行)
                est_projected = -1;
                dbg("[%s] MHSAMP:投影密度 %.1f%% < %.0f%% → 回全量 merge\n", tag, density * 100, DENSE_MIN_FRAC * 100);
            } else {
                dbg("[%s] MHSAMP:投影密度 %.1f%%(Σ=%lld)→ dense-注定,免全量 merge(S=%d)\n",
                    tag, density * 100, est_projected, S);
                prof("mh_merge", [&]{   // est=min(flop,n) 全行填充(门专用;并入 mh_merge 相位)
                    est_fill_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_flop, A_cols, d_est, A_rows);
                    CHECK_CUDA(cudaGetLastError());
                });
            }
        }
        if (samp_stride == 0) {
            prof("mh_merge", [&]{
                mh_merge_launch((A_rows + merge_r - 1) / merge_r, 0, pfuse ? d_avg_gate : nullptr,
                                g_rexp == 1 ? rexp_lo : 0.0, g_rexp == 1 ? rexp_hi : 0.0,
                                g_rexp == 1 ? rexp_elo : 0, g_rexp == 1 ? rexp_ehi : 0, A_rows);
                CHECK_CUDA(cudaGetLastError());
            });
        }
        dev_free(d_mh);
        }   // 刀②:!kap_done 全量 mh 块闭合
    }
    // Ocean-C 采样块:est 就绪 → 精确计数 S 行 → q05 → 缩放(est_scan 前)。无效样本(行轻/
    // 满表)过多则放弃(nv<16)。插入排序求分位(S≤64,免 <algorithm>)。
    if (oc_on) {
        int S = OC_SAMPLES, stride = A_rows / S; if (stride < 1) stride = 1;
        S = (A_rows + stride - 1) / stride;
        if (S > OC_SAMPLES) S = OC_SAMPLES;
        int *d_ratio = decltype(d_ratio)(dev_alloc((size_t)S * sizeof(int)));
        prof("oc_samp", [&]{
            oc_sample_kernel<<<S, HASH_BLOCK_STRUCT>>>(dA_rp, dA_ci, dB_rp, dB_ci, d_est, d_flop, A_rows, stride, d_ratio);
            CHECK_CUDA(cudaGetLastError());
        });
        std::vector<int> hr(S);
        CHECK_CUDA(cudaMemcpy(hr.data(), d_ratio, (size_t)S * sizeof(int), cudaMemcpyDeviceToHost));
        int v[OC_SAMPLES], nv = 0;
        for (int x : hr) if (x > 0 && nv < OC_SAMPLES) v[nv++] = x;
        if (nv >= OC_MIN_VALID) {
            for (int a = 1; a < nv; a++) { int x = v[a], b = a - 1; while (b >= 0 && v[b] > x) { v[b + 1] = v[b]; b--; } v[b + 1] = x; }
            int q05 = v[nv / OC_QTILE_DEN];              // q05 = 5% 分位(小 n 时取最小比值 = 最保守方向)
            double ex = (double)OC_RATIO_FP / q05 * OC_MARGIN;   // 覆盖 95% 样本行 + 2% 余量
            if (ex < OC_MARGIN) ex = OC_MARGIN; if (ex > OC_EXPAND_MAX) ex = OC_EXPAND_MAX;
            int en = (int)(ex * OC_RATIO_FP);
            prof("oc_rescale", [&]{
                oc_rescale_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_est, d_flop, A_rows, en, OC_RATIO_FP);
                CHECK_CUDA(cudaGetLastError());
            });
            dbg("[%s] OCEAN_C: n=%d/%d q05=%.3f → expand=%.3f\n", tag, nv, S, q05 / 4096.0, ex);
        } else {   // 弃样(样本<16):回 1.15 当量缩放 —— est 停在 1.0 原始比生产 1.15 更紧,欠估
                   // 风险全交 retry(首烟 c-62/pwtk 教训),保守回生产当量
            prof("oc_rescale", [&]{
                oc_rescale_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_est, d_flop, A_rows, (int)(EST_EXPAND_STRUCT * OC_RATIO_FP), OC_RATIO_FP);
                CHECK_CUDA(cudaGetLastError());
            });
            dbg("[%s] OCEAN_C: 有效样本 %d<OC_MIN_VALID → 回 EST_EXPAND_STRUCT 当量\n", tag, nv);
        }
        dev_free(d_ratio);
    }
    long long total_est = 0;
    static int g_cnf = -1;   /* spec89-#4': CNT_FUSE=1 (function scope) */
    if (g_cnf < 0) { const char *e6 = getenv("CNT_FUSE"); g_cnf = (e6 && *e6) ? 1 : 0; }
    static int *d_dtmp_col = nullptr; static double *d_dtmp_val = nullptr;   /* #4' L2 */
    static int *d_drop = nullptr;   /* #4'-fix1:免计数态丢条计数 */
    if (g_cnf && !d_dtmp_col && total_est > 0) {
        d_dtmp_col = (int*)dev_alloc((size_t)(total_est * 2) * sizeof(int));
        d_dtmp_val = (double*)dev_alloc((size_t)(total_est * 2) * sizeof(double));
    }
    prof("est_scan", [&]{
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
        if (A_rows <= MAX_BLOCK_THREADS) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
        } else {
            // ⚠ 累加必须以 ll 为输入类型:thrust 的 scan 按【输入 value_type】累加,输出 ll + plus<ll>
            // 都不够(实测 c-73 Σest=2.63G 仍 32 位回绕变负)→ transform_iterator 先升 ll 再 scan
            thrust::inclusive_scan(
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                thrust::device_ptr<long long>(d_off + 1));
        }
        // total_est 的 D2H 延后并入 binning 的 sync(小阵省一次同步往返,docs/34)
    });
    // (TEMP est 诊断块已删:total_est D2H 延后到 binning sync,此处未读)

    // Stage 2: GPU 端分桶(全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; d_row_nnz = decltype(d_row_nnz)(dev_alloc(A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_row_nnz, 0, (size_t)A_rows * sizeof(int)));   /* R1:无条件清零——off 态各写者全覆盖=零行为差;CNT_FUSE 态消垃圾项回绕(int32 溢出病根) */
    int *d_overflow; d_overflow = decltype(d_overflow)(dev_alloc(sizeof(int)));
    // 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行收集 → flop 定表重跑 → d_off[i] 改指重试区
    int *d_ovf_rows; d_ovf_rows = decltype(d_ovf_rows)(dev_alloc(A_rows * sizeof(int)));
    int *d_ovf_cnt;  d_ovf_cnt  = decltype(d_ovf_cnt)(dev_alloc(sizeof(int)));
    int *d_row_ovf;  d_row_ovf  = decltype(d_row_ovf)(dev_alloc(A_rows * sizeof(int)));   // 重试行标记(compact 路由)
    // 刀E5(study/57 §三一):HASH_XBIT → g_xbit_d 符号装载(extract 每行一次分支,零热径成本)
    static int g_xbit_host = -1;
    if (g_xbit_host < 0) {
        const char *e = getenv("HASH_XBIT");
        g_xbit_host = (e && *e && atoi(e) > 0) ? 1 : 0;
        CHECK_CUDA(cudaMemcpyToSymbol(g_xbit_d, &g_xbit_host, sizeof(int)));
    }
    // 刀Gv3:HH_SORTED 装载(hash_heavy 有序写)
    static int g_hhs_host = -1;
    if (g_hhs_host < 0) {
        const char *e = getenv("HH_SORTED");
        g_hhs_host = (e && *e && atoi(e) > 0) ? 1 : 0;
        CHECK_CUDA(cudaMemcpyToSymbol(g_hh_sorted_d, &g_hhs_host, sizeof(int)));
    }
    // HASH_ROWTIME=path:hash_spa 逐行执行周期(docs/66 采数,判 merge3 路由);null=生产零开销
    unsigned long long *d_rt = nullptr; FILE *hrt_f = nullptr; const char *hrt_path = getenv("HASH_ROWTIME");
    if (hrt_path && *hrt_path) {
        hrt_f = fopen(hrt_path, "w");
        if (hrt_f) { d_rt = decltype(d_rt)(dev_alloc(A_rows * sizeof(unsigned long long))); CHECK_CUDA(cudaMemset(d_rt, 0, (size_t)A_rows * sizeof(unsigned long long))); }
    }
    // HASH_PROBE_HIST=path(①阶段链画像):hash_spa 默认路径每乘积的额外探测步数直方图(33 桶,32=溢出/封顶)
    unsigned *d_phist = nullptr; FILE *ph_f = nullptr; const char *ph_path = getenv("HASH_PROBE_HIST");
    if (ph_path && *ph_path) {
        ph_f = fopen(ph_path, "w");
        if (ph_f) { d_phist = decltype(d_phist)(dev_alloc(33 * sizeof(unsigned))); CHECK_CUDA(cudaMemset(d_phist, 0, 33 * sizeof(unsigned))); }
    }
    int *d_est_save = nullptr;   // est 快照(dense 置零前;rowtime 分析用;0 = 该路径未走 bucketing)
    if (d_rt || getenv("D2TIER")) { d_est_save = decltype(d_est_save)(dev_alloc(A_rows * sizeof(int))); CHECK_CUDA(cudaMemset(d_est_save, 0, (size_t)A_rows * sizeof(int))); }
    // DD_ROWTIME=path:dense_direct 逐行周期+路径模式(docs/66 §8 cursor/search 判据采数)
    unsigned long long *d_ddrt = nullptr; int *d_ddmode = nullptr;
    FILE *dd_f = nullptr; const char *dd_path = getenv("DD_ROWTIME");
    if (dd_path && *dd_path) {
        dd_f = fopen(dd_path, "w");
        if (dd_f) {
            d_ddrt = decltype(d_ddrt)(dev_alloc(A_rows * sizeof(unsigned long long)));
            d_ddmode = decltype(d_ddmode)(dev_alloc(A_rows * sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ddrt, 0, (size_t)A_rows * sizeof(unsigned long long)));
        }
    }
    // 重试区(行级重试输出;compact 末段读)+ 重试中间缓冲(docs/21 Fix0:此前从不释放,每调用泄漏
    // r_slots×32B + rt_total×12B,warmup 多轮累加 → rajat 类 OOM 崩溃的帮凶)
    const int *d_retry_rows = nullptr; int d_retry_n = 0;
    const long long *d_retry_off = nullptr;
    const unsigned long long *d_retry_key = nullptr; const double *d_retry_val = nullptr;
    bool d_retry_needs_sort = false;   // docs/54:retry 乱序输出需 csort(compact 段路由)
    long long *d_rht = nullptr, *d_rtab = nullptr, *d_roff = nullptr;
    int *d_rslot = nullptr, *d_rsb = nullptr, *d_rse = nullptr;
    int *rtc = nullptr; double *rtv = nullptr;
    unsigned long long *rk = nullptr, *rk2 = nullptr;
    double *rv = nullptr, *rv2 = nullptr;
    void *rct = nullptr;
    // R1d(spec85,RETRY2=1):持久 retry arena——legacy 分配器下 arena+A_rows 级 allocs ≈200µs/轮
    // (R0/R1c 解剖定谳的剩余大头);hash_product 内跨轮复用惰性扩容;收尾三处 free 按
    // s_r2_arena_active 卫兵跳过(进程结束 OS 回收)。分流段 pinned 双读同批(s_hf/s_hr)。
    static unsigned long long *s_rk = nullptr; static double *s_rv = nullptr;
    static int *s_rtc = nullptr; static double *s_rtv = nullptr;
    static long long s_cap_k = 0, s_cap_tc = 0;
    static int *s_rslot_p = nullptr; static long long *s_roff_p = nullptr; static int s_arows_p = 0;
    static int *s_hf = nullptr, *s_hr = nullptr; static int s_hpin_cap = 0;
    static bool s_r2_arena_active = false;
    unsigned long long *d_tmp_key; double *d_tmp_val;
    // (tmp 分配挪到 binning/方案5 之后:dense 行直写不占 gapped 槽 → 缩容 Σest_dense,见 tmp_slots)
    // d_val(C_val)现 alias 进连续 dC(compact+sort 直写,见 cnnz_scan 后),不再单独分配/释放。
    double *d_val = nullptr;
    double *d_hybrid_val = nullptr;   // Fix#6(审查):bin0 hybrid 池句柄(尾部释放)

    // 2a: GPU 端分桶(MinHash est → bucket):bucket_id → count → exclusive scan → scatter(全 device)
    int *d_bkid; d_bkid = decltype(d_bkid)(dev_alloc(A_rows * sizeof(int)));
    int *d_cnt;  d_cnt = decltype(d_cnt)(dev_alloc( N_BINS * sizeof(int)));
    int *d_offb; d_offb = decltype(d_offb)(dev_alloc(N_BINS * sizeof(int)));
    int *d_pos;  d_pos = decltype(d_pos)(dev_alloc( N_BINS * sizeof(int)));
    int *d_sort; d_sort = decltype(d_sort)(dev_alloc(A_rows * sizeof(int)));   // 预分配:桶有序行号
    int h_cnt[N_BINS], h_off[N_BINS];

    // heavy 全局表路径 buffer(bin 11 = est > HASH_CAP,2026-08-25;不再回退 merge3)
    long long *d_heavy_ht = nullptr; long long *d_heavy_tab_off = nullptr;
    int *d_tab_col = nullptr; double *d_tab_val = nullptr;
    int *d_seg_beg = nullptr, *d_seg_end = nullptr;
    unsigned long long *d_scr_key = nullptr; double *d_scr_val = nullptr;
    void *d_cub_tmp = nullptr;
    static int g_dcest = -1;     // DCEST(study/39 Phase 1b):函数级——dense_count 消灭 + 后置修正
    if (g_dcest < 0) { const char *e = getenv("DCEST"); g_dcest = (e && *e) ? atoi(e) : 0; }
    static int g_dcest_margin = -1;   // DCEST_MARGIN:alloc = raw_est × margin%(默认 200)。margin 时间中立
                                      // (数值/拷贝/d2h 都只碰 actual 条,仅 allocation 间距变大)→ 按"证书零触发"选:
                                      // 10-12 直方图实测 c-64/pre2 125 即零溢出,in-2004 有 2 行 ratio∈[0.75,1.0)@150
                                      // (MinHash 结构性欠估尾巴,actual=raw×1.5-2)→ 200 全覆盖;125-150 留给 L0 按阵选
    if (g_dcest_margin < 0) { const char *e = getenv("DCEST_MARGIN"); g_dcest_margin = (e && *e) ? atoi(e) : 200; }
    static int g_dcest_ovf = -1;      // DCEST_OVF=1:溢出仅报告不判废(采数探针;输出可能损坏,值门会标出)
    if (g_dcest_ovf < 0) { const char *e = getenv("DCEST_OVF"); g_dcest_ovf = (e && *e) ? atoi(e) : 0; }
    static int g_compact_t = -1; // docs/13 v2:bin0/BATCH_S compact 线程级行(函数级:compact 相位亦用)
    if (g_compact_t < 0) { const char *e = getenv("COMPACT_T"); g_compact_t = (e && *e) ? atoi(e) : 1; }
    static int g_largest = -1;   // docs/74:Ocean-D straggler 位图窗(函数级:binning 传参用)
    if (g_largest < 0) { const char *e = getenv("LARGEST"); g_largest = (e && *e) ? atoi(e) : 1; }
    static int g_esc = -1;       // docs/76 v31-④:零表 counting-sort 短行
    if (g_esc < 0) { const char *e = getenv("ESC_K"); g_esc = (e && *e) ? atoi(e) : 1; }
    static int g_esc64 = -1;     // study/50 靶1:flop∈(32,64] IPT=2 零表档(ESC64=1 开,默认关)
    if (g_esc64 < 0) { const char *e = getenv("ESC64"); g_esc64 = (e && *e && atoi(e) > 0) ? 2 : 0; }
    static int g_ovr_se = -1;   // DITER_OVR_SPANEST(study/54 §11):HEAVY→DITER 无条件改判的 span/est 经济学门;默认 8(边界扫描:G6 SiO2 破防 399ms/G8 SiO2 8.4+para 12.0+c-64 3.7 全绿/G12 para 回吐);0=旧无条件
        if (g_ovr_se < 0) { const char *e_ = getenv("DITER_OVR_SPANEST"); g_ovr_se = (e_ && *e_) ? atoi(e_) : 8; }
        static int g_esc128 = -1;   // ESC128(T3 study/53):flop(64,128] 走 esc_rank_kernel<128>;默认关待电池
        if (g_esc128 < 0) { const char *e_ = getenv("ESC128"); g_esc128 = (e_ && *e_ && atoi(e_) > 0) ? 4 : 0; }
        int esc_on_pass = g_esc | g_esc64 | g_esc128;   // 编码:1=32档;2=64档;4=128档(T3)
    static int g_diter_est_min = -1; static bool g_est_env = false;   // docs/76 §16:diter est 下限(默认 EST_DENSE_MIN/2;Ocean dense 分流启示)
    if (g_diter_est_min < 0) { const char *e = getenv("DITER_EST_MIN"); if (e && *e) { g_diter_est_min = atoi(e); g_est_env = true; } else g_diter_est_min = DITER_EST_TIER; }   // v33d 翻默认:档内行结构档 1024(超档老路径在 kernel 内钳 2048);g_est_env=显式实验值,小阵守卫不改写
    static int g_dstruct_win = -2;    // §22:-1 = 档位梯(span≤7277 五档联动);0 = 关;>0 = 手调单窗
    if (g_dstruct_win == -2) { const char *e = getenv("DSTRUCT_WIN"); g_dstruct_win = (e && *e) ? atoi(e) : -1; }   // v33d 翻默认:-1 = 档位梯分流(数值侧按 span 分两路,路由不驱逐)
    int k7_nr = 0; const int *k7_rows = nullptr;   // kernel₇ 行集(BIN_K7;count/numeric 两段共用,binning 内赋值)
    long long k7_nnz_sum = 0;
    prof("binning", [&]{
        static int g_tgt_alpha = -1;   // HASH_TGT_ALPHA(表配 target=α%×est;连续 est 下默认 150 单次吸附,study/13)
        if (g_tgt_alpha < 0) { const char *e = getenv("HASH_TGT_ALPHA"); g_tgt_alpha = (e && *e) ? atoi(e) : (g_est_cont ? 150 : 200); }
        binning_zero2_kernel<<<1, N_BINS>>>(d_cnt, d_pos, N_BINS);   // 1c-②a:双计数器合一清零(d_pos 原在 scan 后 memset,提前无依赖)
        static int g_diter = -1; static bool g_diter_env = false;   // dense-iter 重行阈值(DITER_MIN_FLOP,默认 4096;0=关)
        if (g_diter < 0) { const char *e = getenv("DITER_MIN_FLOP"); if (e && *e) { g_diter = atoi(e); g_diter_env = true; } else g_diter = DITER_FLOP_TIER; }   // v33d 默认 2048 复归:数值相 docs/46 前缀第三遍缺失已修(study/18:
                                                                                                                                 // woff46 漏 warp 内偏移 → rank 撞车 → msc 3261 垃圾行;修复后 diter 开 17 位 PASS);g_diter_env=显式实验值,小阵守卫不改写
        // v33d 小阵守卫(T10 10-05 定谳,推导见 structural_params.cuh):nnz<4M 时档内地板回 v32 值;
        // 只改写默认域(env 未显式设置的那一侧),DITER_SMALL_NNZ=0 可关
        static long long g_dsmall = -1;
        if (g_dsmall < 0) { const char *e = getenv("DITER_SMALL_NNZ"); g_dsmall = (e && *e) ? atoll(e) : DITER_SMALL_NNZ_DEFAULT; }
        const bool dsmall_on = g_dsmall > 0 && (long long)A_nnz < g_dsmall;
        const int diter_thr_run  = g_fastpath ? 999999999 : ((dsmall_on && !g_diter_env) ? EST_DENSE_MIN : g_diter);   // FASTPATH:强制关 diter;小阵回 v32 地板 4096
        const int diter_est_min_run = (dsmall_on && !g_est_env) ? EST_DENSE_MIN / 2 : g_diter_est_min;   // 小阵回 v32 地板 2048
        static int g_dim2 = -1;
        if (g_dim2 < 0) { const char *e = getenv("DIM2"); g_dim2 = (e && *e) ? atoi(e) : 0; }   // docs/61:双维路由基建(默认关 —— c-big straggler 实测挂 v4 span 门 19491/19578,路由无解需窗口内核经济学升级)
        static int g_dtier_cmp = -1;   // §22b Ocean 双维档位比较(默认关;电池 in-2004 95.3% 通过率校准)
        if (g_dtier_cmp < 0) { const char *e = getenv("DTIER_CMP"); g_dtier_cmp = (e && *e) ? atoi(e) : 0; }   // v34 翻关:隔离电池定谳 cmp 害大于利(Zd −33.7/c-43 −5/in-2004 +0.6 中性/bundle1 类驱逐税)
        static int g_spanf = -1;
        if (g_spanf < 0) { const char *e = getenv("SPANF"); g_spanf = (e && *e) ? atoi(e) : 16; }   // docs/62:重行 span 门系数(默认 16 = v4;净窗实验 4)
        static int g_occgate = -1;   // docs/67 占用轴:est≥span/16(Ga 族 3.6% 占用 O(span) 窗税实锤,
        if (g_occgate < 0) { const char *e = getenv("OCCGATE"); g_occgate = (e && *e) ? atoi(e) : 1; }   // 17 阵 0 回归(Ga −7/3Dspec2 −32/crankseg −16/nd24k −3.5/c-64 wash);v29 默认开;OCCGATE=0 关回
        static int g_ladder = -1;    // docs/67 §6 自制双梯:两侧成本律逐行比较(86% oracle 一致方向)
        if (g_ladder < 0) { const char *e = getenv("LADDER"); g_ladder = (e && *e) ? atoi(e) : 0; }      // 默认关
        static int g_batch2 = -1;    // docs/67 §8:est 65-256 小行批处理(web 图地板税,BIN_BATCH2)
        if (g_batch2 < 0) { const char *e = getenv("BATCH2"); g_batch2 = (e && *e) ? atoi(e) : 0; }      // 默认关
        static int g_sroute = -1;    // docs/72 Phase 2:结构化字节路由
        if (g_sroute < 0) { const char *e = getenv("SROUTE"); g_sroute = (e && *e) ? atoi(e) : 0; }      // 默认关
        static int g_batch_s = -1;   // docs/13 v2:est≤32 ∧ kc≤32 → 64 槽表子档(净窗电池后定默认)
        if (g_batch_s < 0) { const char *e = getenv("BATCH_S"); g_batch_s = (e && *e) ? atoi(e) : 1; }
        static int g_count_tiny = -1;// docs/13 v2:avg row ≤ 8 阵的 count 线程级行
        if (g_count_tiny < 0) { const char *e = getenv("COUNT_TINY"); g_count_tiny = (e && *e) ? atoi(e) : 1; }
        // SROUTE 用的 hash 表槽 = 半 SM 预算÷12B(结构推导,与 est 无关的物理上界;行内再按 est 选小表)
        const int ht_sroute = g_sroute > 0 ? hash_slots_pow2(SMEM_BUDGET_2BLK()) : 0;
        // 行级表配(解耦版 HASH_REXP=2)launch 侧参数:估计器段在 !pfuse 嵌套域,此处独立解析同源 env;
        // 豁免逻辑与估计器段一致(OCEAN_C 后置缩放/小阵 STREAMLINE 域不参与)
        static int g_rexp_bk = -1;
        if (g_rexp_bk < 0) { const char *e = getenv("HASH_REXP"); g_rexp_bk = (e && *e) ? atoi(e) : 0; }
        if (oc_on || A_nnz < STREAMLINE_NNZ_STRUCT) g_rexp_bk = 0;
        double tbl_glo_bk = est_expand_final, tbl_ghi_bk = 1.4; int tbl_elo_bk = 4096, tbl_ehi_bk = 4096;
        if (const char *e = getenv("HASH_REXP_HI"))  tbl_ghi_bk = atof(e);
        if (const char *e = getenv("HASH_REXP_ELO")) tbl_elo_bk = atoi(e);
        if (const char *e = getenv("HASH_REXP_EHI")) tbl_ehi_bk = atoi(e);
        static int g_occ_coeff = -1;   // OCCGATE 系数(env OCCGATE_COEFF;REXP 轮盘教训的直接调门版)
        if (g_occ_coeff < 0) { const char *e = getenv("OCCGATE_COEFF"); g_occ_coeff = (e && *e) ? atoi(e) : SPANF_COEFF; }   // 停靠:OCC_GATE_COEFF=10 待③阶段翻默认
                static int g_fdest = -1;   // 决赛 v2 真强制:FORCE_DITER_EST=8 类(env;0=关)
                if (g_fdest < 0) { const char *e = getenv("FORCE_DITER_EST"); g_fdest = (e && *e) ? atoi(e) : 0; }
        static int g_had_l1 = -1;   // HAD L1 路由修正(④;env HAD_L1=阈值:1=旧30 兼容,8=09-30 判别式;默认关)
        if (g_had_l1 < 0) { const char *e = getenv("HAD_L1"); g_had_l1 = (e && *e) ? atoi(e) : 0; if (g_had_l1 == 1) g_had_l1 = 30; }
        static int g_k7 = -1;   // kernel₇(BIN_K7):双遍位图 dup≈1 人口(10-06;默认关,51 阵屏全绿再议)
        if (g_k7 < 0) { const char *e = getenv("K7"); g_k7 = (e && *e) ? atoi(e) : 0; }   // T5b(10-23)三度回滚:eu-2005 5.55×爆(37.6→208.6ms)+rajat28 1.41×/rajat18 1.35×/bloweya 1.30×——dense否决修了hangGlider但新人群爆雷;K7需系统性edge case审计后再试   // T5(10-21)批级证伪回滚:夜9 GM 0.9617 vs 夜8 0.9183(+4.7%)——16阵筛面漏的代价=vsp_model1 219×/c-big 2.3×/lowThrust ~2×/hangGlider_4/5反向,23阵>15%;筛面推翻批级旧结论=违反结论稳定纪律;靶阵赢面真(bloweybq 2.27)待行级门修复后T5b重试
        g_k7 = g_k7 && !att;   // att(upper-tri)形态未适配:span_lo 基址不含 tri 过滤
        // D6EXP(D3② PVD 学生派发,默认关):maxbl 预计算 + 阵级 ctx 三量 + 传参
        static int g_d6 = -1;
        if (g_d6 < 0) { const char *e = getenv("D6EXP"); g_d6 = (e && *e) ? atoi(e) : 0; }
        int *d6_maxbl = nullptr; float d6_cr = 0.f, d6_cs = 0.f, d6_cm = 0.f;
        if (g_d6) {
            d6_maxbl = decltype(d6_maxbl)(dev_alloc(A_rows * sizeof(int)));
            row_maxbl_kernel<<<(A_rows + 255) / 256, 256>>>(dA_rp, dA_ci, A_rows, d6_maxbl);
            long long s_fl = thrust::reduce(thrust::device_ptr<const int>(d_flop),
                                            thrust::device_ptr<const int>(d_flop + A_rows), 0LL);
            long long s_es = thrust::reduce(thrust::device_ptr<const int>(d_est),
                                            thrust::device_ptr<const int>(d_est + A_rows), 0LL);
            d6_cr = (float)log((double)A_rows);
            d6_cs = (float)log((double)s_fl);
            d6_cm = (float)log((double)s_es / (double)A_rows);
            dbg("[%s] D6EXP on:ctx rows=%.2f sumfl=%.2f meanes=%.2f\n", tag, d6_cr, d6_cs, d6_cm);
        }
        compute_bucket_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_est, dA_rp, A_rows, EST_ULTRA_THR, d_flop, diter_thr_run, d_span_len, d_bkid, d_cnt, (g_dim2 || g_largest) ? A_cols : 0, g_spanf, g_occgate, g_ladder, g_batch2, g_sroute, ht_sroute, g_batch_s ? BATCH_S_EST_MAX : 0, g_largest, esc_on_pass, diter_est_min_run, g_dstruct_win, g_dim2, g_dtier_cmp, g_tgt_alpha,
            g_rexp_bk == 2 ? tbl_glo_bk : 0.0, g_rexp_bk == 2 ? tbl_ghi_bk : 0.0, g_rexp_bk == 2 ? tbl_elo_bk : 0, g_rexp_bk == 2 ? tbl_ehi_bk : 0, est_expand_final, g_occ_coeff, g_fdest, g_had_l1, g_k7,
            g_d6, d6_maxbl, d6_cr, d6_cs, d6_cm, g_d6 ? A_cols : 0, g_ovr_se);
        if (d_est_save) CHECK_CUDA(cudaMemcpy(d_est_save, d_est, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToDevice));   // est 快照:dense 路径后续会置零(D2TIER 工作量轴也吃)
        thrust::exclusive_scan(thrust::device_ptr<int>(d_cnt),
                               thrust::device_ptr<int>(d_cnt + N_BINS),
                               thrust::device_ptr<int>(d_offb));
        // (d_pos 清零已前移并入 binning_zero2 —— 1c-②a)
        scatter_rows_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_bkid, A_rows, d_offb, d_pos, d_sort);
        // 3 个 D2H 合并:async + 单 sync(省 2 同步点;total_est 从 est_scan 延后到此)
        CHECK_CUDA(cudaMemcpyAsync(h_cnt, d_cnt,  N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(h_off, d_offb, N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(&total_est, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
        if (pfuse)   // PIPEFUSE:count_flop 的 device 规约在此取回(消一个独立 host 同步点)
            CHECK_CUDA(cudaMemcpyAsync(&total_flop, d_tflop, sizeof(long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaStreamSynchronize(0));
        if (d6_maxbl) { dev_free(d6_maxbl); d6_maxbl = nullptr; }   // bucket 消费完即释(D3②)
        if (pfuse) {
            avg_product = A_rows ? (double)total_flop / A_rows : 0.0;
        if (g_fastpath) avg_product = 0.0;   // FASTPATH:强制 est=flop 捷径(覆写真实值)
            dbg("[%s] avg_product=%.1f (total_flop=%lld, PIPEFUSE 延迟取回)\n", tag, avg_product, total_flop);
            dev_free(d_tflop); dev_free(d_avg_gate);
            d_tflop = nullptr; d_avg_gate = nullptr;
        }
        if (h_cnt[BIN_K7] > 0) { k7_nr = h_cnt[BIN_K7]; k7_rows = d_sort + h_off[BIN_K7]; }   // kernel₇ 行集(count/numeric 两段共用)
        if (est_projected >= 0) total_est = est_projected;   // MHSAMP:d_est=min(flop,n) 的 scan 非 Σest 真值 → 投影覆写(dup 路由保持诚实)
        // DUMP_ROUTING=<path>:路由决策数据落盘(est/span/flop/bkid 四数组;docs/76 §22 收口的
        // 校准数据源 —— 档位梯双维门[span 档 vs est 档]的赢家/输家分离度验证,只跑一次)
        if (const char *dp = getenv("DUMP_ROUTING")) {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                FILE *df = fopen(dp, "wb");
                if (df) {
                    int hdr[2] = {A_rows, BIN_DITER};
                    fwrite(hdr, sizeof(int), 2, df);
                    auto dump_arr = [&](const char *src_name, int *d_src) {
                        std::vector<int> tmp(A_rows);
                        CHECK_CUDA(cudaMemcpy(tmp.data(), d_src, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
                        fwrite(tmp.data(), sizeof(int), A_rows, df);
                    };
                    dump_arr("est", d_est);
                    dump_arr("span", d_span_len);
                    dump_arr("flop", d_flop);
                    dump_arr("bkid", d_bkid);
                    {   // §26:a_len 维(A_row_ptr 差分;成本律拟合原料)
                        std::vector<int> rp(A_rows + 1), al(A_rows);
                        CHECK_CUDA(cudaMemcpy(rp.data(), dA_rp, (size_t)(A_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
                        for (int r = 0; r < A_rows; r++) al[r] = rp[r + 1] - rp[r];
                        fwrite(al.data(), sizeof(int), A_rows, df);
                    }
                    fclose(df);
                    dbg("[%s] DUMP_ROUTING → %s (%d 行)\n", tag, dp, A_rows);
                }
            }
        }
        dbg("[%s] diter bin=%d 行 / %d\n", tag, h_cnt[BIN_DITER], A_rows);   // TEMP:docs/24 路由观测
        dbg("[%s] k7 bin=%d 行 / %d\n", tag, h_cnt[BIN_K7], A_rows);   // kernel₇ 路由观测(10-06)
        { char bs[256]; int off = 0;
          for (int bi = 0; bi < N_BINS; bi++) if (h_cnt[bi]) off += snprintf(bs+off, sizeof(bs)-off, "%d:%d ", bi, h_cnt[bi]);
          dbg("[%s] bins: %s\n", tag, bs); }
        if (getenv("BINS_DBG")) {   // 每 bin 行数分布(小行族定位:docs/13 v2/road 家族)
            for (int bi = 0; bi < N_BINS; bi++)
                if (h_cnt[bi]) dbg("[%s]   bin[%2d](ht=%5d) = %d 行\n", tag, bi,
                                   (bi <= HASH_LADDER_MAX_BIN) ? (WARP_SIZE << bi) : -1, h_cnt[bi]);
        }

    });

    // ---- 全 bin Hybrid Value(spec91 刀E v2):ht≥4096 大表 bin 的 gmem 值池(keys 留 SMEM)。 ----
    // v1(块作用域全量池 ~1.1GB)实测 in-2004 accumulate -25.6%/nd24k -2.07×:RED 散射打 DRAM RMW。
    // v2 = 有界循环缓冲池(ocean 机理):per-bin N_BUF×ht×8B ≤ L2 预算,blockIdx%N_BUF 复用
    // (并发块 < N_BUF ⇒ 同缓冲两用者不共存),块内自清(免 host memset);池在发射点按 hblk 配额。
    double *d_gv = nullptr; long long *d_gv_off = nullptr;   // v2: per-bin 池发射点挂 d_gv;d_gv_off 恒 null(旧 free 点共用)
    double *d_gv_bins[HASH_LADDER_MAX_BIN + 1] = {};          // 刀E v2:每 VG bin 独立循环池(多流并发 bin 不共享),统一 free
    static int g_hyb_tier = -1;
    // 默认 0(2026-08-27 A/B 否决的是 hash_spa_hkv 残缺形态;刀E 重开为模板实例,SASS 定谳 inno/91 §1.5)
    if (g_hyb_tier < 0) { const char *e = getenv("HYB_TIER"); g_hyb_tier = (e && *e) ? atoi(e) : 0; }

    // 2b: opt-in max SMEM(read-first 已入本体,RF/CAS 两实例化同待命 >48KB 表)
    {
        size_t maxsm = (size_t)HASH_CAP * (sizeof(int) + sizeof(double));   // sh_col[int]+sh_val[double]
        if (maxsm > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM) {
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        }
        // spec91 刀E:VG 实例 opt-in(ids-only 表 ht×4B;HASH_CAP=16384 → 64KB > 48KB 默认)
        size_t maxsm_vg = (size_t)HASH_CAP * sizeof(int);
        if (maxsm_vg > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm_vg));
        // spec91 ①回归修复:仪器实例 <0,*,1> 同样需要 opt-in(ht≥4096 bin 的 48KB+静态 1296B
        // 超 默认上限 → 发射 invalid argument,probe_hist 仪器路径直接崩;① 前仪器=运行时参数
        // 共用 <0,*> 主实例(已 opt-in),模板化后成为独立实例漏了这里)
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,0,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        // spec91 刀F:LOG_HT 实例(bins 7-9 = ht 2^12/2^13/2^14)同需 opt-in
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,12>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,13>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,14>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        // E3 修复(study/57 §四十六):WM_T 模板参后 LHT2 发射的是 <...,P2,0> 独立符号,
        // 上面默认参(WM_T=-1)实例的 opt-in 不覆盖它们 → bins7-9 大表 launch invalid argument
        // (c-64 实证,两 build 同崩;10-23 后只跑 Bump bins4/5 小表未暴露——批跑禁重建的活教训)
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,7,0,0,0>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,8,0,0,0>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,0,0>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,10,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,11,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,12,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,13,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,14,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        // 刀F-主刀:素数紧档实例 opt-in(5449 档 64KB/16253 档 195KB 超 48K;2719 档 32KB 免)
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,5449>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,0,16253>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,5449>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0,1,1,0,0,16253>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
    }
    // 2b-priv: warp 私有 SPA 门控(HASH_PRIV=1,HASH_PRIV_W 默认 8);仅 SMEM 可容纳的桶走 priv
    static int g_priv = -1, g_privW = PRIV_W_MAX;
    if (g_priv < 0) {
        const char *e = getenv("HASH_PRIV");
        g_priv = (e && *e && atoi(e) > 0) ? 1 : 0;
        const char *ew = getenv("HASH_PRIV_W");
        if (ew && *ew) { int w = atoi(ew); if (w >= 2 && w <= PRIV_W_MAX) g_privW = w; }
        if (g_priv)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_priv_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PRIV_SMEM_MAX));   // H100 ≤228KB;launch 时按 bin 再判
    }
    const int PRIV_W = g_privW;
    // 2c: per-bin launch(hash 插入 + extract;prof 内部已 sync);memset+overflow D2H 纳入计时
    int overflow;
    // ===== S4 路由表集成(study/60):查表替代手拍门 =====
    // 签名:bin9_frac/bin45_frac/规模 → 3×3×3 格 → H69/H45/H55 配置
    // 数据出处:470 阵标定(356 真实 + 153 合成);RTABLE=0 逃生门回手拍模式
    static int g_rtable = -1;
    if (g_rtable < 0) { const char *e = getenv("RTABLE"); g_rtable = (e && *e) ? atoi(e) : 1; }
    int rt_h69 = 1, rt_h45 = 1, rt_h55 = 1;   // 默认=全开(70% 人口的测量最优)
    if (g_rtable) {
        long long rt_total = 0;
        for (int b = 0; b <= HASH_LADDER_MAX_BIN; b++) rt_total += h_cnt[b];
        double rt_b9f = rt_total > 0 ? (double)h_cnt[9] / rt_total : 0;
        double rt_b45f = rt_total > 0 ? (double)(h_cnt[4] + h_cnt[5]) / rt_total : 0;
        int rt_b9 = (rt_b9f < 0.01) ? 0 : ((rt_b9f < 0.3) ? 1 : 2);
        int rt_b45 = (rt_b45f < 0.01) ? 0 : ((rt_b45f < 0.3) ? 1 : 2);
        int rt_sz = (rt_total < 10000) ? 0 : ((rt_total < 1000000) ? 1 : 2);
        // 内联查表(与 routing_table.h 同构,避免头文件依赖)
        if (rt_b9==0 && rt_b45==2 && rt_sz==1) { rt_h69=1; rt_h45=1; }  // none_high_medium n=144
        else if (rt_b9==0 && rt_b45==1 && rt_sz==1) { rt_h69=1; rt_h45=1; }  // none_low_medium n=48
        else if (rt_b9==0 && rt_b45==2 && rt_sz==0) { rt_h69=1; rt_h45=1; }  // none_high_small n=28
        else if (rt_b9==0 && rt_b45==0 && rt_sz==2) { rt_h69=1; rt_h45=1; }  // none_none_large n=19
        else if (rt_b9==2 && rt_b45==0 && rt_sz==1) { rt_h69=1; rt_h45=1; }  // high_none_medium n=7
        else if (rt_b9==1 && rt_b45==1 && rt_sz==1) { rt_h69=1; rt_h45=1; }  // low_low_medium n=12
        else if (rt_b9==2 && rt_b45==1 && rt_sz==1) { rt_h69=0; rt_h45=0; }  // high_low_medium n=14
        // 其余格:默认全开(无人口/分裂格)
        dbg("[%s] RTABLE: b9=%d(%.2f) b45=%d(%.2f) sz=%d → h69=%d h45=%d h55=%d\n",
            tag, rt_b9, rt_b9f, rt_b45, rt_b45f, rt_sz, rt_h69, rt_h45, rt_h55);
    }
    // dense 累积器门(2026-08-26,exdata_1 类):小 n + 输出 ≥15% 稠密 → 直接寻址免探测免 CAS
    #define DENSE_WIN_MAX_N DENSE_WIN_MAX_N_STRUCT
    const bool dense_mode = (A_cols <= DENSE_MAX_N) && A_rows > 0 &&
        ((double)total_est / ((double)A_rows * A_cols) >= DENSE_MIN_FRAC)
        && h_cnt[BIN_K7] == 0    // k7 接管时矩阵级 dense 语义失效(accumulate 兜底+compact 矩阵拷贝双侧;10-06 零化案根修)
        && !((g_rtable ? rt_h69 : (h69_route_on()?1:0)) && h_cnt[9] > 0);   // S4:查表或手拍(bin9 阵压制 dense)
    const bool dense_win_mode = !dense_mode && (A_cols <= DENSE_WIN_MAX_N) && A_rows > 0 &&
        ((double)total_est / ((double)A_rows * A_cols) >= DENSE_MIN_FRAC)
        && h_cnt[BIN_K7] == 0    // 同上
        && !((g_rtable ? rt_h69 : (h69_route_on()?1:0)) && h_cnt[9] > 0);   // S4:同上
    // ---- 方案5(docs/27 §4.1):dense 行精确计数 + 直写终态。DIRECT5=1 开,默认关。----
    // dense 行集合 = 矩阵级 dense/dense_win(全体行)或 Phase A bin(v4 门含 dup<8 = 两遍直写
    // 交通量赢面);hash 行不动(结构同 Ocean type2 est 工作流)。count pass 在 accumulate 前
    // 出精确 row_nnz → cnnz_scan 后数值 pass 直写 dC;dense 行不再占 gapped tmp 槽(缩容 Σest_dense)
    // 且永不 ovf(计数精确,免 retry)。
    static int g_direct5 = -1;
    if (g_direct5 < 0) { const char *e = getenv("DIRECT5"); g_direct5 = (e && *e) ? atoi(e) : 1; }   // refresh9 全量验证 -0.99% → 默认开;DIRECT5=0 关
    static int g_d5h = -1;
    if (g_d5h < 0) { const char *e = getenv("D5H"); g_d5h = (e && *e && atoi(e) > 0) ? 1 : 0; }
    int dense_nr = 0; const int *dense_rows = nullptr;   // dense 行集合(null = 全体行)
    long long dense_est_sum = 0, dense_nnz_sum = 0, tmp_slots_dev = 0;
    // 1c-③(10-14,审计弹药):est 坍缩尾声公共化 —— 原三处原样重复(DCEST/方案5/D5H);
    // rows=null 时全量清零(矩阵级 dense 语义)。D5H 调用方借 d_sort 段连续性单发置零(原 ≤11 发)。
    auto est_collapse_tail = [&](const int *rows, int nr) {
        if (rows && nr > 0)
            zero_est_kernel<<<(nr + HASH_BLOCK_STRUCT - 1) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_est, rows, nr);
        else
            CHECK_CUDA(cudaMemset(d_est, 0, (size_t)A_rows * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
        if (A_rows <= MAX_BLOCK_THREADS) {
            int b = 1; while (b < A_rows) b <<= 1;
            scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
        } else {
            thrust::inclusive_scan(
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                thrust::device_ptr<long long>(d_off + 1));
        }
        CHECK_CUDA(cudaMemcpy(&tmp_slots_dev, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
    };
    // §22d span 分流(count 前算好,count/numeric 两阶段共用):DSTRUCT≥2 时 dense 行按自身
    // span 排序分两段 —— 档内段(≤DENSE_TIER_SPAN_MAX)走档位梯;超档段走老窗口内核。
    // (count 侧同分:c-64 定谳 —— 50K 全幅行在位图结构 count 里多窗重扫乘积,1.8→8.8ms 主凶)
    static int g_dstr2 = -1;   // docs/72 Phase 1b:DSTRUCT=2 时 numeric 走结构内核(位图字级前缀)
    if (g_dstr2 < 0) { const char *e = getenv("DSTRUCT"); g_dstr2 = (e && *e) ? atoi(e) : 2; }   // v33d 翻默认:结构数值 + 双侧分流
    int *d_rows_sorted = nullptr;                          // span 升序行表(档内段在前,超档尾段在后)
    int h_tier_end[DENSE_TIER_COUNT] = {0};                // 每档【终点】(前缀差分)
    int h_big_off = 0;                                     // 超档段起点 = 老内核人群
    int h_tier_hmax = 0;                                   // 档内段最大 span(结构 count 窗)
    int h_tier_w[DENSE_TIER_COUNT] = {0};                  // §26:每档段内实际最大 span(per-tier 窗宽自适应)
    int h_t3_end[DENSE_TIER_COUNT*2] = {0};                // T3:子分档终点(每档 2 子档,span 升序)
    int *d_d3_rows = nullptr; int d_d3_n = 0; int h_d3_n = 0;   // D3(T7c):merge 核行单/计数
    int h_t3_w[DENSE_TIER_COUNT*2] = {0};                  // T3:子档窗=子档内实际最大 span
    auto dstr_split = [&]() {                              // §22d 分流体(dense_rows 就绪后调用)
        { static int n_call = 0; n_call++; if (n_call <= 12) dbg("[%s] dstr_split 调用#%d(dense_nr=%d)\n", tag, n_call, dense_nr); }
        // 单次 D2H(融合 span+行号+est 三元组)+ 异步 H2D:同步点 3→1(nsys 定谳:prof 内同步
        // memcpy = 流水线停顿,in-2004 每迭代 ~1.5ms 空泡)
        int *d_sp = decltype(d_sp)(dev_alloc(4 * (size_t)dense_nr * sizeof(int)));
        span_row_est_gather_kernel<<<(dense_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(dense_rows, dense_nr, d_span_len, d_est, dA_rp, d_sp);
        std::vector<int> h_both(4 * dense_nr);
        CHECK_CUDA(cudaMemcpy(h_both.data(), d_sp, 4 * (size_t)dense_nr * sizeof(int), cudaMemcpyDeviceToHost));
        dev_free(d_sp);
        const int *h_sp = h_both.data();                    // [0, n) = span
        const int *h_dense = h_both.data() + dense_nr;      // [n, 2n) = 行号
        const int *h_est = h_both.data() + 2 * dense_nr;    // [2n, 3n) = est(est 置零前快照)
        const int *h_alen = h_both.data() + 3 * dense_nr;   // [3n, 4n) = a_len(§26 成本律维)
        std::vector<int> idx(dense_nr);
        for (int k = 0; k < dense_nr; k++) idx[k] = k;
        // ⚠ 按【档位索引】稳定排序,不按 span 排:稳定 = 同档内保持行号序(nsys 定谳:span 升序
        //   打乱行号 → CSR 访存失局部性 → count/direct 同内核同人口 +15-20%;行号序 = bin 散射序
        //   的既有赢家形态)。超档行 = 档 6,同样保序。
        // §25 密度门(DDENS,默认开):2×est < span 的档内行也并入老内核段 —— 结构内核成本
        //   ∝ span(窗扫描/位图前缀),老内核 ∝ est(乘积);est/span 稀疏行(c-43 族 0.34,
        //   in-2004 0.59)结构窗白扫 → c-43 count 2.15→~0.3 / numeric 1.26→0.80 实测定谳。
        static int g_ddens = -1;   // §25 密度门(默认关:in-2004 定谳 —— 档内稀疏行进老内核反 +28ms,
        if (g_ddens < 0) { const char *e = getenv("DDENS"); g_ddens = (e && *e) ? atoi(e) : 0; }  // 老内核成本含 a_len 维,密度单维不构成判据;代码留档)
        static int g_alen_old = -1;   // §26 a_len 门(v34 翻默认开,×5 中位电池背书:c-43 −4.0/in-2004
        if (g_alen_old < 0) { const char *e = getenv("ALEN_OLD"); g_alen_old = (e && *e) ? atoi(e) : 1; }  // −2.2/Zd/bbmat/std1/3dtube 噪声带;wb-edu/c-64/nd24k no-op 零 a_len≤32 档内行)
        auto tier_key = [&](int k_row) {
            int s = h_sp[k_row];
            int t = 0; while (t < DENSE_TIER_COUNT && s > dense_tier_max(t)) t++;
            if (g_alen_old && t < DENSE_TIER_COUNT && h_alen[k_row] <= WARP_SIZE) t = DENSE_TIER_COUNT;   // §26 短链行 → 老内核段
            if (g_ddens && t < DENSE_TIER_COUNT && 2 * (long long)h_est[k_row] < (long long)s) t = DENSE_TIER_COUNT;   // (§25 密度门,默认关)
            return t;
        };
        // D3(T7c,env D3EXP 默认关):低 dup 短 a_len merge 核人口抽取(pre2 型载体;
        // 谓词=a_len∈[1,32]∧est∈[256,4096]∧span≤16384——pre2 est 662-1282 全入,
        // c-64 型(est 16K)/in-2004 中带(est 4.4K)排除)
        static int g_d3exp = -1;
        if (g_d3exp < 0) { const char *e = getenv("D3EXP"); g_d3exp = (e && *e && atoi(e) > 0) ? 1 : 0; }
        if (g_d3exp) {
            std::vector<int> d3list;
            for (int q2 = 0; q2 < dense_nr; q2++) {
                int r2 = h_dense[q2];
                int al2 = h_alen[r2];
                if (al2 >= 1 && al2 <= 32 && h_est[r2] >= 256 && h_est[r2] <= 4096 && h_sp[r2] <= 16384)
                    d3list.push_back(r2);
            }
            if (!d3list.empty()) {
                // 重排 h_dense:非 D3 在前(保序)
                std::vector<int> keep;
                for (int q2 = 0; q2 < dense_nr; q2++) {
                    int r2 = h_dense[q2];
                    int al2 = h_alen[r2];
                    bool d3 = (al2 >= 1 && al2 <= 32 && h_est[r2] >= 256 && h_est[r2] <= 4096 && h_sp[r2] <= 16384);
                    if (!d3) keep.push_back(r2);
                }
                int w = (int)keep.size();
                for (int q2 = 0; q2 < w; q2++) h_both[dense_nr + q2] = keep[q2];   // h_dense=h_both+n 段,写母数组
                dense_nr = w;
                d_d3_rows = decltype(d_d3_rows)(dev_alloc(d3list.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpyAsync(d_d3_rows, d3list.data(), d3list.size() * sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)0));
                d_d3_n = (int)d3list.size();
                h_d3_n = d_d3_n;
                dbg("[%s] D3EXP:merge 核人口 %d 行(梯子/direct 余 %d)\n", tag, d_d3_n, w);
            }
        }
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return tier_key(a) < tier_key(b); });
        int off = 0;
        for (int t = 0; t < DENSE_TIER_COUNT; t++) {
            while (off < dense_nr && tier_key(idx[off]) <= t) off++;   // 边界按【键】走(密度失败行键=6 但 span 小,span 序已破)
            h_tier_end[t] = off;
        }
        h_big_off = off;
        // §28 矩阵级 DSTRUCT 判据:档内行中位 est/span < 1/2 → 全人口走老内核(窗口机制在
        // 中位半空 = 不可摊销;Zd 0.40/c-43 0.34 → old,in-2004 0.59 → struct;矩阵级 ≠ §25
        // 逐行驱逐,行全留 diter 只选内核族,Zd 四除后的末刀)
        if (off > 0) {
            std::vector<double> dens;
            for (int k = 0; k < off; k++) if (h_sp[idx[k]] > 0) dens.push_back((double)h_est[idx[k]] / h_sp[idx[k]]);
            std::sort(dens.begin(), dens.end());
            double med = dens[dens.size() / 2];
            static int g_ds28 = -1;   // 判决实验仪器(09-22):DS_MED_OFF=1 关 §28 矩阵级门(默认关,零生产影响)
            if (g_ds28 < 0) { const char *e = getenv("DS_MED_OFF"); g_ds28 = (e && *e) ? 1 : 0; }
            if (med < 0.5 && !g_ds28) { h_big_off = 0; h_tier_end[0] = h_tier_end[1] = h_tier_end[2] = h_tier_end[3] = h_tier_end[4] = h_tier_end[5] = 0; }
            dbg("[%s] §28 矩阵级判据:档内中位密度 %.2f → %s\n", tag, med, med < 0.5 ? "老内核" : "结构内核");
        }
        // T3 子分档(env T3EXP;默认关):档内行 span 升序 + 对半 → 子档窗=子档实际 max span
        // (低子档窗 ~50KB → 2r 块(两行)SMEM ~100KB → 2块×2行=4行/SM,=彼方水平)
        static int g_t3exp0 = -1;
        if (g_t3exp0 < 0) { const char *e = getenv("T3EXP"); g_t3exp0 = (e && *e && atoi(e) > 0) ? 1 : 0; }
        if (g_t3exp0) {
            int b = 0;
            for (int t = 0; t < DENSE_TIER_COUNT; t++) {
                int e = h_tier_end[t];
                if (e - b >= 128) {
                    std::stable_sort(idx.begin() + b, idx.begin() + e,
                                     [&](int x, int y){ return h_sp[x] < h_sp[y]; });
                    int mid = b + (e - b) / 2;
                    int w0 = 0, w1 = 0;
                    for (int k = b; k < mid; k++) w0 = w0 > h_sp[idx[k]] ? w0 : h_sp[idx[k]];
                    for (int k = mid; k < e; k++) w1 = w1 > h_sp[idx[k]] ? w1 : h_sp[idx[k]];
                    h_t3_end[t*2] = mid;   h_t3_w[t*2] = w0;
                    h_t3_end[t*2+1] = e;   h_t3_w[t*2+1] = w1;
                } else {
                    // 空档或小段:终点必须=当前基址 b(留 0 会把发射侧 off 重置回 0 →
                    // 后续子档从行表头重复处理错误行=T3 首轮 18.65M 错位真凶)
                    h_t3_end[t*2] = e; h_t3_end[t*2+1] = e;
                    if (e > b) {
                        int w0 = 0;
                        for (int k = b; k < e; k++) w0 = w0 > h_sp[idx[k]] ? w0 : h_sp[idx[k]];
                        h_t3_w[t*2] = w0;
                    }
                }
                b = e;
            }
            dbg("[%s] T3 子分档 armed\n", tag);
        }
        h_tier_hmax = 0;                       // 档内段最大 span(档内非 span 序,取段内 max)
        for (int k = 0; k < off; k++) h_tier_hmax = h_tier_hmax > h_sp[idx[k]] ? h_tier_hmax : h_sp[idx[k]];
        {   // §26 per-tier 窗宽自适应:窗 = 该档段内【实际】最大 span(非档上限)—— 段内每行
            //   span ≤ 窗由构造保证;tier-5 固定 131KB 窗把刚过 7277 的行压到 1 block/SM,
            //   按实际 hmax 缩窗(c-43 hmax 11125 → 100KB → 2/SM)。
            int b = 0;
            for (int t = 0; t < DENSE_TIER_COUNT; t++) {
                int e = h_tier_end[t], w = 0;
                for (int k = b; k < e; k++) w = w > h_sp[idx[k]] ? w : h_sp[idx[k]];
                h_tier_w[t] = w;               // 0 = 空档
                b = e;
            }
        }
        d_rows_sorted = decltype(d_rows_sorted)(dev_alloc((size_t)dense_nr * sizeof(int)));
        std::vector<int> h_rows(dense_nr);
        for (int k = 0; k < dense_nr; k++) h_rows[k] = h_dense[idx[k]];
        CHECK_CUDA(cudaMemcpyAsync(d_rows_sorted, h_rows.data(), (size_t)dense_nr * sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)0));
        dbg("[%s] §22d 分流:档内 %d 行(每档 %d/%d/%d/%d/%d/%d,hmax=%d)+ 超档 %d 行\n",
            tag, h_big_off, h_tier_end[0], h_tier_end[1] - h_tier_end[0], h_tier_end[2] - h_tier_end[1],
            h_tier_end[3] - h_tier_end[2], h_tier_end[4] - h_tier_end[3],
            h_tier_end[5] - h_tier_end[4], h_tier_hmax, dense_nr - h_big_off);
    };
    if (g_direct5) {
        // 矩阵级只取 dense_win:n≤DENSE_MAX_N 的 dense_mode 用按 n 尺寸的 v1 内核(occupancy 高),
        // direct 是固定 14980 窗口(195KB,1CTA/SM)——exdata_1 实测 13.5→33.4ms 占用减半;且
        // dense_mode 的 compact 仅 ~1.4% 无税可省。dense_win 两路同为固定窗口 → 纯赢(A/B 实证)。
        if (dense_win_mode) dense_nr = A_rows;
        else if (!dense_mode && h_cnt[BIN_DITER] > 0) { dense_nr = h_cnt[BIN_DITER]; dense_rows = d_sort + h_off[BIN_DITER]; }
        if (dense_nr > 0) {
            // DCFUSE(docs/56):dense_count 相位瘦身 —— 砍 D2H 同步/门 pre-pass/est 坍缩重扫
            // + count kernel 动态窗宽与向量化归约。"省一遍产品扫描"经论证不成立(低 dup 下
            // gapped+compact 的 24B/entry 搬运 ≥ 4B/product 的 count 扫描;D5H csort 教训同向),
            // 但相位内 ~0.5-0.7ms 纯开销可省 → 翻 brainpc2(1.020)/c-64(1.039)级近赢阵。
    static int g_dcfuse = -1;
            if (g_dcfuse < 0) { const char *e = getenv("DCFUSE"); g_dcfuse = (e && *e) ? atoi(e) : 1; }   // 交替 A/B 全胜(brainpc2 -5.6/c-58 -13.5/mult_dcop -9.9/bloweya -4.8,无回归)→ 默认开;DCFUSE=0 关
            const bool dcf = g_dcfuse > 0;
            // DCEST(study/39 Phase 1b):消灭 dense_count 的 O(flop) 扫描——
            // 用 est 还原后的原始估计 × margin 填 row_nnz(一个 O(dense_nr) 微核,近零成本)。
            // dense_count 从 82ms → ≈0ms。V3(vs V2 est×1.2):
            //   ①去 est 的 expand 泄漏(1.8 默认):over-alloc raw×2.16 → raw×margin;
            //   ②flop 封顶行(est==flop)是精确上界,不除 expand(dup≈1 人口除则必炸);
            //   ③溢出不再靠 2.16× 蛮力覆盖 → 证书核 + 判废回退(见 dense_direct 后的修正链)。
            if (g_dcest && g_cnf) { fprintf(stderr, "[hash] DCEST 与 CNT_FUSE 互斥(cnf 有自己的 tmp/晚 cnnz 数据流)→ DCEST 关\n"); g_dcest = 0; }
            if (g_dcest && dense_nr > 0 && dense_rows) {
                prof("dcest", [&]{
                    // ① 填 row_nnz = alloc(在 est 置零前)
                    dcest_fill_kernel<<<(dense_nr + HASH_BLOCK_STRUCT - 1) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                        d_row_nnz, d_est, d_flop, dense_rows, dense_nr,
                        (int)(est_expand_final * 100.0 + 0.5), g_dcest_margin);
                    CHECK_CUDA(cudaGetLastError());
                    // ② dstr_split(数值侧需要)
                    if (g_dstr2 >= 2 && dense_rows) dstr_split();
                    // ③ est 置零 + d_off 重扫(dense 行不占 gapped tmp)—— 1c-③ 公共尾
                    est_collapse_tail(dense_rows, dense_nr);
                });
                dense_nnz_sum = 0; dense_est_sum = 0;   // DCEST 下不精确;underflow 由 cnnz 校正
                dbg("[%s] DCEST V3: dense_count 相位消灭(%d 行,est/%.2f×%.2f 替代精确计数)\n",
                    tag, dense_nr, est_expand_final, g_dcest_margin / 100.0);
                goto dcest_done;
            }
            // 矩阵级路径不再走下方 accumulate 分支(那三处 memset 挪到这里,计时内);mixed 路径 per-bin 头会再清,无害
            prof("dense_count", [&]{
                CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
                CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
                CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
                // 规模门 pre-pass(refresh9 实证:dup 门不成立 —— 赢家 dup 1.54/2.11/6.14 与输家
                // <1.5/1.74/4.64 完全重叠,docs/29 §8 开放问题;机械可解释的只有微型 dense 集:
                // cnr-2000(73 行/0.3% est)/ohne2(69 行/0.13%)纯固定开销亏损)。
                // 门 = dense 行数 ≥ 1000 且 Σest_dense ≥ 5% total_est,否则方案5 整体跳过回 legacy。
                // DCFUSE:矩阵级(dense_nr==A_rows 且 ≥1000)两条件平凡成立 → 免 dense_sum+D2H。
                if (!(dcf && dense_nr == A_rows && dense_nr >= DENSE5_MIN_ROWS)) {
                    unsigned long long *d_s2 = decltype(d_s2)(dev_alloc(2 * sizeof(long long)));
                    CHECK_CUDA(cudaMemset(d_s2, 0, 2 * sizeof(long long)));
                    dense_sum_kernel<<<(dense_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                        dense_rows, dense_nr, d_est, nullptr, d_flop, d_s2, d_s2 + 1);
                    unsigned long long h2[2];
                    CHECK_CUDA(cudaMemcpy(h2, d_s2, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
                    dev_free(d_s2);
                    long long est_d = (long long)h2[0], flop_d = (long long)h2[1];
                    if (dense_nr < DENSE5_MIN_ROWS || est_d * DENSE5_MIN_FRAC_DEN < total_est) {
                        dbg("[%s] 方案5 规模门:跳过(dense_nr=%d Σest=%lld/%lld, flop=%lld)\n",
                            tag, dense_nr, est_d, total_est, flop_d);
                        dense_nr = 0;
                        return;
                    }
                }
                if (g_dstr2 >= 2 && dense_rows) dstr_split();   // §22d 分流(count/numeric 共用;dense_rows 此刻就绪)
                size_t csm = (size_t)DENSE_CNT_W;   // byte flag(docs/45 bitmap 回退)
                int cnt_w = DENSE_CNT_W;
                if (dcf) {   // 动态窗宽:n 小则窗缩(n=27k 阵从 64KB→32KB SMEM = 2→3 CTA/SM)
                    unsigned long long want = ((unsigned long long)A_cols + 3) & ~3ull;
                    cnt_w = (int)(want < (unsigned long long)DENSE_CNT_W ? want : DENSE_CNT_W);
                    csm = (size_t)cnt_w;
                }
                static int g_ccur = -1;   // spec87:COUNT_CURSOR=1 门(默认关;游标区动态扩段)
                /* g_cnf: function scope */
                if (g_ccur < 0) { const char *e9 = getenv("COUNT_CURSOR"); g_ccur = (e9 && *e9) ? 1 : 0; }
                if (g_ccur) csm += 1024 * sizeof(int);
                if (csm + SMEM_STATIC_HEADROOM > (size_t)SMEM_DEFAULT_MAX)   // 动态+静态 >48K 才 opt-in(CUDA 总量规则)
                    CHECK_CUDA(cudaFuncSetAttribute(hash_dense_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)csm));
                // docs/70 位图 symbolic(BCNT=1):span 一次位图单遍,免窗免 lower_bound 重搜。
                // 需 dense 行最大 span 定统一 SMEM(小 span 阵不牺牲 occupancy);超帽回退旧窗式。
                static int g_bcnt = -1;
                if (g_bcnt < 0) { const char *e = getenv("BCNT"); g_bcnt = (e && *e) ? atoi(e) : 0; }
                // docs/72 Phase 1a 结构化(DSTRUCT=1):dense_struct_count 替代所有 count 变体,
                // 窗宽/档位全部由 structural_params.cuh 推导(零拟合常数),span ≤ 档 → 单窗免重搜。
                static int g_dstruct = -1;
                if (g_dstruct < 0) { const char *e = getenv("DSTRUCT"); g_dstruct = (e && *e) ? atoi(e) : 2; }   // v33d 翻默认:结构 count + 分流
                static int g_dcnt = -2;   // §25 count 侧 A/B 旋钮:0=恒老窗式 / 2=分流(默认,=DSTRUCT 语义);
                if (g_dcnt == -2) { const char *e = getenv("DCNT"); g_dcnt = (e && *e) ? atoi(e) : 0; }   //  §25 电池定谳翻默认 0=恒老窗式(c-43 −35.8/3dtube −17.6/c-58 −15.5/Zd −11.4;唯一代价 in-2004 +1.5)
                bool bcnt_launched = false;
                const bool cnt_struct_on = (g_dcnt >= 0) ? (g_dcnt > 0) : (g_dstruct > 0);   // §25:DCNT 覆盖(count 侧内核选择)
                const bool cnt_split_on  = (g_dcnt >= 0) ? (g_dcnt == 2) : (g_dstruct >= 2);
                if (cnt_struct_on && dense_rows) {
                    // §22d:DSTRUCT≥2 且分流已完成 → 档内段走结构位图 count(窗 = 档内 hmax,
                    // 单窗免重扫);超档段走老窗式 count(50K 全幅行位图 count 多窗重扫乘积,
                    // c-64 实测 1.8→8.8ms 主凶 —— 2026-09-04 count 相分解定谳)。
                    // DSTRUCT=1(无分流)保持原行为:全体 dense 行单发结构 count。
                    const bool split_cnt = (cnt_split_on && g_dstr2 >= 2 && d_rows_sorted);
                    int hmax_span = 0;
                    if (!split_cnt) {
                        int *d_sp = decltype(d_sp)(dev_alloc(dense_nr * sizeof(int)));
                        span_gather_kernel<<<(dense_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(dense_rows, dense_nr, d_span_len, d_sp);
                        {   auto mx = thrust::max_element(thrust::device_ptr<int>(d_sp), thrust::device_ptr<int>(d_sp + dense_nr));
                          CHECK_CUDA(cudaMemcpy(&hmax_span, mx.get(), sizeof(int), cudaMemcpyDeviceToHost)); }
                        dev_free(d_sp);
                    } else hmax_span = h_tier_hmax;
                    // 窗宽 = count 档(仅位图 1/8 B/列)由 SMEM 半并发预算推导;SMEM = 窗宽/8 字节
                    const int win_cols = bitmap_count_cols_max(SMEM_BUDGET_2BLK());
                    const int use_cols = hmax_span < win_cols ? hmax_span : win_cols;
                    size_t ssm = (size_t)((use_cols + 31) >> 5) * sizeof(unsigned) + 8;   // 位图字 + 8B 对齐余量
                    CHECK_CUDA(cudaFuncSetAttribute(dense_struct_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm));
                    const int cnt_nr = split_cnt ? h_big_off : dense_nr;
                    const int *cnt_rows = split_cnt ? d_rows_sorted : dense_rows;
                    if (cnt_nr > 0) {
                        dbg("[%s] DSTRUCT: max_span=%d win=%d smem=%zuKB(档内 %d 行;超档 %d 行走老窗式)\n",
                            tag, hmax_span, use_cols, ssm / 1024, cnt_nr, dense_nr - cnt_nr);
                        prof("dense_count", [&]{
                            dense_struct_count_kernel<<<cnt_nr, HASH_BLOCK_STRUCT, ssm>>>(
                                dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows,
                                cnt_rows, d_span_lo, d_span_len, use_cols, d_row_nnz);
                            CHECK_CUDA(cudaGetLastError());
                        });
                    }

                    if (!g_cnf && split_cnt && h_big_off < dense_nr) {   // 超档段:老窗式 count(#4' CNT_FUSE=1 跳过=免计数)
                        int old_nr = dense_nr - h_big_off;
                        hash_dense_count_kernel<<<old_nr, DENSE_BLK, csm>>>(
                            dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows, A_cols,
                            d_row_nnz, d_rows_sorted + h_big_off, d_span_lo, d_flop, d_maxbl, cnt_w, g_ccur);
                        CHECK_CUDA(cudaGetLastError());
                    }
                    bcnt_launched = true;   // DSTRUCT 命中时不再走 BCNT/旧窗式
                }
                else if (g_bcnt && dense_rows) {
                    int *d_sp = decltype(d_sp)(dev_alloc(dense_nr * sizeof(int)));
                    span_gather_kernel<<<(dense_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(dense_rows, dense_nr, d_span_len, d_sp);
                    int hmax_span = 0;
                    {   auto mx = thrust::max_element(thrust::device_ptr<int>(d_sp), thrust::device_ptr<int>(d_sp + dense_nr));
                      CHECK_CUDA(cudaMemcpy(&hmax_span, mx.get(), sizeof(int), cudaMemcpyDeviceToHost)); }
                    size_t bsm = (size_t)(((unsigned)hmax_span + 31) >> 5) * sizeof(unsigned) + 8;
                    dbg("[%s] BCNT: max_span=%d smem=%zuKB\n", tag, hmax_span, bsm / 1024);
                    if (bsm <= (size_t)BCNT_SMEM_MAX) {
                        CHECK_CUDA(cudaFuncSetAttribute(hash_dense_bcnt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bsm));
                        prof("dense_count", [&]{
                            hash_dense_bcnt_kernel<<<dense_nr, DENSE_BLK, bsm>>>(
                                dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows, A_cols,
                                d_row_nnz, dense_rows, d_span_lo, d_span_len);
                            CHECK_CUDA(cudaGetLastError());
                        });
                        bcnt_launched = true;
                    }
                    dev_free(d_sp);
                }
                if (!bcnt_launched) {   // #4':CNT_FUSE 时收缩为仅梯段(老段免计数)
                    int cnt_nr = (g_cnf && d_rows_sorted) ? h_big_off : dense_nr;
                    const int *cnt_rows = (g_cnf && d_rows_sorted) ? d_rows_sorted : dense_rows;
                    if (cnt_nr > 0)
                    hash_dense_count_kernel<<<cnt_nr, DENSE_BLK, csm>>>(
                        dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows, A_cols,
                        d_row_nnz, cnt_rows, cnt_rows ? d_span_lo : nullptr, d_flop, d_maxbl, cnt_w, g_ccur);
                }
                CHECK_CUDA(cudaGetLastError());
                unsigned long long *d_sums = decltype(d_sums)(dev_alloc(2 * sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_sums, 0, 2 * sizeof(long long)));
                dense_sum_kernel<<<(dense_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                    dense_rows, dense_nr, d_est, d_row_nnz, nullptr, d_sums, d_sums + 1);
                if (dcf) {   // 一次 16B D2H 取两和(两次小 D2H = 两次流水线停顿)
                    unsigned long long h_sums[2];
                    CHECK_CUDA(cudaMemcpy(h_sums, d_sums, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
                    dense_est_sum = (long long)h_sums[0]; dense_nnz_sum = (long long)h_sums[1];
                } else {
                    CHECK_CUDA(cudaMemcpy(&dense_est_sum, d_sums, sizeof(long long), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(&dense_nnz_sum, d_sums + 1, sizeof(long long), cudaMemcpyDeviceToHost));
                }
                dev_free(d_sums);
                // dense 行 est 置 0 → d_off 重扫:gapped 偏移坍缩进 hash 侧空间(tmp 才能按缩容量分配;
                // d_off 是全行 est 前缀,不重扫则 heavy/hash 行偏移仍指旧空间 → 越界)。
                // 置零安全:binning 已完成,后续无消费者按 dense 行的 est 寻址(它们不走 hash/compact)。
                // ⚠ D5H(方案5 扩展到 hash 行,docs/31):bins 1-10 先 MODE=1 count-only 出精确 row_nnz,
                //    数值 pass 改 MODE=2 直写 dC(小行有序免排/大行原地 csort),compact 对这些行消失。
                //    门 = dup = total_flop/total_est < 8(Ocean type1 的 compaction 经济学同款)。
                // DCFUSE 免坍缩(实验性,默认关):d_off 保留全行 est 前缀,tmp 按 host 已知
                // total_est 定容,免 zero_est+重扫+D2H。分解实测(ab_dcfuse_decomp)pool 扩容
                // 成本 > 省下的重扫 —— brainpc2 10.18 vs 10.10 / c-64 21.86 vs 21.38,全阵净负
                // → 默认 DCF_BUDGET_GB=0(永远坍缩);kernel 改进(窗宽+归约)才是收益来源。
                static long long dcf_budget = -1;
                if (dcf_budget < 0) {
                    const char *e = getenv("DCF_BUDGET_GB");
                    dcf_budget = (e && *e) ? atoll(e) * 1024LL * 1024LL * 1024LL : 0LL;
                }
                const bool no_collapse = dcf && total_est > 0 && (unsigned long long)total_est * COLLAPSE_DIRECT_BYTES_PER_EST <= (unsigned long long)dcf_budget;
                if (no_collapse) {
                    tmp_slots_dev = total_est;
                } else {
                est_collapse_tail(dense_rows, dense_nr);   // 1c-③ 公共尾(原方案5 内联副本)
                }
            });
            dbg("[%s] 方案5:count pass %d dense 行(Σest=%lld Σnnz=%lld)→ cnnz_scan 后直写终态\n",
                tag, dense_nr, dense_est_sum, dense_nnz_sum);
        }
    }
    dcest_done:;   // DCEST goto 目标:跳过整个 dense_count 相位
    // ---- kernel₇:BIN_K7 行 count(位图标记+popcount → d_row_nnz;方案5 同构直写族)----
    if (k7_nr > 0) {
        prof("k7_count", [&]{
            int *d_k7err = decltype(d_k7err)(dev_alloc(sizeof(int)));
            CHECK_CUDA(cudaMemset(d_k7err, 0, sizeof(int)));
            k7_count_kernel<<<k7_nr, K7_BLOCK, (size_t)K7_W * 4>>>(
                dA_rp, dA_ci, dB_rp, dB_ci, A_rows, k7_rows, d_span_lo, d_row_nnz, d_k7err);
            int herr = 0; CHECK_CUDA(cudaMemcpy(&herr, d_k7err, sizeof(int), cudaMemcpyDeviceToHost));
            if (herr) fprintf(stderr, "[hash] k7_count 防御带 %d 产品(span_lo/路由不变式破坏,值门必查)\n", herr);
            dev_free(d_k7err);
            {   // Σnnz(underflow 检查扣减项;gather+ToLL 归约,文件同款 idiom)
                int *d_k7n = decltype(d_k7n)(dev_alloc((size_t)k7_nr * sizeof(int)));
                thrust::gather(thrust::device_pointer_cast(k7_rows), thrust::device_pointer_cast(k7_rows + k7_nr),
                               thrust::device_pointer_cast(d_row_nnz), thrust::device_pointer_cast(d_k7n));
                k7_nnz_sum = thrust::reduce(
                    thrust::make_transform_iterator(thrust::device_ptr<int>(d_k7n), ToLL()),
                    thrust::make_transform_iterator(thrust::device_ptr<int>(d_k7n + k7_nr), ToLL()),
                    (long long)0, thrust::plus<long long>());
                dev_free(d_k7n);
            }
            zero_est_kernel<<<(k7_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_est, k7_rows, k7_nr);   // 与 dense 同窗:hash 侧 tmp 不为 k7 行定容
        });
        dbg("[%s] k7_count: %d 行(BIN_K7)位图精确计数 Σnnz=%lld\n", tag, k7_nr, k7_nnz_sum);
    }
    // ---- D5H:方案5 扩展到 hash 行(docs/31;D5H=1 开,默认关)----
    // bins 1-10 行 MODE=1 count-only → row_nnz 精确 → MODE=2 数值 pass 直写 dC。
    // 小行(ht≤CSORT_HT)count-sort 有序直写免善后;大行无序直写 + 原地 csort。
    // 计入 tmp 缩容(与 dense 行共用置零+重扫)。
    int d5h_rows = 0; long long d5h_nnz_sum = 0;
    const bool g_d5h_on = g_d5h && !att && !(dense_nr > 0 && (dense_mode || dense_win_mode))
        && (total_est > 0 ? (double)total_flop / (double)total_est < (double)DUP_LIGHT : false);   // dup<DUP_LIGHT(Ocean type1 经济学)
    if (g_d5h_on) {
        prof("hash_count", [&]{
            for (int bi = 1; bi <= HASH_LADDER_MAX_BIN; bi++) {
                int n = h_cnt[bi];
                if (n == 0) continue;
                int *rows_ptr = d_sort + h_off[bi];
                int ht = WARP_SIZE << bi;
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                {   // 一次性无条件 opt-in(49152 恰在 48KB 边界也会失败;legacy 对 <0> 即此做法)
                    static bool done1 = false;
                    if (!done1) {
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<1,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<1,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));   // 逃生门
                        done1 = true;
                    }
                }
                // docs/70 §6②(D5B=1 ∧ D5H=1):合格行(span≤8×est ∧ 位图装得下)的 count 走
                // 位图(免 CAS 探测),其余行仍 hash_spa<1>(带跳过谓词)。逐 bin 双 launch 各司其职。
                static int g_d5b = -1;
                if (g_d5b < 0) { const char *e = getenv("D5B"); g_d5b = (e && *e) ? atoi(e) : 0; }
                const int *skip_e = nullptr, *skip_s = nullptr;
                if (g_d5b) {
                    skip_e = d_est; skip_s = d_span_len;
                    int *d_es = decltype(d_es)(dev_alloc(n * sizeof(int)));
                    elig_span_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(rows_ptr, n, d_est, d_span_len, BCNT_SPAN_MAX_BITS, d_es);
                    int hmax = 0;
                    auto mx = thrust::max_element(thrust::device_ptr<int>(d_es), thrust::device_ptr<int>(d_es + n));
                    CHECK_CUDA(cudaMemcpy(&hmax, mx.get(), sizeof(int), cudaMemcpyDeviceToHost));
                    dev_free(d_es);
                    if (hmax > 0) {
                        size_t bsm = (size_t)((hmax + 31) / 32) * sizeof(unsigned) + 8;
                        CHECK_CUDA(cudaFuncSetAttribute(hash_row_bcnt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bsm));
                        hash_row_bcnt_kernel<<<n, HASH_BLOCK_STRUCT, bsm>>>(   // 小行位图:256T 块(1024T 的块开销反主导,首测 +7~18% 教训)
                            dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows, A_cols,
                            rows_ptr, n, d_est, d_span_lo, d_span_len, d_row_nnz, 1, (int)(bsm / 4));
                        CHECK_CUDA(cudaGetLastError());
                    }
                }
                if (probe_rf_on())
                    hash_spa_kernel<1,1><<<n, HASH_BLOCK, smem_flat>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, 0, nullptr, nullptr, nullptr, d_rt, skip_e, skip_s);
                else
                    hash_spa_kernel<1,0><<<n, HASH_BLOCK, smem_flat>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, 0, nullptr, nullptr, nullptr, d_rt, skip_e, skip_s);
                d5h_rows += n;
                CHECK_CUDA(cudaGetLastError());
            }
            {   // docs/69 §7 修复:Σnnz(count 后)供下溢检查 —— 原 if(false) 封存致 d5h_nnz_sum
                // 恒 0 → D5H 行输出全记到 hash 侧 → 下溢检查误回退(F2 实锤,nnz 红旗抓的)。
                // 封存疑为 Fix#8(grid=0)之前的 invalid-config 残留;n==0 continue 守卫已在。
                // ⚠ 口径近似:bins 1-10 的 MODE=1 溢出行(交 retry)nnz 不在此和内 → 检查偏保守,
                // 极端欠估仍可回退(安全方向,非误放行)。
                unsigned long long *d_s3 = decltype(d_s3)(dev_alloc(sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_s3, 0, sizeof(long long)));
                for (int bi = 1; bi <= HASH_LADDER_MAX_BIN; bi++) {
                    if (h_cnt[bi] == 0) continue;   // Fix#8(审查):空 bin 的 <<<0,256>>> = invalid-config
                                                      // = docs/31/35 "invalid argument" 之谜根因(grid=0)
                    dense_sum_kernel<<<(h_cnt[bi] + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                        d_sort + h_off[bi], h_cnt[bi], nullptr, d_row_nnz, nullptr, nullptr, d_s3);
                }
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaMemcpy(&d5h_nnz_sum, d_s3, sizeof(long long), cudaMemcpyDeviceToHost));
                dev_free(d_s3);
            }
            if (d5h_rows > 0) {   // 置零 bins 1-10 行的 est + d_off 重扫(与 dense 共用机制)
                // 1c-③:bins 1-10 在 d_sort 段连续 → 单发置零(原逐 bin ≤11 发)
                est_collapse_tail(d_sort + h_off[1], h_off[HASH_LADDER_MAX_BIN + 1] - h_off[1]);
            }
        });
        dbg("[%s] D5H:bins1-10 count-only %d 行 → MODE=2 直写\n", tag, d5h_rows);
    }
    // tmp 缩容:dense 行直写不占 gapped 槽(heavy scratch 同口径;鲸鱼阵 Phase A 行恰是 est 大头)
    const long long tmp_slots = (dense_nr > 0 || d5h_rows > 0) ? tmp_slots_dev : total_est;
    d_tmp_key = decltype(d_tmp_key)(dev_alloc((size_t)tmp_slots * sizeof(unsigned long long)));
    d_tmp_val = decltype(d_tmp_val)(dev_alloc((size_t)tmp_slots * sizeof(double)));
    {   // P0-2 修复(10-14,study/43 审计):tmp_val 的消费者含 atomicAdd 累加路径
        // (dense_largest pass3:3776、batched:1779)——依赖零起点;此前靠 cudaMalloc 新页
        // 恰为零的运气,池模式/内存复用下=垃圾起加的错值。清零改无条件(鲸鱼 ~0.3ms)。
        // INITDBG 历史注释保留:study/18 曾用此分叉定位 denormal 垃圾来源。
        CHECK_CUDA(cudaMemset(d_tmp_val, 0, (size_t)tmp_slots * sizeof(double)));
    }
    if (dense_nr > 0 && (dense_mode || dense_win_mode)) {
        // 方案5 矩阵级:count 已出精确 row_nnz,数值 pass 在 cnnz_scan 后直写(此分支无 accumulate)
    } else
    if (dense_mode && k7_nr == 0) {   // k7 接管时禁入:矩阵级兜底会重算 k7 行(10-06 实测 187ms 主因);残留行走 per-bin 正路
        dbg("[%s] dense 累积器路径(n=%d, est 占比 %.1f%%)\n", tag, A_cols,
            100.0 * total_est / ((double)A_rows * A_cols));
        prof("accumulate", [&]{
            CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
            // T6-v4:host 判 multi:窗 3072×13≈40KB + smap(maxrow×4);
            // multi 条件:省 smem(窗+map < n×13)且 map 装得下(≤16KB)
            int maxrow_dense = 0;
            {   const int *hrp_ = (const int*)A_buffer;   // A_buffer = [row_ptr | col | val]
                for (int r_ = 0; r_ < A_rows; r_++) {
                    int l_ = hrp_[r_+1] - hrp_[r_];
                    if (l_ > maxrow_dense) maxrow_dense = l_;
                }
            }
            int smap_cap = 128;   // T6-v5:固定 128 槽(512B);ak>128 的行 kernel 内走二分
            // T6-v7:窗数门控——密度不判(41.6%赢 vs 40.5%输,非密度问题);
            // 窗数∝n 决定开销(exdata n=6001→6窗赢 8×;case9 n=14454→15窗输 72%)。
            // n≤8192(8 窗)为甜点位:smem 已够小(13×8192=107KB→2块/SM 原路径),
            // 切到窄窗(13×1024=13KB→16块/SM)的占用跳跃最大而窗开销尚低。
            int t6_multi = (A_cols * 13 > 48 * 1024) && (A_cols <= 8192);
            size_t dsm = t6_multi
                ? (size_t)smap_cap * 4 + (size_t)1024 * 13
                : ((size_t)A_cols * 9 + 3) / 4 * 4 + (size_t)A_cols * sizeof(int);
            if (dsm > SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM)
                CHECK_CUDA(cudaFuncSetAttribute(hash_dense_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm));
            hash_dense_kernel<<<A_rows, DENSE_BLK, dsm>>>(
                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                d_ovf_rows, d_ovf_cnt, d_row_ovf, t6_multi, smap_cap);
            CHECK_CUDA(cudaGetLastError());
        });
    } else if (dense_win_mode && k7_nr == 0) {   // 同上:k7 接管时禁入
        dbg("[%s] dense 窗口路径(n=%d, est 占比 %.1f%%)\n", tag, A_cols,
            100.0 * total_est / ((double)A_rows * A_cols));
        prof("accumulate", [&]{
            CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
            size_t wsm = ((size_t)DENSE_MAX_N * 9 + 3) / 4 * 4 + (size_t)DENSE_MAX_N * sizeof(int);
            CHECK_CUDA(cudaFuncSetAttribute(hash_dense_window_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
            hash_dense_window_kernel<<<A_rows, DENSE_BLK, wsm>>>(
                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                d_ovf_rows, d_ovf_cnt, d_row_ovf, nullptr, nullptr, d_flop, d_maxbl);
            CHECK_CUDA(cudaGetLastError());
        });
    } else
    prof("accumulate", [&]{
        CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
        static int g_bsearch_s = -1;
    if (g_bsearch_s < 0) { const char *e = getenv("BSEARCH"); g_bsearch_s = (e && *e && atoi(e) > 0) ? 1 : 0; }
    // LLB=1:hash_spa_kernel 按行动态 G(localLoadBalance 移植,docs/27 §4.2;默认关待 A/B)
    static int g_llb = -1;
    if (g_llb < 0) { const char *e = getenv("LLB"); g_llb = (e && *e && atoi(e) > 0) ? 1 : 0; }
    const bool g_bsearch = g_bsearch_s;
    // 方案4修正: HYBRID=1 时 value 走全局 L2 原子(学 Ocean HYBRID_HASHMAP)
    static int g_hybrid_env = -2;
    if (g_hybrid_env == -2) { const char *e = getenv("HYBRID"); g_hybrid_env = (e && *e) ? atoi(e) : -1; }
    // 方案4: dup 因子 > 5 时自动开 hybrid(L2 原子吞吐 > SMEM,高 dup 阵受益 5-29%)
    // 小阵不开(全局池分配 + L2 延退 > 收益);HYBRID=0/1 强制覆盖
    const bool g_hybrid = (g_hybrid_env >= 0) ? g_hybrid_env
        : (total_est > 0 && (double)total_flop / total_est > (double)DUP_HYBRID_THR && A_rows > HYB_MIN_ROWS);
    // 多 stream per-bin(HASH_NSTREAMS env 调优,0=单流对照;对标 Ocean 20-stream)
        static int g_nstream = -2;
        if (g_nstream == -2) {
            const char *e = getenv("HASH_NSTREAMS");
            g_nstream = (e && *e) ? atoi(e) : NSTREAM_DEFAULT;
            if (g_nstream < 0 || g_nstream > NSTREAM_MAX) g_nstream = NSTREAM_MAX / 2;
        }
        const bool use_ms = (g_nstream > 0);
        static cudaStream_t bin_s[NSTREAM_MAX];
        static bool bin_s_init = false;
        if (use_ms && !bin_s_init) {
            for (int i = 0; i < g_nstream; i++) CHECK_CUDA(cudaStreamCreate(&bin_s[i]));
            bin_s_init = true;
        }
        for (int bi = 0; bi < N_BINS; bi++) {
            cudaStream_t cur_s = use_ms ? bin_s[bi % g_nstream] : (cudaStream_t)0;
            int n = h_cnt[bi];
            if (n == 0) continue;
            if (bi == BIN_K7) continue;   // kernel₇:有自己的 k7_count/k7_direct 相位,禁入 hash 梯(漏跳过=百万槽表灾难,10-06 实测 187ms)
            if (g_d5h_on && bi >= 1 && bi <= HASH_LADDER_MAX_BIN) continue;   // D5H:bins1-10 延后 MODE=2(scan 后直写)
            int *rows_ptr = d_sort + h_off[bi];
            // ⚠ ca7bb79(subwarp8 实验)曾把 subwarp8 分支插在 heavy 前面挡死了 BIN_HEAVY 的 dispatch
            //   (est>16384 的重行全走 32 槽小表 → 必然溢出 → 全体进 flop 定表 retry = DNF 内存爆炸推手)。
            //   subwarp8 已否决(ca7bb79 自述"路由回退"但 dispatch 漏改),此分支删除,bin 布局见 N_BINS 注释。
            if (bi == 0 && g_bsearch) {
                // 方案1: 预排序二分寻址(HSMU 思路): symbolic 排序列 + numeric 二分定位
                bsearch_symbolic_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
                bsearch_numeric_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_row_nnz, d_tmp_key, d_tmp_val);
            } else if (bi == 0) {
                // 小行批量: HYBRID=1 时 value 走全局 L2(Ocean 杀手锏;池句柄 d_hybrid_val 尾部统一释放,Fix#6)
                if (g_hybrid && !d_hybrid_val)
                    d_hybrid_val = decltype(d_hybrid_val)(dev_alloc((size_t)n * BATCH_HT * sizeof(double)));
                // batched 战役刀(study/59):BATCHMR=1→multirow<128,2>(16-lane 组);=2→multirow<64,4>(8-lane 组,
                // est>48 行走 overflow→retry 逃生);默认 0=原核逐位。B 行短人口(atmosmodj/delaunay 型)的 lane 利用率修正。
                // §批次刀复盘(10-29 凌晨):默认 2 全局翻=第二次人口联锁失误(confirm17 中段
                // 128 阵恶化 vs 7 改善,B 行长族 bcsstk/bcspwr j 并行度被砍 4×)——回滚默认 0,
                // 加标量人口门 avg_j=total_flop/A_nnz(B 行均长):≤12 才走 multirow(atmosmodj ~7/
                // delaunay 型短 B 行人口),env BATCHMR=1/2 强制 / 0 强制关(三态诊断)。
                static int g_batchmr = -1;
                if (g_batchmr < 0) { const char *e = getenv("BATCHMR"); g_batchmr = (e && *e) ? atoi(e) : -1; }
                double avg_j = A_nnz > 0 ? (double)total_flop / A_nnz : 1e9;
                int mr_on = (g_batchmr >= 0) ? g_batchmr : 0;   // S1 死码剔除验证(10-29):multirow 整体 #if 0,发射分支移除;BATCHMR env 保留读但无效果
                (void)mr_on; (void)avg_j;
                hash_spa_batched_kernel<BATCH_HT, BATCH_WPB><<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * WARP_SIZE, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf, d_hybrid_val);
            } else if (bi == BIN_ESC64 || bi == BIN_ESC128) {   // T3(study/53):esc_rank smem 广播秩排序(64/128 档)
                static int g_sm_count = -1;
                if (g_sm_count < 0) { cudaDeviceProp prop_; CHECK_CUDA(cudaGetDeviceProperties(&prop_, 0)); g_sm_count = prop_.multiProcessorCount; }
                if (bi == BIN_ESC64) {   // 64 档(旧 hash_esc64=shfl 双键已证伪 study/50)
                    int nlaunch = (n + (HASH_BLOCK_STRUCT / 64) - 1) / (HASH_BLOCK_STRUCT / 64);
                    if (nlaunch > g_sm_count * 8) nlaunch = g_sm_count * 8;   // 256T 块 8/SM=满占用
                    esc_rank_kernel<64><<<nlaunch, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val,
                        d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                } else {
                    int nlaunch = (n + (HASH_BLOCK_STRUCT / 128) - 1) / (HASH_BLOCK_STRUCT / 128);
                    if (nlaunch > g_sm_count * 8) nlaunch = g_sm_count * 8;   // 256T 块 8/SM=满占用
                    esc_rank_kernel<128><<<nlaunch, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val,
                        d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                }
            } else if (bi >= BIN_ESC8 && bi <= BIN_ESC32) {   // v31-④ v2:零表 counting-sort,行宽分档
                int subw = (bi == BIN_ESC8) ? 8 : (bi == BIN_ESC16) ? 16 : 32;
                int rpb = HASH_BLOCK_STRUCT / subw;                       // 行/块 = 块宽÷子组宽
                int nlaunch = (n + rpb - 1) / rpb;
                if (subw == 8)
                    hash_esc_kernel<8><<<nlaunch, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val,
                        d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else if (subw == 16)
                    hash_esc_kernel<16><<<nlaunch, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val,
                        d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else
                    hash_esc_kernel<32><<<nlaunch, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val,
                        d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_BATCH_S) {   // docs/13 v2:est≤32 小行 64 槽表(清/扫税减半;road 族 100% bin0)
                hash_spa_batched_kernel<BATCH_S_HT, BATCH_WPB><<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * WARP_SIZE, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_BATCH2) {   // docs/67 §8:est≤256 小行批处理(<512,4>,消块地板税)
                hash_spa_batched_kernel<BATCH2_HT, BATCH2_WPB><<<(n + BATCH2_WPB - 1) / BATCH2_WPB, BATCH2_WPB * WARP_SIZE, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_LARGEST) {   // docs/74:1 CTA/行,SMEM = 0.25B/列 × win
                const int win_l = (int)((size_t)LARGEST_SMEM_BUDGET * 4);   // 列/窗 = 预算÷0.25B
                size_t lsm = (size_t)((win_l + 31) >> 5) * sizeof(unsigned) * 2 + SMEM_ALIGN_SLACK;
                static bool lattr = false;
                if (!lattr) {
                    CHECK_CUDA(cudaFuncSetAttribute(dense_largest_numeric_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lsm));
                    lattr = true;
                }
                dense_largest_numeric_kernel<<<n, HASH_BLOCK_STRUCT, lsm, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                    rows_ptr, d_span_lo, d_span_len, win_l, d_est,
                    d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_ULTRA) {
                // ultra(est≤EST_ULTRA_THR):线性,免 hash
                hash_ultra_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_DITER) {
                // dense-iter(docs/22 Phase A):重行列窗口累积,直接寻址免探测;
                // 窗口升序 → 行内天然有序 → compact 走 copy。输出仍写 tmp(est-gapped)+ ovf 行照旧 retry。
                // 方案5(DIRECT5):count pass 已出精确 row_nnz,数值 pass 在 cnnz_scan 后直写 → 此 bin 跳过
                if (dense_nr == 0 && h_d3_n == 0) {   // D3 抽空≠兜底(D3EXP 语义)
                    size_t wsm = ((size_t)DENSE_MAX_N * 9 + 3) / 4 * 4 + (size_t)DENSE_MAX_N * sizeof(int);
                    CHECK_CUDA(cudaFuncSetAttribute(hash_dense_window_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
                    hash_dense_window_kernel<<<n, DENSE_BLK, wsm, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                        d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf, rows_ptr, d_span_lo, d_flop, d_maxbl);
                }
            }
            else if (bi >= BIN_SSPAN0 && bi <= BIN_SSPAN0 + 3) {
                // 小跨度 dense 4 子桶(docs/37):span 定 SMEM/块型,小块高行密度
                int sb = bi - BIN_SSPAN0;
                if (sb == 0)
                    hash_sspan2_kernel<SMALL_SPAN_MAX, SMALL_SPAN_MAX / 4><<<n, SMALL_SPAN_MAX / 4, (((size_t)SMALL_SPAN_MAX * 9) + 3) / 4 * 4 + (size_t)SMALL_SPAN_MAX * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else if (sb == 1)
                    hash_sspan2_kernel<SMALL_SPAN_MAX * 2, SMALL_SPAN_MAX / 2><<<n, SMALL_SPAN_MAX / 2, (((size_t)(SMALL_SPAN_MAX * 2) * 9) + 3) / 4 * 4 + (size_t)(SMALL_SPAN_MAX * 2) * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else if (sb == 2)
                    hash_sspan2_kernel<SMALL_SPAN_MAX * 4, HASH_BLOCK_STRUCT><<<n, HASH_BLOCK_STRUCT, (((size_t)(SMALL_SPAN_MAX * 4) * 9) + 3) / 4 * 4 + (size_t)(SMALL_SPAN_MAX * 4) * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else
                    hash_sspan2_kernel<SMALL_SPAN_MAX * 8, HASH_BLOCK_STRUCT><<<n, HASH_BLOCK_STRUCT, (((size_t)(SMALL_SPAN_MAX * 8) * 9) + 3) / 4 * 4 + (size_t)(SMALL_SPAN_MAX * 8) * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // 即时检查(invalid-argument 之谜防复发,docs/35 §4)
            }
            else if (bi == BIN_HEAVY) {
                // heavy(est>HASH_CAP):全局内存 hash 表 + cub 分段排序(不回退 merge)
                d_heavy_ht     = decltype(d_heavy_ht)(dev_alloc(n * sizeof(long long)));   // ll:scan 同型(混型 scan 产出坏偏移的坑)
                d_heavy_tab_off= decltype(d_heavy_tab_off)(dev_alloc((n + 1) * sizeof(long long)));
                heavy_prep_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(rows_ptr, n, d_est, d_heavy_ht, d_overflow);
                // exclusive scan:[k] = Σ_{i<k} ht[i] —— 恰为行 k 的表偏移(inclusive 才需要 +1 偏移,别搞混)
                thrust::exclusive_scan(thrust::device_ptr<long long>(d_heavy_ht),
                                       thrust::device_ptr<long long>(d_heavy_ht + n),
                                       thrust::device_ptr<long long>(d_heavy_tab_off));
                long long total_slots, last_ht;
                CHECK_CUDA(cudaMemcpy(&total_slots, d_heavy_tab_off + (n - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&last_ht, d_heavy_ht + (n - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                total_slots += last_ht;
                long long ght_cap = HEAVY_ARENA_MAX;                  // SAFETY:表区 arena 上限(字节)
                if (const char *e = getenv("GHT_MAX_BYTES")) ght_cap = atoll(e);
                dbg("[hash] heavy rows=%d slots=%lld arena=%.2fGB\n", n, total_slots,
                    total_slots * HASH_BYTES_PER_SLOT / (1 << 30));
                if (total_slots > 0 && total_slots * (long long)(sizeof(int) + sizeof(double)) > ght_cap) {
                    fprintf(stderr, "[hash] heavy arena %.2fGB 超 cap %.2fGB → 按 overflow 回退\n",
                            total_slots * 12.0 / (1 << 30), ght_cap / 1073741824.0);
                    CHECK_CUDA(cudaMemset(d_overflow, 1, sizeof(int)));
                } else if (total_slots > 0) {
                    d_tab_col = decltype(d_tab_col)(dev_alloc((size_t)total_slots * sizeof(int)));
                    d_tab_val = decltype(d_tab_val)(dev_alloc((size_t)total_slots * sizeof(double)));
                    d_seg_beg = decltype(d_seg_beg)(dev_alloc(n * sizeof(int)));
                    d_seg_end = decltype(d_seg_end)(dev_alloc(n * sizeof(int)));
                    d_scr_key = decltype(d_scr_key)(dev_alloc((size_t)tmp_slots * sizeof(unsigned long long)));
                    d_scr_val = decltype(d_scr_val)(dev_alloc((size_t)tmp_slots * sizeof(double)));
                    hash_global_kernel<<<n, HASH_BLOCK>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_heavy_ht, d_heavy_tab_off, d_tab_col, d_tab_val,
                        d_off, d_est, (long long)total_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf);
                    heavy_seg_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                        rows_ptr, n, d_off, d_row_nnz, d_seg_beg, d_seg_end);
                    size_t cub_bytes = 0;
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        nullptr, cub_bytes, d_tmp_key, d_scr_key, d_tmp_val, d_scr_val,
                        total_est, n, d_seg_beg, d_seg_end));
                    d_cub_tmp = dev_alloc(cub_bytes);
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        d_cub_tmp, cub_bytes, d_tmp_key, d_scr_key, d_tmp_val, d_scr_val,
                        total_est, n, d_seg_beg, d_seg_end));
                }
            } else {
                static int g_ht_shift = -1;   // HASH_HT_SHIFT(study/54 LF 弹性探针):bins≤6 表松弛位;默认 0=旧行为零变更
                if (g_ht_shift < 0) { const char *e = getenv("HASH_HT_SHIFT"); g_ht_shift = (e && *e) ? atoi(e) : 0; }
                int ht = (WARP_SIZE << bi) << ((bi <= 5) ? g_ht_shift : 0);
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                // spec 81 两级排序 hist 预算:3×1024×4B=12KB(仅 512≤ht≤1024 档——kernel 侧分支同界;
                // bin5 表 12KB+12KB=24KB,256 线程 8 块 ×24KB=192KB < 228KB 不压 occupancy。
                // ⚠ ht>CSORT_HT 档绝不能加:bin9 表 196KB+12KB=208KB > opt-in(HASH_CAP×12)→ 发射失败)
                static int g_extract2 = -1;   // B'(09-24 用户授权默认全局开):两级计数排序;HASH_EXTRACT=0 逃生门
                // (bin 级 dup 条件版试作判负:stat 核在 F2/msc 触发未定位 fault,且条件仅护 msc +0.06ms 绝对值——不值)
                if (g_extract2 < 0) { const char *e = getenv("HASH_EXTRACT"); g_extract2 = (e && atoi(e) == 0) ? 0 : 1; }
                if (g_extract2 && ht >= CS2_MIN_HT && ht <= CSORT_HT) smem_flat += 3 * 1024 * sizeof(int);
                size_t smem_priv = (size_t)PRIV_W * ht * (sizeof(int) + sizeof(double));
                // spec91 刀E:VG 发射所需 env statics + 块梯(原在下游 else 内,上提共用)
                static int g_force_logG = -1;   // GFORCE(study/15 诊断):强制 G;0=正常梯
                if (g_force_logG < 0) { const char *e = getenv("GFORCE"); g_force_logG = (e && *e) ? __builtin_ctz(atoi(e)) : 0; }
                static int g_uo_x = -1;
                if (g_uo_x < 0) { const char *e = getenv("HASH_UO_XTRACT"); g_uo_x = (e && *e && atoi(e) > 0) ? 1 : 0; }
                static int g_rt_nox = -1;
                if (g_rt_nox < 0) { const char *e = getenv("HASH_RT_NOEXTRACT"); g_rt_nox = (e && *e && atoi(e) > 0) ? 1 : 0; }
                static int g_wm = -1;
                if (g_wm < 0) { const char *e = getenv("HASH_WM"); g_wm = (e && *e && atoi(e) > 0) ? 1 : 0; }
                // docs/44 per-bin block 梯(Ocean BLOCK_SIZES 同哲学)+ v37 修正梯(study/06 §7 F2 对拍):
                //   默认 = 旧三档 64/256/512 + bi≥8→1024T(修 occupancy 缺陷:ht≥8192 表 ≥96KB ×512T
                //   只驻 2 CTA = 1024 线程;1024T 恢复满格 2048。F2/电池已证)。
                //   HASH_BLK_LADDER=legacy 纯旧三档;=6:384,7:768 逐档覆写(32 倍数,免重编)。
                // (spec91 刀E:梯计算上提——VG 发射同吃 hblk;旧 hkv 固定 256T 是 8月否决的
                //  病因之一:同 bin SMEM 路径 512/1024T,VG 256T = 行车道少 2-4× → 行延迟 4-5×)
                static int g_blk_lad[HASH_LADDER_MAX_BIN + 1];
                static int g_blk_init = -1;
                if (g_blk_init < 0) {
                    for (int b = 0; b <= HASH_LADDER_MAX_BIN; b++)
                        g_blk_lad[b] = (b <= ilog2c(SMALL_SPAN_MAX / WARP_SIZE)) ? HASH_BLOCK_STRUCT / 4
                                         : (b <= ilog2c(CSORT_HT / WARP_SIZE)) ? HASH_BLOCK_STRUCT : DENSE_BLK;
                    const char *e = getenv("HASH_BLK_LADDER");
                    if (e && !strcmp(e, "legacy")) {
                        // 纯旧梯,不加默认覆写
                    } else if (e && *e) {
                        // 用户逐档覆写全权(不再叠加默认)
                        for (char *s = strdup(e), *tok = strtok(s, ","); tok; tok = strtok(nullptr, ","))
                        {
                            int b, t;
                            if (sscanf(tok, "%d:%d", &b, &t) == 2 && b >= 0 && b <= HASH_LADDER_MAX_BIN &&
                                t > 0 && t <= 1024 && (t & 31) == 0)
                                g_blk_lad[b] = t;
                        }
                    } else {
                        for (int b = 8; b <= HASH_LADDER_MAX_BIN; b++) g_blk_lad[b] = 1024;   // 默认(×5 电池已证)
                    }
                    g_blk_init = 1;
                }
                int hblk = g_blk_lad[bi <= HASH_LADDER_MAX_BIN ? bi : HASH_LADDER_MAX_BIN];
                static int g_vg45 = -1;   // 刀E2(study/57 §廿六):bin4→VG 空格补测——VAL_GMEM(L2 原生 RED.E.ADD.F64,免 smem CAST.SPIN 模拟税)×128T 窄块 = ocean <7,673> 耦合配置;默认关
                if (g_vg45 < 0) { const char *e = getenv("HASH_VG45"); g_vg45 = (e && *e) ? atoi(e) : 0; }
                    // ===== P1 cutover(study/58 附录A.1):本 TU(TU-90/sm90)只留 bi≤3 LHT2 主路;
                    // bi≥4 及一切托付形态(VG/priv/escape/仪器/runtime)→ TU-61(compute_61)。
                    // 实例集不相交:本 TU 仅实例化 LHT2 LHT∈{7,8};运行时形态只在 TU-61(免 TB_TAG)。
                    static int g_lht = -1;
                    if (g_lht < 0) { const char *e = getenv("HASH_LHT"); g_lht = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
                    static int g_lht23 = -1;   // 刀A(study/57 tick2):bins2-3 常量折叠(sm90 LHT2 本 TU)。
                    // P1 电池定谳(§五十六):默认必须 1——bins2-3 走 TU-61 运行时形态=病理(F2 acc 16×/c-64 9×);
                    // =1 时 F2/TSOPF/c-64/bcsstk17 全部追平或超越双参考(刀A 九阵银行数据复现)。HASH_LHT23=0 逃生门(诊断用)。
                    if (g_lht23 < 0) { const char *e = getenv("HASH_LHT23"); g_lht23 = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
                    static int g_probe2 = -1;
                    if (g_probe2 < 0) { const char *e = getenv("HASH_PROBE2"); g_probe2 = (e && *e) ? atoi(e) : 0; }
                    if (false && bi <= 3 && g_lht && g_lht23 && probe_rf_on() && !d_phist && !g_wm) {   // §五十六:TU-90 untagged LHT7/8 = web-Google smem 越界雷(正确性级编译条件),封;bins2-3 全托 TU-61 tagged
#define SPA_LAUNCH_LHT2(LHT_, P2_) hash_spa_kernel<0,1,0,0,(LHT_),0,(P2_),0><<<n, hblk, smem_flat, cur_s>>>( \
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, \
        rows_ptr, n, ht, d_off, \
        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, \
        d_flop, d_maxbl, g_llb, nullptr, nullptr, nullptr, d_rt, nullptr, nullptr, \
        g_force_logG, g_uo_x, dB16, g_wm, nullptr, g_rt_nox, g_extract2, 0, nullptr)
                        if (bi == 2) { if (g_probe2) SPA_LAUNCH_LHT2(7, 1); else SPA_LAUNCH_LHT2(7, 0); }
                        else { if (g_probe2) SPA_LAUNCH_LHT2(8, 1); else SPA_LAUNCH_LHT2(8, 0); }   /* bi==3 */
#undef SPA_LAUNCH_LHT2
                        CHECK_CUDA(cudaGetLastError());
                    } else {
                        HashAccumCtx c61 = { dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, rows_ptr, n,
        d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
        d_flop, d_maxbl, d_rt, d_est, d_phist, dB16, d_gv_bins, cur_s, tag,
        upper_tri, A_rows, A_cols, g_llb, g_hyb_tier, g_priv, PRIV_W, att };
                        hash_accum_c61(c61, bi);
                    }
                    CHECK_CUDA(cudaGetLastError());   // 发射即时检查(>48KB 表 opt-in 类问题即抓)
            }
        }
        if (use_ms)
            for (int i = 0; i < g_nstream; i++) CHECK_CUDA(cudaStreamSynchronize(bin_s[i]));
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
        if (ph_f) {   // 链画像 dump(accumulate 后全流已同步):桶=额外探测步数,32=溢出/封顶
            unsigned h33[33];
            CHECK_CUDA(cudaMemcpy(h33, d_phist, 33 * sizeof(unsigned), cudaMemcpyDeviceToHost));
            unsigned long long tot = 0, wt = 0;
            for (int b = 0; b <= 32; b++) tot += h33[b];
            for (int b = 0; b <= 32; b++) wt += (unsigned long long)h33[b] * b;
            for (int b = 0; b <= 32; b++) fprintf(ph_f, "%d %u\n", b, h33[b]);
            fprintf(ph_f, "TOTAL %llu  AVGPROBE %.4f\n", tot, tot ? (double)wt / tot : 0.0);
            fclose(ph_f); ph_f = nullptr;
            dbg("[%s] probe_hist → %s (tot=%llu avg=%.3f)\n", tag, ph_path, tot, tot ? (double)wt / tot : 0.0);
        }
    });

    // ---- 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行(estimate<distinct)用
    //      flop(精确上界)定表重跑 hash_global,输出写独立重试区,compact 末段按 d_row_ovf 路由。
    //      全部重试成功 → 清 overflow 继续;存在不可重试行(flop 超表上限)→ 保持 overflow 整阵回退。----
    auto run_retry = [&] {
        int h_ovf = 0;
        CHECK_CUDA(cudaMemcpy(&h_ovf, d_ovf_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[hash] dbg: overflow=%d ovf_cnt=%d\n", overflow, h_ovf);   // TEMP
        // ADAPT_EXPAND 状态机 v2(docs/60 修订:时序判据替代"减半保留"):①基线 1.15 下
        // ovf>50 → 记已计相位和,下轮试 1.4;②1.4 轮相位和 > 1.15 轮 ×1.05 → 永久回退。
        // 修订原因:v1 的减半规则误升大表矩阵(c-big +45%/Flan_1565 +45%/inline_1 +46%,
        // overflow 确实减半但表更大更慢);rajat25(9亿nnz 赢家)与 cage15(9.3亿 受害者)同规模
        // ⇒ 规模门无效,只有实测时序能分开。prof.total = 截至本相位(含 count..retry)累计,
        // 两轮同口径可比。
        {
            static double g_try14_t = 0.0;
            double t_now = prof.total;
            if (g_ovf_hint_expand == 0 && h_ovf > OVFL_HINT_MIN_ROWS) {
                g_ovf_hint_expand = 1;              // 下轮试 1.4
            } else if (g_ovf_hint_expand == 1) {
                g_try14_t = t_now;                   // 本轮 = 1.4,记时
                g_ovf_hint_expand = 3;              // 下轮回测 1.15
            } else if (g_ovf_hint_expand == 3) {
                g_ovf_hint_expand = (g_try14_t <= t_now * EXPAND_KEEP_MARGIN) ? 4 : 2;   // 相邻轮对比,1.4 不慢 2% 才留
            }
        }
        if (h_ovf > 0) {
            dbg("[hash] row-retry: %d 行 → flop 定表重跑\n", h_ovf);
            prof("retry", [&]{
                CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));   // 清 accumulate 的旧标志;不可重试行会重置
                dbg("[rtprof] 2 memset_overflow\n");   // R0:run_retry 编排队分解(spec85,dbg 时戳差=步耗时)
                d_rht  = decltype(d_rht)(dev_alloc(h_ovf * sizeof(long long)));
                d_rtab = decltype(d_rtab)(dev_alloc((h_ovf + 1) * sizeof(long long)));
                retry_prep_kernel<<<(h_ovf + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                    d_ovf_rows, h_ovf, d_flop, d_rht, d_row_ovf, d_overflow, A_cols);
                static int g_retry2 = -1;   // R1b:提前解析(prep 的 thrust scan 也要按它让路)
                if (g_retry2 < 0) { const char *e2 = getenv("RETRY2"); g_retry2 = (e2 && *e2) ? 1 : 0; }
                if (!g_retry2 || h_ovf > R2B_MAX_N)   // fused kernel 兼做 rtab scan
                    thrust::exclusive_scan(thrust::device_ptr<long long>(d_rht),
                                           thrust::device_ptr<long long>(d_rht + h_ovf),
                                           thrust::device_ptr<long long>(d_rtab));
                // 重试区槽位(按行 id;非重试行 0)。
                // RETRY2=1 子集快路(spec85 R1):排序 ovf 子集+子集 scan+scatter 写 d_roff/d_rslot
                // + 4 读合批 1 次 sync,替代全 A_rows memset+inclusive_scan+3 次阻塞 D2H(R0 解剖 ~195μs)
                // (g_retry2 已在 prep 段解析)
                long long r_slots = 0, last_h = 0, rt_prev = 0;
                if (!g_retry2) {
                    d_rslot = decltype(d_rslot)(dev_alloc(A_rows * sizeof(int)));
                    CHECK_CUDA(cudaMemset(d_rslot, 0, (size_t)A_rows * sizeof(int)));
                    retry_slot_kernel<<<(h_ovf + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_ovf_rows, h_ovf, d_flop, d_rslot, A_cols);
                    dbg("[rtprof] 3 prep_scan+rslot(全A_rows memset %d)\n", A_rows);
                    d_roff = decltype(d_roff)(dev_alloc((A_rows + 1) * sizeof(long long)));
                    CHECK_CUDA(cudaMemset(d_roff, 0, sizeof(long long)));
                    thrust::inclusive_scan(thrust::make_transform_iterator(thrust::device_ptr<int>(d_rslot), ToLL()),
                                           thrust::make_transform_iterator(thrust::device_ptr<int>(d_rslot + A_rows), ToLL()),
                                           thrust::device_ptr<long long>(d_roff + 1));   // Fix#3(审查):int 累加 2^31 回绕,同 est_scan 坑
                    CHECK_CUDA(cudaMemcpy(&r_slots, d_roff + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
                    dbg("[rtprof] 4 rslot_scan(全A_rows inclusive)+D2H#1\n");
                    CHECK_CUDA(cudaMemcpy(&last_h, d_rht + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(&rt_prev, d_rtab + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                } else if (h_ovf <= R2B_MAX_N) {
                    // R1b fused:排序+双 scan+scatter+尾值,单 kernel 单 D2H,零 thrust 零临时分配
                    static long long *d_r2tails = nullptr;
                    if (!d_r2tails) d_r2tails = (long long*)dev_alloc(3 * sizeof(long long));
                    if (A_rows > s_arows_p) {   // R1d:A_rows 级两数组持久
                        if (s_rslot_p) { dev_free(s_rslot_p); dev_free(s_roff_p); }
                        s_rslot_p = (int*)dev_alloc((size_t)A_rows * sizeof(int));
                        s_roff_p = (long long*)dev_alloc((size_t)(A_rows + 1) * sizeof(long long));
                        s_arows_p = A_rows;
                    }
                    d_rslot = s_rslot_p;        // 仅 ovf 条目有效
                    d_roff  = s_roff_p;         // 仅 ovf 条目被读
                    retry_fused_off_kernel<256><<<1, 256>>>(
                        d_ovf_rows, h_ovf, d_flop, A_cols, d_rht, d_rtab, d_rslot, d_roff, d_r2tails);
                    dbg("[rtprof] 3f fused sort+双scan+scatter(n=%d)\n", h_ovf);
                    long long h3[3];
                    CHECK_CUDA(cudaMemcpy(h3, d_r2tails, 3 * sizeof(long long), cudaMemcpyDeviceToHost));
                    r_slots = h3[0]; last_h = h3[1]; rt_prev = h3[2];
                    dbg("[rtprof] 4f 单次 24B D2H(r_slots=%lld)\n", r_slots);
                } else {
                    // 表 arena 总量 = rtab[h_ovf-1] + rht[h_ovf-1](exclusive scan 只写 [0..h_ovf-1])
                    int *rsort = (int*)dev_alloc((size_t)h_ovf * sizeof(int));
                    CHECK_CUDA(cudaMemcpy(rsort, d_ovf_rows, (size_t)h_ovf * sizeof(int), cudaMemcpyDeviceToDevice));
                    thrust::sort(thrust::device_ptr<int>(rsort), thrust::device_ptr<int>(rsort + h_ovf));
                    long long *rsub  = (long long*)dev_alloc((size_t)h_ovf * sizeof(long long));
                    long long *rosub = (long long*)dev_alloc((size_t)h_ovf * sizeof(long long));
                    retry_rsub_gather_kernel<<<(h_ovf + 127) / 128, 128>>>(rsort, h_ovf, d_flop, rsub, A_cols);
                    thrust::exclusive_scan(thrust::device_ptr<long long>(rsub),
                                           thrust::device_ptr<long long>(rsub + h_ovf),
                                           thrust::device_ptr<long long>(rosub));
                    d_rslot = s_rslot_p ? s_rslot_p : decltype(d_rslot)(dev_alloc(A_rows * sizeof(int)));        // R1d 持久(仅 ovf 条目有效)
                    d_roff  = s_roff_p ? s_roff_p : decltype(d_roff)(dev_alloc((A_rows + 1) * sizeof(long long)));   // 仅 ovf 条目被读
                    if (!s_rslot_p) { s_rslot_p = d_rslot; s_roff_p = d_roff; s_arows_p = A_rows; }
                    retry_suboff_scatter_kernel<<<(h_ovf + 127) / 128, 128>>>(rsort, h_ovf, rosub, rsub, d_rslot, d_roff);
                    dbg("[rtprof] 3r 子集路 sort+scan+scatter(n=%d)\n", h_ovf);
                    // 4 尾读合批:总量=rosub尾+rsub尾;rht/rtab 尾供 arena 定容
                    long long h4[4];
                    CHECK_CUDA(cudaMemcpyAsync(&h4[0], rsub + h_ovf - 1, sizeof(long long), cudaMemcpyDeviceToHost, 0));
                    CHECK_CUDA(cudaMemcpyAsync(&h4[1], rosub + h_ovf - 1, sizeof(long long), cudaMemcpyDeviceToHost, 0));
                    CHECK_CUDA(cudaMemcpyAsync(&h4[2], d_rht + h_ovf - 1, sizeof(long long), cudaMemcpyDeviceToHost, 0));
                    CHECK_CUDA(cudaMemcpyAsync(&h4[3], d_rtab + h_ovf - 1, sizeof(long long), cudaMemcpyDeviceToHost, 0));
                    CHECK_CUDA(cudaStreamSynchronize(0));
                    r_slots = h4[0] + h4[1];   // Σ子集槽位(exclusive_scan 补末元)
                    last_h = h4[2]; rt_prev = h4[3];
                    dbg("[rtprof] 4r 合批读回 4async→1sync\n");
                    dev_free(rsort); dev_free(rsub); dev_free(rosub);
                }
                if (r_slots > 0) {
                    long long rt_total = rt_prev + last_h;
                    if (g_retry2) {   // R1d:持久 arena 惰性扩容,免每轮 4 次 alloc
                        s_r2_arena_active = true;
                        if (rt_total > s_cap_tc) {
                            if (s_rtc) { dev_free(s_rtc); dev_free(s_rtv); }
                            s_rtc = (int*)dev_alloc((size_t)rt_total * sizeof(int));
                            s_rtv = (double*)dev_alloc((size_t)rt_total * sizeof(double));
                            s_cap_tc = rt_total;
                        }
                        if (r_slots > s_cap_k) {
                            if (s_rk) { dev_free(s_rk); dev_free(s_rv); }
                            s_rk = (unsigned long long*)dev_alloc((size_t)r_slots * sizeof(unsigned long long));
                            s_rv = (double*)dev_alloc((size_t)r_slots * sizeof(double));
                            s_cap_k = r_slots;
                        }
                        rtc = s_rtc; rtv = s_rtv; rk = s_rk; rv = s_rv;
                    } else {
                        rtc = decltype(rtc)(dev_alloc((size_t)rt_total * sizeof(int)));
                        rtv = decltype(rtv)(dev_alloc((size_t)rt_total * sizeof(double)));
                        rk  = decltype(rk)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                        rv  = decltype(rv)(dev_alloc((size_t)r_slots * sizeof(double)));
                    }
                    dbg("[rtprof] 5 D2H#2/3(rtab尾)+arena_alloc=%lld 槽\n", rt_total);
                    // docs/54:retry 免排序(仅 h_ovf<1000;web-Google 8720 行 +16% 教训 = 大集 csort 慢于 cub)
                    // v40 SMEM 重试车(study/16 ①:溢出载体——治"保险系数被迫开大"的根)。
                    // 按行分流:2×flop ≤ HASH_CAP ∧ flop ≥ 2^10 的行走 hash_spa(SMEM 原子,单批统一
                    // ht = pow2 ≥ 2×max(flop),负载 ≤0.5,数学不可再溢);鲸鱼行单独留老全局表路
                    // (positional 数组按子集重建,arena 复用)。RETRY_SMEM=0 全走老路。
                    static int g_retry_smem = -1;
                    if (g_retry_smem < 0) { const char *e = getenv("RETRY_SMEM"); g_retry_smem = (e && *e) ? atoi(e) : 1; }
                    int h_ovf_g = h_ovf;
                    std::vector<int> rows_s, rows_g;   // v40 按行分流结果,提升到两层可见:鲸鱼子集重建
                    long long mf = 0;                  // 必须复用 rows_g 本体(study/17 雷:二次重筛条件
                    long long mf_g = 0;                // R1c:全局路最大 flop → S 段数
                                                       // ≠ 分流补集 → h_ovf_g > rg.size() → 拷未初始化尾段
                                                       // = 垃圾行号 → retry_prep flop[]/row_ovf[] 越界读写)
                    if (g_retry_smem && h_ovf > 0) {
                        int *d_ovf_flop = decltype(d_ovf_flop)(dev_alloc((size_t)h_ovf * sizeof(int)));
                        gather_flop_kernel<<<(h_ovf + 255) / 256, 256>>>(d_ovf_rows, h_ovf, d_flop, d_ovf_flop, A_cols);
                        std::vector<int> hf(h_ovf), hr(h_ovf);
                        if (g_retry2) {
                            if (h_ovf > s_hpin_cap) {   // R1d:分流双读 pinned 合批(2 阻塞 drain→1);仅刀路径
                                if (s_hf) { cudaFreeHost(s_hf); cudaFreeHost(s_hr); }
                                CHECK_CUDA(cudaMallocHost(&s_hf, (size_t)h_ovf * sizeof(int)));
                                CHECK_CUDA(cudaMallocHost(&s_hr, (size_t)h_ovf * sizeof(int)));
                                s_hpin_cap = h_ovf;
                            }
                            CHECK_CUDA(cudaMemcpyAsync(s_hf, d_ovf_flop, (size_t)h_ovf * sizeof(int), cudaMemcpyDeviceToHost, 0));
                            CHECK_CUDA(cudaMemcpyAsync(s_hr, d_ovf_rows, (size_t)h_ovf * sizeof(int), cudaMemcpyDeviceToHost, 0));
                            CHECK_CUDA(cudaStreamSynchronize(0));
                            std::copy(s_hf, s_hf + h_ovf, hf.begin());
                            std::copy(s_hr, s_hr + h_ovf, hr.begin());
                        } else {   // 原路径原样(RETRY2=0 零触碰纪律)
                            CHECK_CUDA(cudaMemcpy(hf.data(), d_ovf_flop, (size_t)h_ovf * sizeof(int), cudaMemcpyDeviceToHost));
                            CHECK_CUDA(cudaMemcpy(hr.data(), d_ovf_rows, (size_t)h_ovf * sizeof(int), cudaMemcpyDeviceToHost));
                        }
                        dbg("[rtprof] 6 gather_flop+2×D2H(分流)\n");
                        dev_free(d_ovf_flop);
                        for (int t = 0; t < h_ovf; t++) {
                            // hf≈0 = 异源溢出标记(dense 组合路径,flop 无效)→ 留老路吃默认大表;
                            // 正常 hash 行:2×flop ≤ HASH_CAP 且 flop ≥ 2^10 才上 SMEM 车
                            if (hf[t] >= (1 << 10) && 2LL * hf[t] <= HASH_CAP) { rows_s.push_back(hr[t]); mf = std::max(mf, (long long)hf[t]); }
                            else { rows_g.push_back(hr[t]); mf_g = std::max(mf_g, (long long)hf[t]); }   // R1c:鲸鱼段数定 S
                        }
                        int h_ovf_s = (int)rows_s.size(); h_ovf_g = (int)rows_g.size();
                        if (h_ovf_s) {
                            int rht = WARP_SIZE;
                            while (rht < 2 * mf) rht <<= 1;
                            int *d_rows_s = decltype(d_rows_s)(dev_alloc((size_t)h_ovf_s * sizeof(int)));
                            CHECK_CUDA(cudaMemcpy(d_rows_s, rows_s.data(), (size_t)h_ovf_s * sizeof(int), cudaMemcpyHostToDevice));
                            size_t smem_r = (size_t)rht * (sizeof(int) + sizeof(double));
                            if (probe_rf_on())
                                hash_spa_kernel<0,1><<<h_ovf_s, 512, smem_r>>>(
                                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                                    d_rows_s, h_ovf_s, rht, d_roff,
                                    rk, rv, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, nullptr);
                            else
                                hash_spa_kernel<0,0><<<h_ovf_s, 512, smem_r>>>(
                                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                                    d_rows_s, h_ovf_s, rht, d_roff,
                                    rk, rv, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf, nullptr);
                            CHECK_CUDA(cudaGetLastError());
                            dev_free(d_rows_s);
                            dbg("[hash] v40 SMEM 重试车: %d/%d 行 ht=%d(SMEM=%.0fKB);鲸鱼全局路 %d 行\n",
                                h_ovf_s, h_ovf, rht, smem_r / 1024.0, h_ovf_g);
                        }
                    }
                    bool retry_sortfree = h_ovf < RETRY_SORTFREE_MAX;
                    if (h_ovf_g) {
                        const int *glb_rows = d_ovf_rows; int glb_n = h_ovf;
                        long long *rht_g = d_rht, *rtab_g = d_rtab;
                        if (h_ovf_g != h_ovf) {   // 子集:positional 重建成(arena 复用,子集总量 ≤ 全量)
                            int *d_rows_g = decltype(d_rows_g)(dev_alloc((size_t)h_ovf_g * sizeof(int)));
                            // 直接复用分流补集 rows_g(与 h_ovf_g 严格同源);旧实现二次 gather 后按
                            // "2×flop>HASH_CAP" 重筛,漏掉 flop<2^10 的非车行(含异源标记)→ h_ovf_g
                            // > rg.size() → 拷未初始化尾段 = 垃圾行号 → retry_prep 越界读写(study/17 雷,
                            // memcheck 实锤:pre2 138 溢出行 47 车 + 91 非车,rg<91 → 双卡各楔死一次)
                            CHECK_CUDA(cudaMemcpy(d_rows_g, rows_g.data(), (size_t)h_ovf_g * sizeof(int), cudaMemcpyHostToDevice));
                            rht_g = decltype(rht_g)(dev_alloc((size_t)h_ovf_g * sizeof(long long)));
                            rtab_g = decltype(rtab_g)(dev_alloc((size_t)(h_ovf_g + 1) * sizeof(long long)));
                            retry_prep_kernel<<<(h_ovf_g + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                                d_rows_g, h_ovf_g, d_flop, rht_g, d_row_ovf, d_overflow, A_cols);
                            thrust::exclusive_scan(thrust::device_ptr<long long>(rht_g),
                                                   thrust::device_ptr<long long>(rht_g + h_ovf_g),
                                                   thrust::device_ptr<long long>(rtab_g));
                            glb_rows = d_rows_g; glb_n = h_ovf_g;
                        }
                        int S_r2 = 1;
                        if (g_retry2 && mf_g > 4096)
                            S_r2 = (int)std::min<long long>(8, (mf_g + 4095) / 4096);   // R1c:鲸鱼行 k 值域 S 段并行
                        if (S_r2 > 1) {
                            CHECK_CUDA(cudaMemsetAsync(rtc, 0xff, (size_t)rt_total * sizeof(int), 0));   // col=-1
                            CHECK_CUDA(cudaMemsetAsync(rtv, 0, (size_t)rt_total * sizeof(double), 0));
                            dim3 g3(glb_n, S_r2);
                            retry2_gprobe_kernel<<<g3, HASH_BLOCK>>>(
                                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                                glb_rows, glb_n, S_r2, rht_g, rtab_g, rtc, rtv,
                                d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                            retry2_gextract_kernel<<<glb_n, HASH_BLOCK>>>(
                                glb_rows, glb_n, rht_g, rtab_g, rtc, rtv,
                                d_roff, d_rslot, rk, rv, d_row_nnz, d_overflow);
                            dbg("[rtprof] 7g 全局路拆段 S=%d(n=%d)\n", S_r2, glb_n);
                        } else {
                            hash_global_kernel<<<glb_n, HASH_BLOCK>>>(
                                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                                glb_rows, glb_n, rht_g, rtab_g, rtc, rtv,
                                d_roff, d_rslot, r_slots, rk, rv, d_row_nnz, d_overflow,
                                d_ovf_rows, d_ovf_cnt, d_row_ovf);
                        }
                    }
                    // retry_prep 遇不可重试行会重置 overflow → 仍有则整阵回退
                    d_retry_rows = d_ovf_rows; d_retry_n = h_ovf;
                    d_retry_off = d_roff; d_retry_key = rk; d_retry_val = rv;
                    dbg("[rtprof] 7 车发射+全局路完成\n");
                    if (retry_sortfree) {
                        d_retry_needs_sort = true;
                        dbg("[hash] docs/54 retry 免排序: %d 行(r_slots=%lld)→csort\n", h_ovf, r_slots);
                    } else {
                        // 大 retry 集:回退 cub 分段排序(单次批量排序 > 逐行 csort)
                        rk2 = decltype(rk2)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                        rv2 = decltype(rv2)(dev_alloc((size_t)r_slots * sizeof(double)));
                        d_rsb = decltype(d_rsb)(dev_alloc(h_ovf * sizeof(int)));
                        d_rse = decltype(d_rse)(dev_alloc(h_ovf * sizeof(int)));
                        heavy_seg_kernel<<<(h_ovf + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                            d_ovf_rows, h_ovf, d_roff, d_row_nnz, d_rsb, d_rse);
                        size_t rcb = 0;
                        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                            nullptr, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                        rct = dev_alloc(rcb);
                        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                            rct, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                        d_retry_key = rk2; d_retry_val = rv2;
                        d_retry_needs_sort = false;
                    }
                }
                dbg("[rtprof] 8 retry排序段完成\n");
            });
        }
    };  // D5H:count/legacy 的 ovf 都在 scan 前覆盖(retry 产精确 row_nnz;MODE=2 跳过 ovf 行)
    // TMPSCAN=<row>:相位时间线仪器(study/18 残余雷:tmp_val 何时出现 key 格式脏比特)
    static int g_tmpscan = -1;
    if (g_tmpscan < 0) { const char *e = getenv("TMPSCAN"); g_tmpscan = (e && *e) ? atoi(e) : -1; }
    auto tmpscan = [&](const char *phase) {
        if (g_tmpscan < 0 || !d_tmp_val) return;
        int r = g_tmpscan;
        long long bo[2]; int er;
        CHECK_CUDA(cudaMemcpy(bo, d_off + r, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&er, d_est + r, sizeof(int), cudaMemcpyDeviceToHost));
        int n = (int)std::min<long long>(bo[1] - bo[0], 4096);
        if (n <= 0) return;
        static double *h_sv = nullptr;
        if (!h_sv) h_sv = (double*)malloc(4096 * sizeof(double));
        CHECK_CUDA(cudaMemcpy(h_sv, d_tmp_val + bo[0], (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
        for (int t = 0; t < n; t++) {
            double v = h_sv[t];
            if (v != 0.0 && fabs(v) < 1e-290) {   // denormal/整型比特
                unsigned long long bits; memcpy(&bits, &v, 8);
                int hi = (int)(bits >> 32), lo = (int)(bits & 0xffffffffu);
                dbg("[TMPSCAN] %s: slot[%d]=%.6e bits=0x%016llx(row=%d est=%d base=%lld) → (hi=%d lo=%d)\n",
                    phase, t, v, bits, r, er, bo[0], hi, lo);
                // 邻域 5 槽的 key+val 对照 + 嫌疑行(hi)的 base/est
                if (hi >= 0 && hi < A_rows && hi != r) {
                    long long hb[2]; int he;
                    CHECK_CUDA(cudaMemcpy(hb, d_off + hi, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(&he, d_est + hi, sizeof(int), cudaMemcpyDeviceToHost));
                    dbg("[TMPSCAN]    嫌疑行 %d: base=%lld est=%d(本行 base=%lld → 距离 %lld)\n",
                        hi, hb[0], he, bo[0], bo[0] - hb[0]);
                }
                static unsigned long long *h_sk = nullptr;
                if (!h_sk) h_sk = (unsigned long long*)malloc(4096 * sizeof(unsigned long long));
                int t0 = t > 2 ? t - 2 : 0;
                CHECK_CUDA(cudaMemcpy(h_sk, d_tmp_key + bo[0] + t0, 5 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_sv + 2048, d_tmp_val + bo[0] + t0, 5 * sizeof(double), cudaMemcpyDeviceToHost));
                for (int u = 0; u < 5; u++)
                    dbg("[TMPSCAN]    slot[%d]: key=0x%016llx val=%.6g\n", t0+u, h_sk[u], h_sv[2048+u]);
                return;
            }
        }
        dbg("[TMPSCAN] %s: clean(row=%d est=%d)\n", phase, r, er);
    };
    tmpscan("pre-retry");
    run_retry();
    tmpscan("post-retry");

    CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (!g_d5h_on && overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        if (!s_r2_arena_active) { dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); }   // R1d:持久 arena 不逐轮 free
        dev_free(rk2); dev_free(rv2); dev_free(rct);   // docs/21 Fix0
        dev_free(d_rht); dev_free(d_rtab);
        if (!s_r2_arena_active) { dev_free(d_rslot); dev_free(d_roff); }
        dev_free(d_rsb); dev_free(d_rse);
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); for (int vb_ = 0; vb_ <= HASH_LADDER_MAX_BIN; vb_++) dev_free(d_gv_bins[vb_]); dev_free(d_gv_off);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val); dev_free(dB16);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        dev_free(d_flop);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; dC_rp = decltype(dC_rp)(dev_alloc((A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    if (h_d3_n > 0 && d_d3_rows) {   // D3 count(T7c 修复v2:必须先于 cnnz_scan,原位置嵌在不执行分支=prefix 漏 D3 行根因)
        size_t d3sm = (size_t)16384 * 9 + 256;
        CHECK_CUDA(cudaFuncSetAttribute(dense_struct_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)d3sm));
        dense_struct_count_kernel<<<h_d3_n, HASH_BLOCK_STRUCT, d3sm>>>(
            dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows,
            d_d3_rows, d_span_lo, d_span_len, 16384, d_row_nnz);
        CHECK_CUDA(cudaGetLastError());
        dbg("[%s] D3 count v2:%d 行\n", tag, h_d3_n);
        {   // T7c 取证:count 后即刻 Σ(分"count 值错"vs"prefix/流程错")
            std::vector<int> rn(A_rows);
            CHECK_CUDA(cudaMemcpy(rn.data(), d_row_nnz, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> rr2(h_d3_n);
            CHECK_CUDA(cudaMemcpy(rr2.data(), d_d3_rows, h_d3_n * sizeof(int), cudaMemcpyDeviceToHost));
            long long sum = 0;
            for (int z = 0; z < h_d3_n; z++) sum += rn[rr2[z]];
            dbg("[%s] D3 count 后 Σ=%lld(对比 numeric Σ=110.3M)\n", tag, sum);
        }
    }
    prof("cnnz_scan", [&]{
        CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
        if (A_rows <= MAX_BLOCK_THREADS) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<int><<<1, b, b * sizeof(int)>>>(d_row_nnz, dC_rp + 1, A_rows);
        } else {
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                                   thrust::device_ptr<int>(d_row_nnz + A_rows),
                                   thrust::device_ptr<int>(dC_rp + 1));
        }
        CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });
    dbg("[hash] C_nnz=%d (est=%lld, %.2fx over-alloc)\n", C_nnz_result, total_est,
        total_est > 0 ? (double)total_est / C_nnz_result : 0.0);

    // study/47 修复(10-20):C_nnz 为负 = 行计数和溢出 int32(c-73 级巨输出阵,DCEST 的
    // est×1.2 padding 和可达 3.5e9 > 2^31)。负值绕过现有判废条件(actual>est 在负数域失效)
    // → 曾带负值进 compact 引爆 cudaMalloc size_t 下溢(1.8e19B)。修复:负值即判废回退
    // merge3(int32 CSR 本就装不下该量级,回退是唯一正确语义,复用 underflow 收尾)。
    if (C_nnz_result < 0) {
        fprintf(stderr, "[hash] C_nnz int32 溢出(C_nnz=%d, est=%lld)→ 判废回退 merge\n",
                C_nnz_result, total_est);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        if (!s_r2_arena_active) { dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); }
        dev_free(rk2); dev_free(rv2); dev_free(d_rht); dev_free(d_rtab);
        if (!s_r2_arena_active) { dev_free(d_rslot); dev_free(d_roff); }
        dev_free(d_rsb); dev_free(d_rse);
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); for (int vb_ = 0; vb_ <= HASH_LADDER_MAX_BIN; vb_++) dev_free(d_gv_bins[vb_]); dev_free(d_gv_off);
        dev_free(dC_rp);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val); dev_free(dB16);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        dev_free(d_flop);
        return;
    }

    // est underflow check: actual C_nnz > estimated total → 回退 merge3
    // 方案5:dense 行计数精确(可 > 其 est,合法),检查改 hash 侧 = C_nnz − Σnnz_dense ≤ tmp_slots
    // DCEST:dense 行用 est×1.2(over-allocated)→ C_nnz 含 padding,检查必假阳 → 跳过
    if (!g_cnf && !g_dcest) {   /* CNT_FUSE: underflow check deferred to late cnnz; DCEST: fixup 后校正 */
    if ((dense_nr > 0 || d5h_rows > 0 || k7_nr > 0)
            ? ((long long)C_nnz_result - dense_nnz_sum - d5h_nnz_sum - k7_nnz_sum > tmp_slots)
            : ((long long)C_nnz_result > total_est)) {
        fprintf(stderr, "[hash] est underflow: C_nnz=%d > total_est=%lld → 回退 merge"
                        " [组件: dense_nr=%d dense_sum=%lld d5h_rows=%d d5h_sum=%lld tmp_slots=%lld hash侧=%lld]\n",
                C_nnz_result, total_est, dense_nr, dense_nnz_sum, d5h_rows, d5h_nnz_sum,
                tmp_slots, (long long)C_nnz_result - dense_nnz_sum - d5h_nnz_sum);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        if (!s_r2_arena_active) { dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); }   // R1d:持久 arena 不逐轮 free
        dev_free(rk2); dev_free(rv2); dev_free(rct);   // docs/21 Fix0
        dev_free(d_rht); dev_free(d_rtab);
        if (!s_r2_arena_active) { dev_free(d_rslot); dev_free(d_roff); }
        dev_free(d_rsb); dev_free(d_rse);
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); for (int vb_ = 0; vb_ <= HASH_LADDER_MAX_BIN; vb_++) dev_free(d_gv_bins[vb_]); dev_free(d_gv_off);
        dev_free(dC_rp);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val); dev_free(dB16);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        dev_free(d_flop);
        return;
    }
    }  /* end !g_cnf wrap */
    /* late cnnz relocated after direct block */

    // Stage 4+5+6: per-row compact+sort(BlockRadixSort)替掉 compact+全局 sort+split_key;按 bin 选 config,直写连续 dC
    size_t C_rp_sz  = (A_rows + 1) * sizeof(int);
    long long C_alloc_n = g_cnf ? (total_est * 2) : (long long)C_nnz_result;   /* #4': 融合态早 cnnz 为垃圾,按 est×2 上界分配(memcheck 越界修) */
    size_t C_ci_sz  = (size_t)C_alloc_n * sizeof(int);
    size_t C_v_sz   = (size_t)C_alloc_n * sizeof(double);
    size_t C_rp_al  = ALIGN8(C_rp_sz);
    size_t C_ci_al  = ALIGN8(C_ci_sz);
    size_t C_total  = C_rp_al + C_ci_al + C_v_sz;
    // ---- compact 原地化(docs/21 Fix2):dC(12B×C)不再新分配,col/val 直接前缀紧缩进 tmp 自己。
    //      安全条件 = ∀j: 前缀 A(j)=Σactual ≤ E(j)=Σest(d_est 原始值,retry 行的欠估已计入)——
    //      此时行 i 的写区间 [4A(i), 4A(i+1)) 恒低于任何 j>i 行的读区间 [8E(j),…) ✓(含 retry 行
    //      从 rk2 的写)。违例(retry 欠估累计 > 正常行富余)→ 回退旧路径。whale 阵省 C×12GB 级峰值。
    bool in_place = false;
    {
        long long *d_eoff = decltype(d_eoff)(dev_alloc((A_rows + 1) * sizeof(long long)));
        CHECK_CUDA(cudaMemset(d_eoff, 0, sizeof(long long)));
        thrust::inclusive_scan(thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                               thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                               thrust::device_ptr<long long>(d_eoff + 1));
        // margin = min_j (E(j) − C_rp[j]);dC_rp 是 int(实际行和 ≤ 2^31 保证:C_nnz 是 int)
        long long *d_marg = decltype(d_marg)(dev_alloc((A_rows + 1) * sizeof(long long)));
        margin_kernel<<<(A_rows + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(d_eoff, dC_rp, A_rows + 1, d_marg);
        long long min_marg;
        thrust::device_ptr<long long> mmin = thrust::min_element(thrust::device_ptr<long long>(d_marg), thrust::device_ptr<long long>(d_marg) + A_rows + 1);
        CHECK_CUDA(cudaMemcpy(&min_marg, mmin.get(), sizeof(long long), cudaMemcpyDeviceToHost));
        // ⚠ 2026-08-27 rajat16 实测 294411 行内乱序(原 0)= 某紧凑路径交叉写读(疑 csort 多趟/
        //   gapped 区间重读)。默认关闭,守卫+代码保留,FIX2=1 可开(普通阵 margin 普遍 ≥0)
        static int g_fix2 = -1;
        if (g_fix2 < 0) { const char *e = getenv("FIX2"); g_fix2 = (e && *e) ? atoi(e) : 0; }
        in_place = (min_marg >= 0) && g_fix2 && ((long long)C_nnz_result > INPLACE_MIN_CNNZ)
                   && (dense_nr == 0) && (h_d3_n == 0) && !g_d5h_on && (k7_nr == 0);   // 方案5/D5H/K7 互斥:direct 写终态与 tmp 前缀紧缩重叠,且 est 已置 0 致 margin 失义
        dev_free(d_eoff); dev_free(d_marg);
        dbg("[%s] compact in_place=%d (min_margin=%lld)\n", tag, (int)in_place, min_marg);
    }
    void *dC = nullptr;
    int   *dC_ci; double *d_val_ip;
    if (in_place) {
        dC_ci = (int*)d_tmp_key;          // col 前缀紧缩进 key 区(4B 写 ≤ 8B 源,同一守卫)
        d_val_ip = d_tmp_val;             // val 原地(等宽,守卫直接适用)
    } else {
        dC = decltype(dC)(dev_alloc(C_total));
        char *cb = (char*)dC;
        dC_ci = (int*)(cb + C_rp_al);
        d_val_ip = (double*)(cb + C_rp_al + C_ci_al);
    }
    d_val = d_val_ip;
    // TMPSCAN 扩展:输出区(dC_val@dC_rp)扫描 —— compact 前后各一次(study/18:tmp 全净,垃圾在输出区)
    auto outscan = [&](const char *phase) {
        if (g_tmpscan < 0 || !d_val) return;
        int r = g_tmpscan;
        int dst; CHECK_CUDA(cudaMemcpy(&dst, dC_rp + r, sizeof(int), cudaMemcpyDeviceToHost));
        int n; CHECK_CUDA(cudaMemcpy(&n, d_row_nnz + r, sizeof(int), cudaMemcpyDeviceToHost));
        if (n <= 0 || n > 4096) return;
        static double *h_ov = nullptr;
        if (!h_ov) h_ov = (double*)malloc(4096 * sizeof(double));
        CHECK_CUDA(cudaMemcpy(h_ov, d_val + dst, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
        int bad = -1;
        for (int t = 0; t < n; t++) {
            double v = h_ov[t];
            if (v != 0.0 && fabs(v) < 1e-290) { bad = t; break; }
        }
        if (bad >= 0) {
            unsigned long long bits; memcpy(&bits, &h_ov[bad], 8);
            dbg("[TMPSCAN] %s: OUT dst=%d n=%d 垃圾槽=%d bits=0x%016llx\n", phase, dst, n, bad, bits);
        } else dbg("[TMPSCAN] %s: OUT clean(dst=%d n=%d)\n", phase, dst, n);
    };
    outscan("pre-compact");
    // 方案5 数值 pass(Phase B v2 游标版):dense 行数据驱动窗口直写终态
    int *d_smap = nullptr, *d_smap_off = nullptr;

    if (dense_nr > 0) {
        prof("dense_direct", [&]{
            // ---- §22d 分流复用(count 阶段已算好:档内段→档位梯数值;超档段→老窗口 cursor 内核)----
            static int g_dtier_single = -1;   // §22c:1 = 档内行单发最大档(实验开关;三臂实验定谳:
            if (g_dtier_single < 0) { const char *e = getenv("DTIER_SINGLE"); g_dtier_single = (e && *e) ? atoi(e) : 0; }  // per-tier vs 单档数值差异 ~0.2ms 噪声级,真凶是 count 窗与行序局部性,非块宽)
            const bool dstr2_on = (g_dstr2 >= 2 && dense_rows);
            // §26 采数:DSN_RT=path → struct numeric 逐行周期落盘(a_len 维成本律原料;
            // 指针经 cudaMemcpyToSymbol 装入,免改 5 个模板实例签名)
            unsigned long long *dsn_rt_dev = nullptr; FILE *dsn_f = nullptr;
            if (const char *dp = getenv("DSN_RT")) {
                if (dsn_rt_dev = decltype(dsn_rt_dev)(dev_alloc((size_t)A_rows * sizeof(unsigned long long)))) {
                    CHECK_CUDA(cudaMemset(dsn_rt_dev, 0, (size_t)A_rows * sizeof(unsigned long long)));
                    dsn_rt_attach(dsn_rt_dev);
                    dsn_f = fopen(dp, "w");
                }
            }
            const int big_off = dstr2_on ? h_big_off : dense_nr;   // 超档段起点(=老内核人群)
            const int *tier_end_p = h_tier_end;
            const int *old_rows = dstr2_on ? (big_off < dense_nr ? d_rows_sorted + big_off : nullptr) : dense_rows;
            const int  old_nr   = dstr2_on ? (dense_nr - big_off) : dense_nr;
            // ---- 老 kernel 人群的 cursor 机械(全局 cursor 区:每行 a_len 个 int,偏移 = alen 按行
            //      散射后的 inclusive scan;Σ ≤ nnz)----
            int sm_tot = 0, uc = 1;
            static int g_pb2avgb = -1;   // docs/66 §8 复审:avgB<64→search 门(docs/39 v3 遗产)
            if (g_pb2avgb < 0) { const char *e = getenv("PB2_AVGB"); g_pb2avgb = (e && *e) ? atoi(e) : 0; }
            if (old_nr > 0) {
                int *d_alen = decltype(d_alen)(dev_alloc((size_t)old_nr * sizeof(int)));
                d_smap_off = decltype(d_smap_off)(dev_alloc((A_rows + 1) * sizeof(int)));
                alen_gather_kernel<<<(old_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(old_rows, old_nr, dA_rp, d_alen);
                CHECK_CUDA(cudaMemset(d_smap_off, 0, (size_t)(A_rows + 1) * sizeof(int)));
                scatter_alen_kernel<<<(old_nr + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(old_rows, old_nr, d_alen, d_smap_off);
                // ⚠ exclusive(偏移语义)+ 无槽位移位:off[i] = Σ_{r<i} alen[r];off[A_rows] = 总量。
                //   (首版 inclusive+1 槽双错:行 chunk 右移互相覆盖 → 游标读到别行 B 位置 → j<0 →
                //    SMEM 下越界 → "illegal instruction";docs/21 同款陷阱再现)
                thrust::exclusive_scan(thrust::device_ptr<int>(d_smap_off),
                                       thrust::device_ptr<int>(d_smap_off + A_rows + 1),
                                       thrust::device_ptr<int>(d_smap_off));
                CHECK_CUDA(cudaMemcpy(&sm_tot, d_smap_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
                d_smap = decltype(d_smap)(dev_alloc((size_t)(sm_tot > 0 ? 2 * sm_tot : 1) * sizeof(int)));   // 2×:活动游标 + 本窗镜像
                dev_free(d_alen);
                size_t wsm = ((size_t)PB2_W * 9 + 3) / 4 * 4 + (size_t)PB2_W * sizeof(int)
                           + 2 * (size_t)SMAP_SMEM_MAX * sizeof(int);
                CHECK_CUDA(cudaFuncSetAttribute(hash_dense_direct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
                static int g_pb2cur = -1;
                if (g_pb2cur < 0) { const char *e = getenv("PB2_CURSOR"); g_pb2cur = (e && *e) ? atoi(e) : 1; }
                // docs/66 §8 终审:dup≥4→search 门删除。三代证据:①逐行周期 cursor 六阵全胜 1.24-6.3×
                // (mult_dcop/3Dspec2/brainpc2/TSOPF/c-58/c-64),无特征门可让 search 赢;②交替矩阵级
                // cursor 5/6 不劣;③"Cube_Coup +263% 灾难"为幻影 —— 其 DD_ROWTIME 0 行,hash_dense_direct
                // 对它零调用,uc 惰性(强制 cursor/search 两版 675-831ms 打平 = 同执行纯噪声)。
                // docs/39 v4 与 docs/63 §6 的门均测于污染窗(web-Google 路由不变 ±70% 铁证)。
                // PB2_CURSOR=0 仍可强制 search(实验用);2 = 兼容别名(恒 cursor)。
                uc = (g_pb2cur == 0) ? 0 : 1;
            }
            // ---- 档位梯 per-tier 发射(窗 = 档上限,块宽 = 档块宽;1:1 落 5 个显式实例)----
            if (dstr2_on) {
                if (h_d3_n > 0 && d_d3_rows) {   // D3:merge 核(pre2 型;count 已出 exact_rp)
                    int wpb = 256 >> 5;
                    d3_merge_kernel<<<(h_d3_n + wpb - 1) / wpb, 256>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        d_d3_rows, h_d3_n, dC_rp, dC_ci, d_val, d_row_nnz, d_rt);
                    CHECK_CUDA(cudaGetLastError());
                    dbg("[%s] D3 merge 核:%d 行(warp/行)\n", tag, h_d3_n);
                    if (getenv("D3CHK")) {   // T7c 取证:实际 row_nnz dump
                        std::vector<int> rn(A_rows), rr(h_d3_n);
                        CHECK_CUDA(cudaMemcpy(rr.data(), d_d3_rows, h_d3_n * sizeof(int), cudaMemcpyDeviceToHost));
                        CHECK_CUDA(cudaMemcpy(rn.data(), d_row_nnz, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
                        FILE *f = fopen(getenv("D3CHK"), "w");
                        for (int z = 0; z < h_d3_n; z++) fprintf(f, "%d %d\n", rr[z], rn[rr[z]]);
                        fclose(f);
                    }
                }
                if (g_dtier_single) {   // §22c:档内行全部单发最大档(窗 7277/块宽 1024;§22 数值形态)
                    if (big_off > 0) {
                        const int t = DENSE_TIER_COUNT - 1;
                        const int uwin = dense_tier_max(t);
                        const int tblk = dense_tier_block(t);
                        size_t nsm = (size_t)uwin * sizeof(double) + (size_t)((uwin + 31) >> 5) * sizeof(unsigned) * 2 + 64;
                        CHECK_CUDA(cudaFuncSetAttribute(dense_struct_numeric_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nsm));
                        dense_struct_numeric_kernel<1024><<<big_off, 1024, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        dbg("[%s]   §22c 单档:%d 行(窗 %d 块宽 %d)\n", tag, big_off, uwin, tblk);
                    }
                } else if (getenv("T3EXP") && atoi(getenv("T3EXP")) > 0) {
                    // T3:两行共块 + 子分档(每子档一发射;grid=(n+1)/2;SMEM=2×半区)
                    int off = 0;
                    for (int t = 0; t < DENSE_TIER_COUNT * 2; t++) {
                        int nr_t = h_t3_end[t] - off;
                        int tier = t >> 1;
                        int tblk0 = dense_tier_block(tier);
                        if (nr_t > 0 && tblk0 < 256) {
                            // T3 修复#1:小档(tblk<256)不走 2r(块宽<模板 TPB → half1 行被丢)
                            // → 单行原 kernel 兜底(人口小:in-2004 共 ~200 行)
                            int uwin0 = h_t3_w[t] > 0 ? h_t3_w[t] : dense_tier_max(tier);
                            size_t nsm0 = (size_t)uwin0 * 8 + (((size_t)uwin0 + 31) >> 5) * 4 * 2;
                            if (tblk0 <= 64) dense_struct_numeric_kernel<64><<<nr_t, 64, nsm0>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin0, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                            else dense_struct_numeric_kernel<128><<<nr_t, 128, nsm0>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin0, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                            CHECK_CUDA(cudaGetLastError());
                            off = h_t3_end[t]; continue;
                        }
                        if (nr_t > 0) {
                            int uwin = h_t3_w[t] > 0 ? h_t3_w[t] : dense_tier_max(tier);
                            int tblk = tblk0;
                            size_t half_bytes = (size_t)uwin * 8 + (((size_t)uwin + 31) >> 5) * 4 * 2;
                            size_t nsm = half_bytes * 2;
                            auto k2r = (tblk <= 256) ? dense_struct_numeric_2r_kernel<256>
                                    : (tblk <= 512) ? dense_struct_numeric_2r_kernel<512>
                                                    : dense_struct_numeric_2r_kernel<1024>;
                            CHECK_CUDA(cudaFuncSetAttribute(k2r, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nsm));
                            k2r<<<(nr_t + 1) / 2, tblk, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, nr_t, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                            CHECK_CUDA(cudaGetLastError());
                            dbg("[%s]   T3 子档%d:%d 行(窗 %d 块宽 %d smem %zuKB 2r)\n", tag, t, nr_t, uwin, tblk, nsm / 1024);
                        }
                        off = h_t3_end[t];
                    }
                } else {
                int off = 0;
                for (int t = 0; t < DENSE_TIER_COUNT; t++) {
                    int nr_t = tier_end_p[t] - off;
                    if (nr_t > 0) {
                        // §26:只缩【最大档】窗(tier-5 固定 131KB 把刚过 7277 的行压到 1 block/SM);
                        // 低档不动 —— wb-edu 实测全档缩窗 +4%(占用↑ → 行工作集并发↑ → L2 局部性
                        // 反损,与 §24 排序毁局部性同族),tier-5 缩窗只升不降(≥2/SM)
                        const int uwin = (t == DENSE_TIER_COUNT - 1 && h_tier_w[t] > 0) ? h_tier_w[t] : dense_tier_max(t);   // D1v2 全档收窗=中性(×3 wash)已回退;tier5 收窗保留(§26 原逻辑)
                        int tblk = dense_tier_block(t);
                        // 刀D1(09-30 审计):同行集实测彼 512T 在 span3636-7277 带赢我 1024T 15-19%,
                        // 256T 在 1818-3636 带赢 1.67×;§22c"块宽无差"定谳系矩阵级噪声口径,作废。
                        // tier3→256/tier4→512(巨鲸 tier5 保 1024 = 我方唯一占优带);env D1TIER。
                        static int g_d1tier = -1;
                        if (g_d1tier < 0) { const char *e = getenv("D1TIER"); g_d1tier = (e && *e && atoi(e) > 0) ? 1 : 0; }
                        if (g_d1tier) {
                            size_t sm_b = (size_t)uwin * 9 + 256;   // ≈SMEM/块(val8+bitmap1+杂)
                            int fit = (int)((SMEM_DEFAULT_MAX - SMEM_STATIC_HEADROOM) / sm_b); if (fit < 1) fit = 1;
                            if (fit > 2048 / 64) fit = 2048 / 64;
                            int want = 2048 / fit;                  // 满占用的块宽
                            if (want > 1024) want = 1024;
                            if (want < 128) want = 128;
                            if (want < tblk) tblk = want;           // 只降不升(巨鲸 1024 保)
                        }
                        // spec90-候选2 二维梯解耦(D2TIER=1):工作量轴覆写——tier 行群均 est 低
                        // (轻行,车道需求小)→降块宽换块数;est 高(重行)保持 span 档默认。
                        static int g_d2tier = -1;
                        if (g_d2tier < 0) { const char *e = getenv("D2TIER"); g_d2tier = (e && *e && atoi(e) > 0) ? 1 : 0; }
                        if (g_d2tier) {
                            static unsigned long long *d_tsum = nullptr;
                            if (!d_tsum) { d_tsum = decltype(d_tsum)(dev_alloc(sizeof(unsigned long long))); }
                            CHECK_CUDA(cudaMemsetAsync(d_tsum, 0, sizeof(unsigned long long), (cudaStream_t)0));
                            tier_est_stat_kernel<<<(nr_t + 255) / 256, 256>>>(d_est_save ? d_est_save : d_est, d_rows_sorted + off, nr_t, d_tsum);
                            unsigned long long se = 0;
                            CHECK_CUDA(cudaMemcpy(&se, d_tsum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                            int mean_est = nr_t > 0 ? (int)(se / nr_t) : 0;
                            int ovl = tblk;
                            if (mean_est < 1024) ovl = tblk / 4;
                            else if (mean_est < 4096) ovl = tblk / 2;
                            if (ovl < 64) ovl = 64;
                            if (ovl != tblk) { tblk = ovl; dbg("[%s]   D2TIER tier %d: mean_est=%d → 块宽 %d\n", tag, t, mean_est, tblk); }
                        }
                        size_t nsm = (size_t)uwin * sizeof(double) + (size_t)((uwin + 31) >> 5) * sizeof(unsigned) * 2 + 64;
                        CHECK_CUDA(cudaFuncSetAttribute(
                            (tblk <= 64) ? dense_struct_numeric_kernel<64>
                            : (tblk <= 128) ? dense_struct_numeric_kernel<128>
                            : (tblk <= 256) ? dense_struct_numeric_kernel<256>
                            : (tblk <= 512) ? dense_struct_numeric_kernel<512>
                                            : dense_struct_numeric_kernel<1024>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nsm));
                        if (tblk <= 64)        dense_struct_numeric_kernel<64><<<nr_t, 64, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        else if (tblk <= 128) dense_struct_numeric_kernel<128><<<nr_t, 128, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        else if (tblk <= 256) dense_struct_numeric_kernel<256><<<nr_t, 256, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        else if (tblk <= 512) dense_struct_numeric_kernel<512><<<nr_t, 512, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        else                   dense_struct_numeric_kernel<1024><<<nr_t, 1024, nsm>>>(dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, d_rows_sorted + off, d_span_lo, d_span_len, uwin, dC_rp, dC_ci, d_val, d_row_nnz, d_flop);
                        dbg("[%s]   tier %d:%d 行(窗 %d 块宽 %d smem %zuKB)\n", tag, t, nr_t, uwin, tblk, nsm / 1024);
                    }
                    off = tier_end_p[t];
                }
                }
                CHECK_CUDA(cudaGetLastError());
            }
            // ---- 老窗口 cursor 内核(大 span 段;或 DSTRUCT<2 的全体 dense 行)----
            if (old_nr > 0) {
                size_t wsm = ((size_t)PB2_W * 9 + 3) / 4 * 4 + (size_t)PB2_W * sizeof(int)
                           + 2 * (size_t)SMAP_SMEM_MAX * sizeof(int);
                if (const char *dp2 = getenv("DFLAG_PROBE")) { int pr = atoi(dp2); cudaMemcpyToSymbol(g_dflag_probe_row, &pr, sizeof(int)); }
                static int g_cb16 = -1;   // spec86-A:CURSOR_B16=1 门(默认关;hash_spa 的 B16 分裂人格前科,系统电池裁决)
                if (g_cb16 < 0) { const char *e3 = getenv("CURSOR_B16"); g_cb16 = (e3 && *e3) ? 1 : 0; }
                static int g_cpf = -1;   // spec86-B:CURSOR_PF=1 门(默认关)
                static int g_cbat = -1;   // spec89-#1:CURSOR_BATCH=1 门(默认关)
                if (g_cbat < 0) { const char *e5 = getenv("CURSOR_BATCH"); g_cbat = (e5 && *e5) ? 1 : 0; }
                // (g_cnf 已提前解析)
                /* d_dtmp alloc moved to function scope */
                if (g_cpf < 0) { const char *e4 = getenv("CURSOR_PF"); g_cpf = (e4 && *e4) ? 1 : 0; }
                static int g_k1fwHost = -1;   // K1 首写免原子(env K1FW;默认关)
                if (g_k1fwHost < 0) { const char *e = getenv("K1FW"); g_k1fwHost = (e && *e && atoi(e) > 0) ? 1 : 0; }
                hash_dense_direct_kernel<<<old_nr, MAX_BLOCK_THREADS, wsm>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                    dC_rp, dC_ci, d_val, d_row_nnz, old_rows, old_rows ? d_span_lo : nullptr,
                    d_flop, d_maxbl, d_smap, d_smap_off, sm_tot, uc, g_k1fwHost, d_ddrt, d_ddmode, g_pb2avgb,
                    (g_cb16 && dB16) ? dB16 : nullptr, g_cpf, g_cbat,
                    (g_cnf && d_dtmp_col) ? d_dtmp_col : nullptr,
                    (g_cnf && d_dtmp_col) ? d_dtmp_val : nullptr,
                    (g_cnf && d_dtmp_col) ? d_off : nullptr,   // spec89-#4'-L2a:tmp 三参(默认 null=旧路)
                    (g_cnf && d_dtmp_col) ? d_drop : nullptr);   // #4'-fix1:丢条检出器
                CHECK_CUDA(cudaGetLastError());
            }
            if (d_rows_sorted) dev_free(d_rows_sorted);
            if (getenv("DDUMP_ROW")) {   // diter 雷探针(study/18):指定行的 dC_ci 切片 + 修正后 row_nnz
                int dr = atoi(getenv("DDUMP_ROW"));
                int b2, rn2, rn0;
                CHECK_CUDA(cudaMemcpy(&b2, dC_rp + dr, sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&rn2, d_row_nnz + dr, sizeof(int), cudaMemcpyDeviceToHost));
                int cap0 = b2 + (rn2 > 0 ? rn2 : 0);
                CHECK_CUDA(cudaMemcpy(&cap0, dC_rp + dr + 1, sizeof(int), cudaMemcpyDeviceToHost));
                static int h_ci[4096];
                int n2 = cap0 - b2; if (n2 > 4096) n2 = 4096;
                CHECK_CUDA(cudaMemcpy(h_ci, dC_ci + b2, (size_t)n2 * sizeof(int), cudaMemcpyDeviceToHost));
                dbg("[DDUMP] row %d: base=%d prefix_len=%d corrected_row_nnz=%d 首12列=", dr, b2, cap0 - b2, rn2);
                for (int t = 0; t < 12 && t < n2; t++) dbg("%d ", h_ci[t]);
                dbg("\n");
            }
            if (dsn_f) {   // §26:struct numeric 逐行周期落盘(非零行 = 档位梯人群)
                std::vector<unsigned long long> rt(A_rows);
                CHECK_CUDA(cudaMemcpy(rt.data(), dsn_rt_dev, (size_t)A_rows * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                for (int r = 0; r < A_rows; r++) if (rt[r]) fprintf(dsn_f, "%d %llu\n", r, rt[r]);
                fclose(dsn_f);
                dsn_rt_attach(nullptr);
                dev_free(dsn_rt_dev);
                dbg("[%s] DSN_RT 落盘\n", tag);
            }
        });
        if (d_ddrt) {   // docs/66 §8:逐行周期 + 路径模式落盘(cursor/search 判据采数)
            std::vector<unsigned long long> h_rt(A_rows);
            std::vector<int> h_md(A_rows);
            CHECK_CUDA(cudaMemcpy(h_rt.data(), d_ddrt, (size_t)A_rows * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_md.data(), d_ddmode, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
            for (int r = 0; r < A_rows; r++)
                if (h_rt[r]) fprintf(dd_f, "%d %llu %d\n", r, h_rt[r], h_md[r]);   // 仅 dense 行(cyc=0 剔)
            fclose(dd_f); dd_f = nullptr;
            dbg("[%s] dd_rowtime → %s\n", tag, dd_path);
        }
    }
    // ── DCEST V3 修正链(study/39 Phase 1b):溢出证书 + 二次 scan + 双 buffer 紧凑拷贝 ──
    // 位置:dense_direct 之后、k7/D5H/compact 之前 —— 此刻 dC 里【只有 dense 行】有数据
    // (k7/D5H 在链后写 exact 基址,免搬;hash 行仍在 tmp)→ 拷贝只覆盖 dense 行。
    // V3 要点:证书先行(梯子核踩邻行后无原地自愈,只能判废);读写分离根除 V2 in-place
    // shift 的行内/跨行两类竞态。
    // ⚠ 插入位必须在 D5H 块之前:D5H 若在链前发射会写 gapped 基址,而拷贝只搬 dense 行
    //   → D5H 行数据滞留旧位 = 丢失(首版落点错误,D5H 默认关所以守门没踩到)。
    if (g_dcest && dense_nr > 0 && dense_rows && !g_cnf) {
        prof("dcest_fixup", [&]() -> void {
            // ① 溢出证书:actual > alloc → 数据已损 → 判废(-1),caller 见负值回退 merge3
            int *d_ovf = decltype(d_ovf)(dev_alloc((2 + 8 + 40) * sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ovf, 0, (2 + 8 + 40) * sizeof(int)));
            dcest_ovf_check_kernel<<<(dense_nr + HASH_BLOCK_STRUCT - 1) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(
                dense_rows, dense_nr, dC_rp, d_row_nnz, d_flop, dA_rp, d_ovf, d_ovf + 2, d_ovf + 10);
            CHECK_CUDA(cudaGetLastError());
            int h_ovf[2 + 8 + 40];
            CHECK_CUDA(cudaMemcpy(h_ovf, d_ovf, sizeof(h_ovf), cudaMemcpyDeviceToHost));
            dev_free(d_ovf);
            dbg("[%s] DCEST 证书: ovf=%d/%d ratio直方图[<0.25:%d <0.5:%d <0.75:%d <1.0:%d <1.25:%d <1.5:%d <2:%d ≥2:%d]\n",
                tag, h_ovf[0], dense_nr, h_ovf[2], h_ovf[3], h_ovf[4], h_ovf[5], h_ovf[6], h_ovf[7], h_ovf[8], h_ovf[9]);
            if (h_ovf[0] > 0)   // 取证:溢出行结构(估计器盲区标本)
                for (int k = 0; k < 8 && k < h_ovf[0]; k++)
                    dbg("[%s] DCEST ovf行#%d: row=%d alloc=%d act=%d flop=%d a_len=%d(est≈alloc×expand/margin)\n",
                        tag, k, h_ovf[10 + 5 * k], h_ovf[11 + 5 * k], h_ovf[12 + 5 * k], h_ovf[13 + 5 * k], h_ovf[14 + 5 * k]);
            if (h_ovf[0] > 0 && !g_dcest_ovf) {
                fprintf(stderr, "[hash] DCEST overflow: %d/%d 行 alloc 不足(margin=%d%%)→ 判废回退 merge3\n",
                        h_ovf[0], dense_nr, g_dcest_margin);
                C_nnz_result = -1;   // 尾部 d2h 按分配期 C_total 有界下载,caller 见 -1 即弃
                return;              // 跳过修正链(k7/D5H/compact 照跑无害:输出已判废)
            }
            // ② 保存 gapped 旧 rp → 二次 scan(exact row_nnz)覆写 dC_rp
            int *dC_rp_old = decltype(dC_rp_old)(dev_alloc((size_t)(A_rows + 1) * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(dC_rp_old, dC_rp, (size_t)(A_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
            if (A_rows <= MAX_BLOCK_THREADS) {
                int b = 1; while (b < A_rows) b <<= 1;
                scan_inclusive_kernel<int><<<1, b, b * sizeof(int)>>>(d_row_nnz, dC_rp + 1, A_rows);
            } else {
                thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                                       thrust::device_ptr<int>(d_row_nnz + A_rows),
                                       thrust::device_ptr<int>(dC_rp + 1));
            }
            CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
            // ③ 双 buffer 紧凑拷贝(仅 dense 行):gapped dC → exact dC2;C 总量收正(d2h 只下 exact)
            C_ci_sz = (size_t)C_nnz_result * sizeof(int);
            C_v_sz  = (size_t)C_nnz_result * sizeof(double);
            C_ci_al = ALIGN8(C_ci_sz);
            C_total = C_rp_al + C_ci_al + C_v_sz;
            void *dC2 = dev_alloc(C_total);
            int    *dC2_ci  = (int *)((char *)dC2 + C_rp_al);
            double *dC2_val = (double *)((char *)dC2 + C_rp_al + C_ci_al);
            dcest_compact_copy_kernel<<<dense_nr, 256>>>(
                dense_rows, dense_nr, dC_rp_old, dC_rp, d_row_nnz, dC_ci, d_val_ip, dC2_ci, dC2_val);
            CHECK_CUDA(cudaGetLastError());
            dev_free(dC_rp_old);
            dev_free(dC);                              // gapped 大 buffer 退役
            dC = dC2; dC_ci = dC2_ci; d_val_ip = dC2_val; d_val = d_val_ip;   // 指针切换(k7/D5H/compact/d2h 消费新布局)
            dbg("[%s] DCEST fixup: C_nnz=%d(exact,紧凑拷贝完成)\n", tag, C_nnz_result);
        });
    }
    // k7 数值:DCEST 链之后发射 → 写 exact 基址(链前写会落在 gapped 位置被拷贝链漏掉)
    if (k7_nr > 0) {
        prof("k7_direct", [&]{
        int *d_k7err2 = decltype(d_k7err2)(dev_alloc(sizeof(int)));
        CHECK_CUDA(cudaMemset(d_k7err2, 0, sizeof(int)));
        k7_numeric_kernel<<<k7_nr, K7_BLOCK, (size_t)K7_SMEM_W * 4>>>(
            dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, A_rows, k7_rows, d_span_lo,
            dC_rp, dC_ci, d_val_ip, nullptr, d_k7err2);
        int herr2 = 0; CHECK_CUDA(cudaMemcpy(&herr2, d_k7err2, sizeof(int), cudaMemcpyDeviceToHost));
        if (herr2) fprintf(stderr, "[hash] k7_numeric 防御带 %d 产品(值门必查)\n", herr2);        dev_free(d_k7err2);
        });
        dbg("[%s] k7_direct: %d 行(BIN_K7)rank 直写 dC(免表/免值原子/免 compact+sort)\n", tag, k7_nr);
    }
        // D5H 数值 pass:bins 1-10 MODE=2 直写 dC(小行 count-sort 有序;大行无序 → 原地 csort)
    if (g_d5h_on) {
        prof("hash_direct", [&]{
            for (int bi = 1; bi <= HASH_LADDER_MAX_BIN; bi++) {
                int n = h_cnt[bi];
                if (n == 0) continue;
                int *rows_ptr = d_sort + h_off[bi];
                int ht = WARP_SIZE << bi;
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                static int g_ex2_d5h = -1;   // HASH_EXTRACT=2(spec 81,与 MODE=0 位点同 env;作用域独立解析)
                if (g_ex2_d5h < 0) { const char *e = getenv("HASH_EXTRACT"); g_ex2_d5h = (e && atoi(e) == 2) ? 1 : 0; }
                if (g_ex2_d5h && ht >= CS2_MIN_HT && ht <= CSORT_HT) smem_flat += 3 * 1024 * sizeof(int);
                {
                    static bool done2 = false;
                    if (!done2) {
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<2,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<2,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));   // 逃生门
                        done2 = true;
                    }
                }
                if (probe_rf_on())
                    hash_spa_kernel<2,1><<<n, HASH_BLOCK, smem_flat>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, 0, dC_rp, dC_ci, d_val, d_rt,
                        nullptr, nullptr, 0, 0, nullptr, 0, nullptr, 0, g_ex2_d5h);
                else
                    hash_spa_kernel<2,0><<<n, HASH_BLOCK, smem_flat>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, 0, dC_rp, dC_ci, d_val, d_rt,
                        nullptr, nullptr, 0, 0, nullptr, 0, nullptr, 0, g_ex2_d5h);
                CHECK_CUDA(cudaGetLastError());
            }
        });
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // MODE=2 理论不再溢出(count 同表已插过);防御复查
        if (overflow) {   // 不可重试行(flop 超表上限)→ 整阵回退(与 legacy 口径一致)
            fprintf(stderr, "[hash] OVERFLOW(D5H): 某 hash 行 distinct > HASH_CAP=%d → 回退 merge\n", HASH_CAP);
            *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
            dev_free(dA); dev_free(d_off); dev_free(d_row_nnz); dev_free(d_overflow);
            dev_free(d_tmp_key); dev_free(d_tmp_val); dev_free(dC_rp); dev_free(dC);
            dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort);
            dev_free(d_est); dev_free(d_flop); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl);
            for (int vb_ = 0; vb_ <= HASH_LADDER_MAX_BIN; vb_++) dev_free(d_gv_bins[vb_]); dev_free(d_gv_off); dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
            dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
            dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
            dev_free(d_smap); dev_free(d_smap_off); dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val); dev_free(dB16);
            return;
        }
    }
    if (g_cnf) {   /* L2b-3/4: late cnnz after direct + underflow recheck */
        CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
        if (A_rows <= MAX_BLOCK_THREADS) {
            int b = 1; while (b < A_rows) b <<= 1;
            scan_inclusive_kernel<int><<<1, b, b * sizeof(int)>>>(d_row_nnz, dC_rp + 1, A_rows);
            /* L2b-5: old-segment squeeze copy (tmp->dC) */
        if (g_cnf && h_big_off > 0 && d_rows_sorted && C_nnz_result >= 0) {
            diter_gapped_copy_kernel<<<(h_big_off + 255) / 256, 256>>>(
                d_rows_sorted, h_big_off, d_off, d_row_nnz, d_dtmp_col, d_dtmp_val, dC_rp, dC_ci, d_val);
        }
    } else {
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                                   thrust::device_ptr<int>(d_row_nnz + A_rows),
                                   thrust::device_ptr<int>(dC_rp + 1));
            /* L2b-5: old-segment squeeze copy (tmp->dC) */
        if (g_cnf && h_big_off > 0 && d_rows_sorted && C_nnz_result >= 0) {
            diter_gapped_copy_kernel<<<(h_big_off + 255) / 256, 256>>>(
                d_rows_sorted, h_big_off, d_off, d_row_nnz, d_dtmp_col, d_dtmp_val, dC_rp, dC_ci, d_val);
        }
    }
        CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        int h_drop = 0;
        if (d_drop) CHECK_CUDA(cudaMemcpy(&h_drop, d_drop, sizeof(int), cudaMemcpyDeviceToHost));
        if (g_cnf && !d_drop) { d_drop = (int*)dev_alloc(sizeof(int)); CHECK_CUDA(cudaMemset(d_drop, 0, sizeof(int))); }
        if ((long long)C_nnz_result > total_est * 2 || h_drop > 0) {
            fprintf(stderr, "[hash] CNT_FUSE INVALID: C=%lld drops=%d — 输出不可信,本刀此阵禁用\n", (long long)C_nnz_result, h_drop);
            C_nnz_result = -1;   /* 判废标记:门/屏见 -1 即弃 */
        }
        /* L2b-5: old-segment squeeze copy (tmp->dC) */
        if (g_cnf && h_big_off > 0 && d_rows_sorted && C_nnz_result >= 0) {
            diter_gapped_copy_kernel<<<(h_big_off + 255) / 256, 256>>>(
                d_rows_sorted, h_big_off, d_off, d_row_nnz, d_dtmp_col, d_dtmp_val, dC_rp, dC_ci, d_val);
        }
    }
    prof("compact+sort", [&]{
        static int g_cs_v2 = -1;   // CSORT_LADDER:默认 v2 开(×5 电池 geomean −6.4%,study/07);=legacy 回旧
        if (g_cs_v2 < 0) { const char *e = getenv("CSORT_LADDER"); g_cs_v2 = (e && !strcmp(e, "legacy")) ? 0 : 1; }
        // COMPACT_FLAT(study/49 K2';10-21 转正默认开):条目级并行拷贝一次性覆盖拷贝桶(bin0/19/21-23),
        // 循环里对应分支跳过;csort/k7/dense 桶路径不变。转正依据:15/15 同会话 A/B 无罪
        // (flat_ab/ab_raw*.log,10-21)+ 预门短路(拷贝桶零行阵免税);=0 回退旧行为
        static int g_cflat = -1;
        if (g_cflat < 0) { const char *e = getenv("COMPACT_FLAT"); g_cflat = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
        static int g_flatb_v = -1;   // FLATB(T4d):bins2-6 hash 行并入条目级拷贝
        if (g_flatb_v < 0) { const char *e = getenv("FLATB"); g_flatb_v = (e && *e && atoi(e) > 0) ? 1 : 0; }
        bool flat_on = g_cflat && !in_place && !(dense_mode || dense_win_mode)
                       && C_nnz_result > 0 && !att;
        // 预门(10-21):拷贝桶行数=0 的阵连 count kernel+D2H 同步都不付(h_cnt 已在 host;
        // ldoor/Long_Coup 类全 hash 梯阵原白付一趟)。零行 → h_ce 必 0 → 门必败,语义不变。
        if (flat_on) {
            long long copy_rows = (long long)h_cnt[0] + h_cnt[BIN_BATCH_S]
                + (long long)h_cnt[BIN_ESC8] + h_cnt[BIN_ESC16] + h_cnt[BIN_ESC32]
                + (BIN_ESC64 < N_BINS ? h_cnt[BIN_ESC64] : 0) + h_cnt[BIN_ESC128]
                + (g_flatb_v ? (long long)h_cnt[2]+h_cnt[3]+h_cnt[4]+h_cnt[5]+h_cnt[6] : 0);   // T4d FLATB
            if (copy_rows == 0) flat_on = false;
        }
        if (flat_on) {   // 条目覆盖门(wb-edu 教训:行比例≠条目比例,浪费随条目数涨):
                        // 拷贝桶条目占比 ≥50% 才放行,否则全量条目线程空转二分
            int *d_ce = (int*)dev_alloc(sizeof(int));
            CHECK_CUDA(cudaMemset(d_ce, 0, sizeof(int)));
            k_count_copy_entries<<<(A_rows + 255) / 256, 256>>>(d_bkid, d_row_nnz, A_rows, d_ce, g_flatb_v);
            int h_ce = 0;
            CHECK_CUDA(cudaMemcpy(&h_ce, d_ce, sizeof(int), cudaMemcpyDeviceToHost));
            if ((long long)h_ce * 2 < (long long)C_nnz_result) flat_on = false;
        }
        if (flat_on) {
            hash_compact_copy_flat_kernel<<<((long long)C_nnz_result + 255) / 256, 256>>>(
                C_nnz_result, A_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val,
                dC_ci, d_val, d_row_ovf, d_bkid, g_flatb_v);
            CHECK_CUDA(cudaGetLastError());
        }
        if (!in_place)
            CHECK_CUDA(cudaMemcpy(dC, dC_rp, C_rp_sz, cudaMemcpyDeviceToDevice));   // row_ptr 落位(~A_rows ints,微秒级,纳入计时)
        for (int bi = 0; bi < ((dense_mode || dense_win_mode) ? 1 : N_BINS); bi++) {
            int n = (dense_mode || dense_win_mode) ? A_rows : h_cnt[bi];
            int *rows_ptr = (dense_mode || dense_win_mode) ? nullptr : (d_sort + h_off[bi]);
            if (dense_mode || dense_win_mode) {
                // 方案5:矩阵级 dense 行已在 dense_direct 直写终态 → copy 整段跳过
                if (dense_nr > 0) break;
                // dense 输出全局有序:identity 行表 + 一趟 warp-batched copy
                int *d_all = decltype(d_all)(dev_alloc((size_t)A_rows * sizeof(int)));
                thrust::sequence(thrust::device_ptr<int>(d_all), thrust::device_ptr<int>(d_all + A_rows));
                launch_copy_warp(A_rows,
                    d_all, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                break;
            }
            if (n == 0) continue;               // 空 bin 免 <<<0,TPB>>> 非法 launch(dense 重构曾吃掉此守卫)
            if (bi == BIN_K7) continue;   // k7_direct 已直写终态(有序+精确);tmp 宽度 0 而 row_nnz 大,csort 必越界读
            if (g_d5h_on && bi >= 1 && bi <= HASH_LADDER_MAX_BIN) {   // D5H:小行已有序直写免善后;大行原地 csort
                if (bi <= ilog2c(CSORT_HT / WARP_SIZE)) continue;
                if (bi <= ilog2c(TABLE_DOUBLING_THR / WARP_SIZE)) launch_csort_ip<CSORT_TPB_A, CSORT_IPT_A>(n, rows_ptr, dC_rp, dC_ci, d_val);
                else         launch_csort_ip<CSORT_TPB_B, CSORT_IPT_B>(n, rows_ptr, dC_rp, dC_ci, d_val);
                CHECK_CUDA(cudaGetLastError());
                continue;
            }
            if (bi == 0) {
                if (flat_on) continue;   // COMPACT_FLAT 已覆盖
                // bin0(est≤64):warp-per-row(输出可达 64/行,warp 协作最稳;线程纂在 est 33-64 行
                // 是 32-64 次串行拷贝 → web-Google +13.6% 教训,线程纂只给 nnz≤32 的 BATCH_S 行)
                launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // TEMP:定位
            } else if (bi == BIN_BATCH_S || (bi >= BIN_ESC8 && bi <= BIN_ESC128)) {
                if (flat_on) continue;   // COMPACT_FLAT 已覆盖
                // BATCH_S 行 est≤32(distinct ≤ 32 上界):串行 ≤32 次 = 0 闲 lane,CTA 3M→94K
                if (g_compact_t)
                    hash_compact_copy_thread_kernel<<<(n + (HASH_BLOCK_STRUCT - 1)) / HASH_BLOCK_STRUCT, HASH_BLOCK_STRUCT>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                else
                    launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == 4 && h45_pilot_on() && !hh_sorted_host()) {
                // 刀E1:bin4 行走 hash_heavy(无序写 tmp)→ csort 善后(CAP 512 紧贴 bin4 表容)
                // Gv3v2(§六十):HH_SORTED=1 时输出已行内有序 → 跳过载体,落下方 copy/FLAT 路(免 counting sort 趟)
                launch_csort<128, 4>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == 5 && h45_pilot_on() && !hh_sorted_host()) {
                // 刀G:bin5 行走 hash_heavy(无序写)→ 两级计数排序善后(spec84 同款;
                // d~200 → CAP 2048 counting O(nnz+CAP) 免 radix padding 税,crankseg bin6/7 已转正同型)
                launch_csort_counting<CSC_TPB, CSC_CAP>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi <= ilog2c(CSORT_HT / WARP_SIZE)) {
                // 小行(ht≤CSORT_HT):accumulate 已 count-sort,这里只 compact_copy(tmp→CSR)
                // Fix#11(审查):warp-per-row(8 行/CTA)替 1CTA/行 —— 万行级 bin 的 CTA 数砍 8×
                // T4d(FLATB):bins2-6 已被 flat 条目级拷贝覆盖,跳过行级拷贝
                if (flat_on && bi >= 2 && bi <= 6) { CHECK_CUDA(cudaGetLastError()); continue; }
                launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            } else if (bi <= ilog2c(TABLE_DOUBLING_THR / WARP_SIZE)) {
                // v37 csort 细梯(study/06 §7):CSORT_LADDER=v2 → bin6 <128,16>(CAP 2048 紧贴表容,
                // 旧 <512,8> 冗余 2× 挤 occupancy)/ bin7 <256,16>(同 CAP 4096,IPT×2 线程减半)。
                // CAP ≥ row_nnz ≤ ht 不变式保持;溢出行 row_nnz=flop 重试值可超 CAP → 只排前 CAP 项
                // 写废,末段 retry 善后整行覆写(同旧语义,csort_fused 本就不带 skip)。
                const bool cs_v2 = g_cs_v2;
                static int g_cs_cnt = -1;   // CSORT_CNT(队列②,已转正默认开):bin6 csort→两级计数排序;
                                            // A级电池×5=F2 −9.6/msc −1.0/c-64 +2.3(噪声带)/crankseg +0.1 + 值级有序 4/4;
                                            // =0 回 cub radix 逃生门
                if (g_cs_cnt < 0) { const char *e = getenv("CSORT_CNT"); g_cs_cnt = (e && *e && atoi(e) == 0) ? 0 : 1; }
                const int mx = cs_v2 ? (bi == 6 ? csort_fused_max_cols(CSORT_TPB_C, CSORT_IPT_C)
                                                : csort_fused_max_cols(CSORT_TPB_D, CSORT_IPT_D))
                                     : csort_fused_max_cols(CSORT_TPB_A, CSORT_IPT_A);   // 融合键:pos 位 → 列位宽约束
                if (g_cs_cnt && bi == 6) {
                    // counting 变体无融合键位约束,免 valid_bits/A_cols 门
                    launch_csort_counting<CSC_TPB, CSC_CAP>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
                } else if (g_cs_cnt && bi == 7 && !hh_sorted_host()) {
                    // spec84:bin7 扩展——radix <512,8> REG:79 = 1 块/SM(25% warps) vs counting 3 块(75%),
                    // 占用+指令双优;SMEM 60KB(helper 条件 opt-in);crankseg bin7 人口主判
                    launch_csort_counting<512, 4096>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
                } else if (bi == 7 && hh_sorted_host()) {
                    // §六十四:HH_SORTED 下 bin7/8 hash_heavy 输出已行内有序 → 免排序载体直拷
                    // (nsys 定谳:有序输入撞 counting<512,4096> 0.59→3.78ms 病理;TSOPF/c-64/pre2 回退元凶)
                    launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                } else if (A_cols < mx && !getenv("NOCSF")) {
                    if (cs_v2 && bi == 6)  launch_csort_fused<CSORT_TPB_C, CSORT_IPT_C>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                    else if (cs_v2)        launch_csort_fused<CSORT_TPB_D, CSORT_IPT_D>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                    else                   launch_csort_fused<CSORT_TPB_A, CSORT_IPT_A>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                } else {
                    if (cs_v2 && bi == 6)  launch_csort<CSORT_TPB_C, CSORT_IPT_C>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                    else if (cs_v2)        launch_csort<CSORT_TPB_D, CSORT_IPT_D>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                    else                   launch_csort<CSORT_TPB_A, CSORT_IPT_A>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                }
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == 8 && hh_sorted_host()) {
                launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);   // §六十四:bin8 有序直拷(同 bin7)
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == 9 && hh_sorted_host() && h69_route_on()) {
                // 刀①'(study/59):H69 开时 bin9 行全部来自 HH(16253)+有序直写 → 免 SEGRTS
                // 排序阶梯直拷(A/B 实证:HH acc −41~−58% 但 SEGRTS compact 天价 42ms 级吃光)
                launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi <= 9) {
                static int g_segrts = -1;   // v31-①':bin≥7 csort O(n²) → cub 分段基数排序(靶纸:pre2 排序税 26ms=57%)
                if (g_segrts < 0) { const char *e = getenv("SEGRTS"); g_segrts = (e && *e) ? atoi(e) : 0; }
                // 刀14 终版:bin8 counting sort 前置(fused <256,64> 占 98% 排序时间——ncu 实锤;
                // radix 排 8192 槽含 66% padding=3.2×浪费;counting O(nnz+CAP) 免此税)
                static int g_k14cs2 = -1;
                if (g_k14cs2 < 0) { const char *e = getenv("K14CS"); g_k14cs2 = (e && *e && atoi(e) == 0) ? 0 : 1; }
                if (g_k14cs2 && bi == 8) {
                    // 刀16b:真 d 劈双 CAP(实验 env CS16B;scratch=2 计数器+2×n 行表)
                    int *cs_scr = decltype(cs_scr)(dev_alloc((size_t)(2 + 2 * n) * sizeof(int)));
                    launch_cs8192_part(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, cs_scr, 4096);
                    dev_free(cs_scr);
                    CHECK_CUDA(cudaGetLastError());
                } else if (g_k14cs2 && bi == 9) {
                    // 刀14 扩展 + P0 修复(09-30):bin9 counting CAP 8192 对 d>8192 行截断写废
                    // (in-2004 431 行实锤:非 ovf→retry 不接管→行尾 stale 垃圾+无序;off-vs-off 自对拍定位)。
                    // 劈分:d≤8192 → counting;d>8192 → csort_fused<256,64>(CAP 16384 覆盖全人群)。
                    {
                        int *s9 = decltype(s9)(dev_alloc((size_t)(2 + 2 * n) * sizeof(int)));
                        CHECK_CUDA(cudaMemsetAsync(s9, 0, 2 * sizeof(int)));
                        cs_part_kernel<<<(n + 255) / 256, 256>>>(rows_ptr, n, d_row_nnz, 8192, s9, s9 + 1, s9 + 2, s9 + 2 + n);
                        int h9[2];
                        CHECK_CUDA(cudaMemcpyAsync(h9, s9, 2 * sizeof(int), cudaMemcpyDeviceToHost));
                        CHECK_CUDA(cudaDeviceSynchronize());
                        static int g_c9 = -1; if (g_c9 < 0) { const char *e3 = getenv("CS16C"); g_c9 = (e3 && *e3 && atoi(e3) == 0) ? 0 : 1; }   // 刀16 翻默认 ON
                        if (h9[0] > 0) { if (g_c9) launch_csort_counting<512, 8192, 4096>(h9[0], s9 + 2, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);   // 刀16c:NB=4096
                                         else      launch_cs8192(h9[0], s9 + 2, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val); }
                        if (h9[1] > 0) {   // P0:d>8192 行走 CAP 16384 老路径(排序覆盖,免截断)
                            if (A_cols < csort_fused_max_cols(CSORT_TPB_B, CSORT_IPT_B) && !getenv("NOCSF"))
                                launch_csort_fused<CSORT_TPB_B, CSORT_IPT_B>(h9[1], s9 + 2 + n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                            else
                                launch_csort<CSORT_TPB_B, CSORT_IPT_B>(h9[1], s9 + 2 + n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                        }
                        dev_free(s9);
                    }
                    CHECK_CUDA(cudaGetLastError());
                } else
                if (bi >= 7 && g_segrts) {
                    launch_bin_segrts(n, rows_ptr, d_off, d_row_nnz, dC_rp, dC_ci, d_val,
                                      d_tmp_key, d_tmp_val, d_row_ovf);
                    CHECK_CUDA(cudaGetLastError());
                } else if (A_cols < csort_fused_max_cols(CSORT_TPB_B, CSORT_IPT_B) && !getenv("NOCSF")) {   // pos 14 位(CAP=16384)
                    // v2 不动 bin8-9:<512,32> 实测 157 regs×512T = 80384 > 65536 块限额 launch 不可
                    // (ptxas 定罪);旧 <256,64> 229×256=58624 贴线可跑,且 bin8-9 行稀无证据
                    launch_csort_fused<CSORT_TPB_B, CSORT_IPT_B>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                } else {
                    // 刀14b:bin8 的 <256,64> 寄存器爆(IPT=64 → 192+ regs → 1块/SM=12.5%占用)
                    // → <512,16>(CAP 同 8192,regs/thread 48+overhead → 2块/SM) ;
                    // K14B=0 逃生门回旧配置
                    // 刀14:bin8 从 cub radix(<256,64> 或 <512,16>)改 counting sort(同 bin7 spec84 转正)
                    // 根因:radix 排 CAP 槽含 ~66% padding → 3.2× 浪费 ≈ 系统排序差距 3.4×;counting O(nnz+CAP) 无此税
                    static int g_k14cs = -1;
                    if (g_k14cs < 0) { const char *e = getenv("K14CS"); g_k14cs = (e && *e && atoi(e) == 0) ? 0 : 1; }
                    if (g_k14cs && bi == 8) launch_cs8192(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
                    else                   launch_csort<CSORT_TPB_B, CSORT_IPT_B>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                }
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == BIN_HEAVY) {
                // heavy:accumulate 内已分段排序(compact 读 scratch)
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_scr_key, d_scr_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // TEMP:定位
            } else if (bi >= BIN_SSPAN0 && bi <= BIN_SSPAN0 + 3) {
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == BIN_DITER) {
                // dense-iter:窗口升序 = 行内天然有序,只 copy。方案5:direct 已直写 → 跳过
                if (dense_nr == 0 && h_d3_n == 0)   // D3 直写免 tmp copy(抽空≠兜底)
                    hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else {
                // ultra(行≤32)和 subwarp8(bin12):kernel 内已插入排序 → warp copy
                launch_copy_warp(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (d_retry_n > 0) {
            if (d_retry_needs_sort) {
                // docs/54:retry 免排序路径 —— hash_global 乱序输出 → csort 善后
                if (d_retry_n > 0 && d_retry_n <= A_rows) {
                    // 用适中的 csort 配置(retry 行的 row_nnz 通常 ≤ 16384)
                    launch_csort<CSORT_TPB_B, CSORT_IPT_B>(d_retry_n, d_retry_rows, d_retry_off, d_row_nnz, dC_rp,
                                          (unsigned long long*)d_retry_key, (double*)d_retry_val,
                                          dC_ci, d_val, nullptr);
                    CHECK_CUDA(cudaGetLastError());
                }
            } else {
                // 原路径(已排序,纯 copy)
                if (d_retry_n > 0 && d_retry_n <= A_rows)
                    hash_compact_copy_kernel<<<d_retry_n, 256>>>(
                        d_retry_rows, d_retry_n, d_retry_off, d_row_nnz, dC_rp,
                        d_retry_key, d_retry_val, dC_ci, d_val, nullptr);
            }
        }
        CHECK_CUDA(cudaGetLastError());
    });
    tmpscan("post-compact");
    outscan("post-compact");
    // docs/21 Fix0:重试中间缓冲(compact 消费完)立即释放 —— 此前从不释放,高 flop 阵每调用
    // 泄漏 r_slots×32B + rt_total×12B,warmup 多轮累加直接 OOM(rajat 类 DNF 帮凶)
    if (!s_r2_arena_active) { dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); }   // R1d:持久 arena 不逐轮 free
    dev_free(rk2); dev_free(rv2); dev_free(rct);
    dev_free(d_rht); dev_free(d_rtab);
    if (!s_r2_arena_active) { dev_free(d_rslot); dev_free(d_roff); }
    dev_free(d_rsb); dev_free(d_rse);

// (hash_check_sorted 校验已移除:DBG 计时必需但校验使命完成,免 43% 墙钟税)

    // Stage 7: 连续 dC 已就位(compact+sort 直写 col/val + row_ptr 已落位)→ 直接 D2H 前 C_total 字节 = packed [rp|ci|val]。免 pack。
    //          in_place 模式:col/val 在 tmp 里 → 3 段拷进同一 pinned 布局(rq: rp 段从 dC_rp)。

    void *C_buffer = nullptr;
    if (d_rt) {   // docs/66:逐行周期 + 特征(flop/est/span/ovf)落盘(管线公共尾,全路径覆盖);
        // ovf 行已由 retry 重做,t_hash 失真 → 分析时剔除;cyc=0 = 该行未走 hash_spa(dense/ultra/全局表路径)
        std::vector<unsigned long long> h_rt(A_rows);
        std::vector<int> h_fl(A_rows), h_es(A_rows), h_sp(A_rows), h_ov(A_rows);
        CHECK_CUDA(cudaMemcpy(h_rt.data(), d_rt, (size_t)A_rows * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_fl.data(), d_flop, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_es.data(), d_est_save, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_sp.data(), d_span_len, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_ov.data(), d_row_ovf, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
        const int *h_rp = (const int*)A_buffer;
        std::vector<int> h_crp(A_rows + 1);   // 行级口径定谳(09-30):第8列=C行nnz=真distinct d
        CHECK_CUDA(cudaMemcpy(h_crp.data(), dC_rp, (size_t)(A_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> h_bk(A_rows);        // 第9列=bin id:窗分类用真bin(kid≤5=ht≤1024=kernel内排序),免律重建
        CHECK_CUDA(cudaMemcpy(h_bk.data(), d_bkid, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToHost));
        for (int r = 0; r < A_rows; r++)
            fprintf(hrt_f, "%d %d %d %d %d %d %llu %d %d\n", r, h_rp[r + 1] - h_rp[r], h_fl[r], h_es[r], h_sp[r], h_ov[r], h_rt[r], h_crp[r + 1] - h_crp[r], h_bk[r]);
        fclose(hrt_f); hrt_f = nullptr;
        dbg("[%s] rowtime(%d 行)→ %s\n", tag, A_rows, hrt_path);
        dev_free(d_est_save);
    }
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total));
    prof("d2h", [&]{
        if (in_place) {
            char *hb = (char*)C_buffer;
            CHECK_CUDA(cudaMemcpyAsync(hb, dC_rp, C_rp_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyAsync(hb + C_rp_al, d_tmp_key, C_ci_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyAsync(hb + C_rp_al + C_ci_al, d_tmp_val, C_v_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaMemcpy(C_buffer, dC, C_total, cudaMemcpyDeviceToHost));
        }
    });

    *C_buffer_out = C_buffer;
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = C_nnz_result;

    dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
    dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
    dev_free(dC_rp); dev_free(dC);   // dC_ci/d_val 是 dC 的偏移别名,随 dC 一起释放
    dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val); dev_free(dB16);   // ATT 的 Aᵀ(AA 时为 null,no-op)
    dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
    dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
    dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
    // 成功路径逐行数组泄漏修复(2026-08-27):d_bkid/d_sort/d_est/d_flop/d_span_*/d_maxbl 此前只在
    // 早退路径释放,成功路径漏 ~28B×A_rows/调用(USE_DEV_POOL 默认关 = cudaMalloc 模式;whale 阵
    // warmup+bench 多轮累加 GB 级 —— c-73 峰值 ~62GB 的隐性推手之一)。尾部释放与 d2h 后既有
    // 12 buffer 同区域,不进 hash-prof 计时相位。
    dev_free(d_hybrid_val);   // Fix#6(审查):bin0 hybrid 值池从不释放(n×1KB/调用)
    dev_free(d_bkid); dev_free(d_sort); dev_free(d_est); dev_free(d_flop);
    dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); for (int vb_ = 0; vb_ <= HASH_LADDER_MAX_BIN; vb_++) dev_free(d_gv_bins[vb_]); dev_free(d_gv_off);
    dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos);
    dev_free(d_smap); dev_free(d_smap_off);   // Phase B v2 全局 cursor 区
    { cudaError_t pe = cudaGetLastError(); if (pe != cudaSuccess) fprintf(stderr, "[probe] frees 后: %s\n", cudaGetErrorString(pe)); }                    // Fix#6(审查)
}

void spgemm_self_product_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                              void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    hash_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/false, C_buffer_out, C_rows, C_cols, C_nnz);
}

// C = A·Aᵀ 上三角(j≥i):AA hash SPA 的忠实拷贝,B=Aᵀ(CSC)+ j≥i 过滤。复用 MinHash sizing/binning/compact_sort。
void spgemm_att_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    hash_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/true, C_buffer_out, C_rows, C_cols, C_nnz);
}

#endif   // P1 G2 结束(hash_product 全域;TU-61 到此为止)
