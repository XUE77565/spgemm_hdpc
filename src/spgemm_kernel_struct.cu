// ============================================================================
// spgemm_kernel_struct.cu —— 结构化 dense 内核(docs/72 Phase 1,任务 #15)
//
// 与旧 hash_dense_window/direct/count 系列的区别(全部指向"成本∝字节"):
//   窗宽   :dense_cols_max(SMEM 档) 推导(旧:固定 5400 拟合)
//   SMEM   :val[cols] + bitmap[cols/32] + prefix[cols/32],由档位公式推出
//   发射   :位图字级并行 popcount 前缀(旧:warp0 串行扫全窗)
//   单窗行 :span ≤ 档位 → 无窗(免每窗 per-k lower_bound 重搜 = Ga 族窗税的根)
// 参数全部来自 structural_params.cuh;本文件【不得】出现裸数值阈值。
// ============================================================================

#include "spgemm.h"
#include "structural_params.cuh"
#include <cstdio>

#define CHECK_CUDA_ST(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(EXIT_FAILURE); } \
} while (0)

__device__ __forceinline__ int st_lower_bound(const int *a, int lo, int hi, int v) {
    while (lo < hi) { int mid = lo + (hi - lo) / 2; if (a[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
}

// Phase 1a:symbolic 计数。一行一块;span ≤ 档 → 单窗;超档 → 多窗(窗宽 = 档,结构量)。
// SMEM 布局:bitmap[words](计数只需位图)。多窗时逐窗累计。
__global__ void dense_struct_count_kernel(
    const int *A_rp, const int *A_ci,
    const int *B_rp, const int *B_ci,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols, int *row_nnz)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    extern __shared__ __align__(8) unsigned int bmp[];
    int lo0 = span_lo[i];
    int span = span_len[i];
    int nw_win = (span + win_cols - 1) / win_cols;
    __shared__ int wcnt[32];
    int total = 0;
    int rs = A_rp[i], re = A_rp[i + 1];
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * win_cols;
        int c1 = min(c0 + win_cols, lo0 + span);
        int words = (c1 - c0 + 31) >> 5;
        for (int w = tid; w < words; w += blockDim.x) bmp[w] = 0;
        __syncthreads();
        for (int p = rs; p < re; p++) {
            int k = A_ci[p];
            int ks = B_rp[k], ke = B_rp[k + 1];
            int from = (nw_win == 1) ? ks : st_lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : st_lower_bound(B_ci, from, ke, c1);
            for (int q = from + tid; q < to; q += blockDim.x) {
                int col = B_ci[q];
                if (upper_tri && col < i) continue;
                unsigned m = 1u << ((col - c0) & 31);
                int w = (col - c0) >> 5;
                if ((bmp[w] & m) == 0) atomicOr(&bmp[w], m);
            }
        }
        __syncthreads();
        int cnt = 0;
        int w4 = words & ~3;
        const uint4 *b4 = (const uint4 *)bmp;
        for (int j4 = tid; j4 < (w4 >> 2); j4 += blockDim.x) {
            uint4 v = b4[j4];
            cnt += __popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w);
        }
        for (int w = w4 + tid; w < words; w += blockDim.x) cnt += __popc(bmp[w]);
        for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, off);
        if ((tid & 31) == 0) wcnt[tid >> 5] = cnt;
        __syncthreads();
        if (tid == 0) {
            int s = 0;
            for (int x = 0; x < (int)(blockDim.x >> 5); x++) s += wcnt[x];
            total += s;
        }
        __syncthreads();
    }
    if (tid == 0) row_nnz[i] = total;
}

// §26 采数:struct numeric 逐行周期(host 经 cudaMemcpyToSymbol 装 ptr;0 = 关)。
// 与 DD_ROWTIME 同语义(clock64 对 DVFS 免疫)—— a_len 维成本律的拟合原料。
__device__ unsigned long long *g_dsn_rt = nullptr;

// Phase 1b-v2:numeric 直写(G-group 乘积 + 位图前缀发射)。继承旧 kernel 的 G-group(降争用)
// + Phase 1b 的位图字级前缀(替 warp0 串行)。全部常数从 structural_params.cuh。
// G 推导:blockDim×a_len 的负载均衡 — G ≈ 2^round(log2(avg_B×a_len/(blockDim×avg_B+a_len×avg_B)))
// 简化:G = 2^clamp(floor(log2(avg_B)), 1, log2(blockDim))(与 local_load_balance 同语义,注释推导)
template<int TPB>
__global__ void dense_struct_numeric_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    unsigned long long rt0 = g_dsn_rt ? clock64() : 0;   // §26 采数
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int lo0 = span_lo[i], span = span_len[i];
    int rs = A_rp[i], re = A_rp[i + 1];
    int a_len = re - rs;

    // G-group:G 线程共担一条 k 链(乘积分散在不同 B 元素 → 不同列 → 争用摊薄)
    // 推导:avg_B = flop/a_len;G = min(warp_size, 2^floor(log2(avg_B)))(结构上限 = 1 warp)
    //        num_groups = blockDim/G(每 k 链一组;G 大 → 组少 → 并行 k 链少但每链快)
    int G = WARP_SIZE;  // 默认 = 1 warp(争用最低的并行单元;≤ warp 免跨 warp 同步)
    {
        // 从 span/flop/a_len 推导 avg_B(免 row_flop 依赖)
        int est_flop = span > 0 ? a_len * (span / 32 > 0 ? span / 32 : 1) : a_len;
        int avg_B = a_len > 0 ? est_flop / a_len : 1;
        if (avg_B > 1) {
            int logG = 31 - __clz((unsigned)avg_B);
            if (logG < 1) logG = 1;
            if (logG > ilog2c(WARP_SIZE)) logG = ilog2c(WARP_SIZE);   // 上限 = log2(WARP_SIZE)(硬件:G-group ≤ warp 才免跨 warp 同步)
            G = 1 << logG;
        }
    }
    int num_groups = blockDim.x / G;
    int my_group = tid / G;
    int my_id = tid % G;

    extern __shared__ __align__(8) unsigned char smem[];
    double *dval = (double *)smem;                                        // [win_cols]
    unsigned int *bmp = (unsigned int *)(smem + (size_t)win_cols * 8);   // [words]
    unsigned int *pref = bmp + ((win_cols + 31) >> 5);                    // [words]
    int base = exact_rp[i];
    int out = 0;
    int nw_win = (span + win_cols - 1) / win_cols;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * win_cols;
        int c1 = min(c0 + win_cols, lo0 + span);
        int w = c1 - c0;
        int words = (w + 31) >> 5;
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; }
        for (int j4 = tid; j4 < words; j4 += blockDim.x) bmp[j4] = 0;
        __syncthreads();
        // G-group 乘积:每 group 拿不同 k 链,组内 G 线程分散 B 元素
        for (int p = rs + my_group; p < re; p += num_groups) {
            int k = A_ci[p];
            double a = A_val[p];
            int ks = B_rp[k], ke = B_rp[k + 1];
            int from = (nw_win == 1) ? ks : st_lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : st_lower_bound(B_ci, from, ke, c1);
            for (int q = from + my_id; q < to; q += G) {
                int col = B_ci[q];
                if (upper_tri && col < i) continue;
                int j = col - c0;
                atomicAdd(&dval[j], a * B_val[q]);
                unsigned m = 1u << (j & 31);
                int wo = j >> 5;
                if ((bmp[wo] & m) == 0) atomicOr(&bmp[wo], m);
            }
        }
        __syncthreads();
        // 位图字级 popcount 前缀(warp0 分段 + 段基址 scan;words=win/32 → 比 5400 项少 32×)
        if (warp == 0) {
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
        }
        __syncthreads();
        // 发射:位图字扫描,置位位升序 = 天然有序
        for (int wo = warp; wo < words; wo += (int)(blockDim.x >> 5)) {
            unsigned bm = bmp[wo];
            while (bm) {
                int b = __ffs(bm) - 1;
                bm &= bm - 1;
                int j = (wo << 5) + b;
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                int pos = base + pref[wo] + __popc(bmp[wo] & ((1u << b) - 1));
                out_col[pos] = col;
                out_val[pos] = dval[j];
            }
        }
        __syncthreads();
        out += (words > 0) ? pref[words - 1] + __popc(bmp[words - 1]) : 0;
    }
    if (tid == 0) { row_nnz[i] = out; if (g_dsn_rt) g_dsn_rt[i] = clock64() - rt0; }   // §26 逐行周期
}

// §22 档位梯显式实例(跨编译单元模板链接所需;档宽 = dense_tier_block)
template __global__ void dense_struct_numeric_kernel<64>(const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, const int*, int, const int*, int*, double*, int*);
template __global__ void dense_struct_numeric_kernel<128>(const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, const int*, int, const int*, int*, double*, int*);
template __global__ void dense_struct_numeric_kernel<256>(const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, const int*, int, const int*, int*, double*, int*);
template __global__ void dense_struct_numeric_kernel<512>(const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, const int*, int, const int*, int*, double*, int*);
template __global__ void dense_struct_numeric_kernel<1024>(const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, const int*, int, const int*, int*, double*, int*);

// §26 采数:host 侧装/卸逐行周期指针(device symbol 跨 TU 不可见 → 本 TU 内代理)
void dsn_rt_attach(unsigned long long *p) { cudaMemcpyToSymbol(g_dsn_rt, &p, sizeof(void*)); }
