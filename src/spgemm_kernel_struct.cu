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

// Phase 1b:numeric 直写。单窗行(span ≤ win_cols)在位图 popcount 前缀上直接发射。
// SMEM: val[win_cols] + bitmap[words] + prefix[words]。多窗行走逐窗模式(同 Phase 1a 结构)。
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
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int lo0 = span_lo[i], span = span_len[i];
    int rs = A_rp[i], re = A_rp[i + 1];
    extern __shared__ __align__(8) unsigned char smem[];
    double *dval = (double *)smem;                                        // [win_cols]
    unsigned int *bmp = (unsigned int *)(smem + (size_t)win_cols * 8);   // [words]
    unsigned int *pref = bmp + ((win_cols + 31) >> 5);                    // [words]
    __shared__ int wcnt[32];
    int base = exact_rp[i];
    int out = 0;
    int nw_win = (span + win_cols - 1) / win_cols;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * win_cols;
        int c1 = min(c0 + win_cols, lo0 + span);
        int w = c1 - c0;
        int words = (w + 31) >> 5;
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; }
        for (int j4 = tid; j4 < words; j4 += blockDim.x) bmp[j4 >> 2] = 0;
        __syncthreads();
        for (int p = rs; p < re; p++) {
            int k = A_ci[p];
            double a = A_val[p];
            int ks = B_rp[k], ke = B_rp[k + 1];
            int from = (nw_win == 1) ? ks : st_lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : st_lower_bound(B_ci, from, ke, c1);
            for (int q = from + tid; q < to; q += blockDim.x) {
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
        // 位图字级 popcount 前缀(替代 warp0 串行 5400:words=win/32,结构上少 32× 且 warp 级并行)
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
        // 发射:扫描位图字,提取置位位(col 升序天然有序)
        for (int wo = warp; wo < words; wo += (int)(blockDim.x >> 5)) {
            unsigned bm = bmp[wo];
            while (bm) {
                int b = __ffs(bm) - 1;
                bm &= bm - 1;
                int j = (wo << 5) + b;
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                out_col[base + pref[wo] + __popc(bmp[wo] & ((1u << b) - 1))] = col;
                out_val[base + pref[wo] + __popc(bmp[wo] & ((1u << b) - 1))] = dval[j];
            }
        }
        __syncthreads();
        out += (words > 0) ? pref[words - 1] + __popc(bmp[words - 1]) : 0;
    }
    if (tid == 0) row_nnz[i] = out;
}
