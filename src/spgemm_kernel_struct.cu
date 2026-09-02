// ============================================================================
// spgemm_kernel_struct.cu —— 结构化 dense 内核(docs/72 Phase 1,任务 #15)
//
// 与旧 hash_dense_window/direct/count 系列的区别(全部指向"成本∝字节"):
//   窗宽   :dense_cols_max(SMEM 档) 推导(旧:固定 5400 拟合)
//   SMEM   :val[cols] + bitmap[cols/32] + prefix[cols/32],由档位公式推出
//   发射   :位图字级并行 popcount 前缀(旧:warp0 串行扫全窗)
//   单窗行 :span ≤ 档位 → 无窗(免每窗 per-k lower_bound 重搜 = Ga 族窗税的根)
// 参数全部来自 structural_params.cuh;本文件【不得】出现裸数值阈值。
// 状态:Phase 1a(计数/symbolic 版);Phase 1b(numeric 直写)随后。
// ============================================================================
#include "spgemm.h"
#include "structural_params.cuh"
#include <cstdio>

#define CHECK_CUDA_ST(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(EXIT_FAILURE); } \
} while (0)

__device__ __forceinline__ int lower_bound(const int *a, int lo, int hi, int v) {
    while (lo < hi) { int mid = lo + (hi - lo) / 2; if (a[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
}

// Phase 1a:symbolic 计数。一行一块;span ≤ 档 → 单窗;超档 → 多窗(窗宽 = 档,结构量)。
// SMEM 布局:bitmap[words](计数只需位图)。多窗时逐窗累计。
// 每窗内乘积源:per-k lower_bound 到 [c0,c1)(窗机制与旧内核同,但窗宽由档位推导;
// 单窗行(多数)天然免重搜 —— 这是 Ga 族 21 窗税的结构性消除)。
__global__ void dense_struct_count_kernel(
    const int *A_rp, const int *A_ci,
    const int *B_rp, const int *B_ci,
    int upper_tri, int A_rows,
    const int *bucket_rows, const int *span_lo, const int *span_len,
    int win_cols,                       // = dense_cols_max(档),host 由公式传入
    int *row_nnz)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    extern __shared__ __align__(8) unsigned int bmp[];   // [win_cols/32] 字
    int lo0 = span_lo[i];
    int span = span_len[i];
    int nw_win = (span + win_cols - 1) / win_cols;      // 窗数 = 结构量
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
            // 单窗(整 span 一窗)时 [c0,c1) ⊇ 乘积列 → 直接迭代,免 lower_bound
            int from = (nw_win == 1) ? ks : lower_bound(B_ci, ks, ke, c0);
            int to   = (nw_win == 1) ? ke : lower_bound(B_ci, from, ke, c1);
            for (int q = from + tid; q < to; q += blockDim.x) {
                int col = B_ci[q];
                if (upper_tri && col < i) continue;
                unsigned m = 1u << ((col - c0) & 31);
                int w = (col - c0) >> 5;
                if ((bmp[w] & m) == 0) atomicOr(&bmp[w], m);   // QUERY 先读后 or(降争用)
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

