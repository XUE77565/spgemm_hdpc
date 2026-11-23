// hh_repro4.cu — 刀三·想法二(位图标记-累积两遍式)微复现器
// 用法: ./hh_repro4 [span blockDim trials]
// 机制:Pass1 atomicOr 标记列占用(~1cyc/乘积) → 前缀 → Pass2 直寻累积(~3cyc/乘积)
// vs 现状 CAS 探测(dup×4-6cyc)。判据:输出恰为升序键值对。
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// 模拟一行:给定 B 行的列集合(有重复),生成位图标记+前缀+直寻累积
// 与生产同构:每线程处理部分乘积,位图在 SMEM
template <int MAX_SPAN>
__global__ void hh_repro4_kernel(
    const int *d_products_col,     // F 个乘积的列号(有重复)
    const double *d_products_val,  // F 个乘积的值
    int F, int col_lo,              // 行的列域 [col_lo, col_lo+span)
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, long long base)
{
    extern __shared__ unsigned bm[];   // [MAX_SPAN/32] 位图
    extern __shared__ double sval[];   // 紧接位图后:值区(按位图前缀分配)
    // 布局:bm[0..MAX_SPAN/32) 然后 sval[0..MAX_SPAN)
    // 实际上 sval 直接用 tmp_val(全局)避免 SMEM 超限——微复现简化
    const int tid = threadIdx.x;
    const int NW = MAX_SPAN / 32;

    // Pass 1:标记
    for (int w = tid; w < NW; w += blockDim.x) bm[w] = 0u;
    __syncthreads();
    for (int p = tid; p < F; p += blockDim.x) {
        const int b = d_products_col[p] - col_lo;
        atomicOr(&bm[b >> 5], 1u << (b & 31));
    }
    __syncthreads();

    // 前缀扫描(单线程简化,生产用组前缀)
    __shared__ int pref[NW > 256 ? 256 : NW];
    if (tid == 0) {
        int acc = 0;
        for (int w = 0; w < NW; w++) { const int t = __popc(bm[w]); pref[w] = acc; acc += t; }
        row_nnz[0] = acc;
    }
    __syncthreads();

    // Pass 2:累积(atomicAdd 到 tmp_val[base + rank])
    // rank = pref[word] + popc(bm[word] & below_mask)
    for (int p = tid; p < F; p += blockDim.x) {
        const int b = d_products_col[p] - col_lo;
        const int w = b >> 5;
        const unsigned bit = 1u << (b & 31);
        const int rank = pref[w] + __popc(bm[w] & (bit - 1));
        // 首写直写,后续 atomicAdd(简化:全 atomicAdd,生产可首写判 old==0)
        atomicAdd(&tmp_val[base + rank], d_products_val[p]);
        tmp_key[base + rank] = (unsigned long long)(col_lo + b);  // 键(多线程写同值无害)
    }
}

int main(int argc, char **argv)
{
    constexpr int MAX_SPAN = 8192;
    const int BD = argc > 2 ? atoi(argv[2]) : 256;
    const int TRIALS = argc > 3 ? atoi(argv[3]) : 500;
    std::mt19937 rng(42);
    int bad = 0;
    for (int t = 0; t < TRIALS; t++) {
        // 生成:span 随机,F = span × dup_factor
        const int span = 1 + rng() % MAX_SPAN;
        const double dup_f = 1.0 + (rng() % 100) / 10.0;  // 1.0-10.0
        const int F = std::max(1, (int)(span * dup_f));
        const int col_lo = rng() % 1000;
        std::uniform_int_distribution<int> cd(0, span - 1);
        std::normal_distribution<double> vd(1.0, 0.5);
        std::vector<int> cols(F); std::vector<double> vals(F);
        for (int p = 0; p < F; p++) { cols[p] = col_lo + cd(rng); vals[p] = vd(rng); }
        // 真值
        std::vector<int> uniq(cols);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::vector<double> truth(uniq.size(), 0.0);
        for (int p = 0; p < F; p++) {
            int idx = std::lower_bound(uniq.begin(), uniq.end(), cols[p]) - uniq.begin();
            truth[idx] += vals[p];
        }
        // GPU
        int *d_cols; double *d_vals; unsigned long long *d_key; double *d_val; int *d_nnz;
        const int NW = (MAX_SPAN + 31) / 32;
        CK(cudaMalloc(&d_cols, F * 4)); CK(cudaMalloc(&d_vals, F * 8));
        CK(cudaMalloc(&d_key, uniq.size() * 8)); CK(cudaMalloc(&d_val, uniq.size() * 8));
        CK(cudaMalloc(&d_nnz, 4));
        CK(cudaMemcpy(d_cols, cols.data(), F * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_vals, vals.data(), F * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_val, 0, uniq.size() * 8));
        CK(cudaMemset(d_key, 0xAA, uniq.size() * 8));
        const size_t smem = (size_t)NW * 4;  // 位图(值区走全局,微复现简化)
        hh_repro4_kernel<MAX_SPAN><<<1, BD, smem>>>(d_cols, d_vals, F, col_lo, d_key, d_val, d_nnz, 0);
        CK(cudaGetLastError());
        int nnz; CK(cudaMemcpy(&nnz, d_nnz, 4, cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> hk(uniq.size());
        std::vector<double> hv(uniq.size());
        CK(cudaMemcpy(hk.data(), d_key, uniq.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(), d_val, uniq.size() * 8, cudaMemcpyDeviceToHost));
        cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_key); cudaFree(d_val); cudaFree(d_nnz);
        // 验:序(nnz 唯一升序)+ 值
        bool ok = (nnz == (int)uniq.size());
        if (ok) for (int i = 0; i < nnz && ok; i++) {
            if ((int)(hk[i] & 0xffffffffu) != uniq[i]) { ok = false; break; }
            if (fabs(hv[i] - truth[i]) > 1e-9) { ok = false; break; }
        }
        if (!ok) { printf("  TRIAL %d: span=%d F=%d nnz=%d/%zu FAIL\n", t, span, F, nnz, uniq.size()); bad++; }
    }
    printf("%s: MAX_SPAN=%d BD=%d trials=%d bad=%d\n", bad ? "FAIL" : "ALL-PASS", MAX_SPAN, BD, TRIALS, bad);
    return bad ? 1 : 0;
}
