// hh_repro2.cu — Gv4 列域位图有序直写 微复现器(study/68 修复方案的先行验证)
// 用法: ./hh_repro2 [HT_P blockDim BM_WORDS ntrials]
// 与生产内核(hash_heavy_kernel)同构:清位图→置位(atomicOr)→组前缀(32词/组,
// 计数→独占前缀)→位次 = 组前缀 + 组内词 popc 和 + 词内低位 popc →直写。
// 判据(比 v1 严):row_nnz==k;tmp[0..k) 恰为键的升序(双射+有序);门控行为。
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ int g_hh_sorted_d = 1;

template <int HT_P, int LOG_NTHR, int BM_WORDS>
__global__ __launch_bounds__((1 << LOG_NTHR), (2048 >> LOG_NTHR)) void hh_repro2_kernel(
    const int *d_seed_col,           // HT_P 槽播种表(-1=空)
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz,
    int *overflow_flag, int *row_ovf, int *gated_cnt,
    long long base, int cap,
    int span_lo, int span_len)       // 门控输入(生产=d_span_lo/len)
{
    extern __shared__ int smem_hh[];
    int *sh_col = smem_hh;
    double *sh_val = (double *)(smem_hh + ((HT_P + 1) & ~1));
    unsigned *bm = (unsigned *)(smem_hh + (((HT_P + 1) & ~1) + 2 * HT_P));   // 表区之后
    const int tid = threadIdx.x;

    // === 门(生产位形:est 域过滤之后,累积之前;超门 → 注册 ovf 交 retry) ===
    if (g_hh_sorted_d && (unsigned)span_len > (unsigned)(BM_WORDS * 32)) {
        if (tid == 0) {
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[0], 1)) atomicAdd(gated_cnt, 1);
        }
        return;
    }

    for (int s = tid; s < HT_P; s += blockDim.x) { sh_col[s] = d_seed_col[s]; sh_val[s] = 1.0; }
    __syncthreads();

    constexpr int NG = (BM_WORDS + 31) >> 5;      // 组数(32 词/组)
    __shared__ int gpre[NG > 1 ? NG : 1];
    const int cbase = span_lo;                     // 位图原点

    for (int w = tid; w < BM_WORDS; w += blockDim.x) bm[w] = 0u;   // 清位图
    __syncthreads();                               // 栅栏 E1

    for (int s = tid; s < HT_P; s += blockDim.x) { // Pass 1:置位
        const int c = sh_col[s];
        if (c != -1) { const int b = c - cbase; atomicOr(&bm[b >> 5], 1u << (b & 31)); }
    }
    __syncthreads();                               // 栅栏 E2

    for (int g = tid; g < NG; g += blockDim.x) {   // 组计数(原地,前缀前)
        int s = 0;
        #pragma unroll 4
        for (int w = g * 32; w < (g + 1) * 32 && w < BM_WORDS; w++) s += __popc(bm[w]);
        gpre[g] = s;
    }
    __syncthreads();                               // 栅栏 E3

    if (tid == 0) {                                // 组独占前缀 + 总数(串行 ≤256,亚μs)
        int acc = 0;
        for (int g = 0; g < NG; g++) { const int t = gpre[g]; gpre[g] = acc; acc += t; }
        row_nnz[0] = acc;
    }
    __syncthreads();                               // 栅栏 E4

    for (int s = tid; s < HT_P; s += blockDim.x) { // Pass 2:位次直写
        const int c = sh_col[s];
        if (c != -1) {
            const int b = c - cbase;
            const int w = b >> 5;
            const unsigned bit = 1u << (b & 31);
            unsigned pre = 0;
            const int gw = w & ~31;
            for (int q = gw; q < w; q++) pre += __popc(bm[q]);
            const int r = gpre[w >> 5] + (int)pre + __popc(bm[w] & (bit - 1));
            if (r < cap) { tmp_key[base + r] = (unsigned int)c; tmp_val[base + r] = sh_val[s]; }
            else { atomicExch(overflow_flag, 1); atomicExch(&row_ovf[0], 2); }   // 超帽(≠1 区分)
        }
    }
}

struct Cfg { int ht, bd, bmw; };

template <int HT_P, int LOG_NTHR, int BM_WORDS>
int run_one(int k, int span_lo, int span_len, const std::vector<int> &seed,
            std::mt19937 &rng, int expect_gate)
{
    const size_t smem = (((size_t)((HT_P + 1) & ~1)) + 2 * (size_t)HT_P) * sizeof(int)
                      + (size_t)BM_WORDS * sizeof(unsigned);
    int *d_seed; unsigned long long *d_key; double *d_val; int *d_nnz, *d_ovf, *d_rovf, *d_gated;
    CK(cudaMalloc(&d_seed, HT_P * sizeof(int)));
    CK(cudaMalloc(&d_key, (size_t)k * 8));
    CK(cudaMalloc(&d_val, (size_t)k * 8));
    CK(cudaMalloc(&d_nnz, 4)); CK(cudaMalloc(&d_ovf, 4)); CK(cudaMalloc(&d_rovf, 4)); CK(cudaMalloc(&d_gated, 4));
    CK(cudaMemset(d_key, 0xAA, (size_t)k * 8));
    CK(cudaMemset(d_nnz, 0, 4)); CK(cudaMemset(d_ovf, 0, 4)); CK(cudaMemset(d_rovf, 0, 4)); CK(cudaMemset(d_gated, 0, 4));
    CK(cudaMemcpy(d_seed, seed.data(), HT_P * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute((hh_repro2_kernel<HT_P, LOG_NTHR, BM_WORDS>), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    hh_repro2_kernel<HT_P, LOG_NTHR, BM_WORDS><<<1, (1 << LOG_NTHR), smem>>>(
        d_seed, d_key, d_val, d_nnz, d_ovf, d_rovf, d_gated, 0LL, k, span_lo, span_len);
    CK(cudaGetLastError());
    int nnz, ovf, rovf, gated;
    CK(cudaMemcpy(&nnz, d_nnz, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ovf, d_ovf, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&rovf, d_rovf, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&gated, d_gated, 4, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> hk(std::max(k, 1));
    CK(cudaMemcpy(hk.data(), d_key, (size_t)k * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_seed); cudaFree(d_key); cudaFree(d_val); cudaFree(d_nnz); cudaFree(d_ovf); cudaFree(d_rovf); cudaFree(d_gated);

    if (expect_gate) {   // 门控行:不写不数,rovf=1
        if (gated != 1 || nnz != 0 || rovf != 1) { printf("  GATE-FAIL k=%d: gated=%d nnz=%d rovf=%d\n", k, gated, nnz, rovf); return 1; }
        return 0;
    }
    std::vector<int> keys;
    for (int s = 0; s < HT_P; s++) if (seed[s] != -1) keys.push_back(seed[s]);
    std::sort(keys.begin(), keys.end());
    if ((int)keys.size() != k) { printf("  SEED-ERR\n"); return 1; }
    if (nnz != k) { printf("  TRIAL k=%d: row_nnz=%d != k\n", k, nnz); return 1; }
    std::vector<int> got;
    for (int t = 0; t < k; t++) {
        unsigned long long v = hk[t];
        if (v == 0xAAAAAAAAAAAAAAAAull) { printf("  TRIAL k=%d: pos %d 空洞\n", k, t); return 1; }
        got.push_back((int)(v & 0xffffffffu));
    }
    if (got != keys) {   // 逐位:必须恰为升序键序列
        printf("  TRIAL k=%d: 输出≠升序键 (got前8:", k);
        for (int t = 0; t < 8 && t < (int)got.size(); t++) printf(" %d", got[t]);
        printf("  want:");
        for (int t = 0; t < 8 && t < (int)keys.size(); t++) printf(" %d", keys[t]);
        printf(")\n");
        return 1;
    }
    if (rovf == 2) { printf("  NOTE k=%d 超帽(cap=k 不应触发,查 cap 逻辑)\n", k); return 1; }
    (void)ovf; (void)rng;
    return 0;
}

// 播种:k 个键 Knuth 散列+线性探查;返回 seed 表;同时给 span 用键 min/max
static std::vector<int> make_seed(int HT_P, int k, std::mt19937 &rng, int &kmin, int &kmax)
{
    std::vector<int> seed(HT_P, -1);
    std::uniform_int_distribution<int> kd(0, 1 << 20);
    std::vector<int> keys;
    keys.reserve(k);
    while ((int)keys.size() < k) {
        int key = kd(rng);
        bool dup = false;
        for (int e : keys) if (e == key) { dup = true; break; }
        if (!dup) keys.push_back(key);
    }
    kmin = *std::min_element(keys.begin(), keys.end());
    kmax = *std::max_element(keys.begin(), keys.end());
    for (int key : keys) {
        unsigned slot = ((unsigned)key * 2654435761u) % (unsigned)HT_P;
        while (seed[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
        seed[slot] = key;
    }
    return seed;
}

template <int HT_P, int LOG_NTHR, int BM_WORDS>
int run_cfg(int trials, std::mt19937 &rng)
{
    int bad = 0, n = 0;
    // 常规随机(键域 2^20 → span 随机,门内/门外都覆盖)
    for (int t = 0; t < trials; t++) {
        int k = 1 + (int)(rng() % (HT_P / 2));
        int kmin, kmax;
        auto seed = make_seed(HT_P, k, rng, kmin, kmax);
        int span_len = kmax - kmin + 1;
        int gated = span_len > BM_WORDS * 32;
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(k, kmin, span_len, seed, rng, gated);
        n++;
    }
    // 聚簇键(全落一个 32 列窗 → 单组重载)
    {
        int k = HT_P / 2; if (k > 32) k = 32;
        std::vector<int> seed(HT_P, -1);
        for (int j = 0; j < k; j++) {
            unsigned slot = ((unsigned)(1000 + j) * 2654435761u) % (unsigned)HT_P;
            while (seed[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
            seed[slot] = 1000 + j;
        }
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(k, 1000, 32, seed, rng, 0); n++;
    }
    // k=1 / 满载 k=HT_P-1
    {
        int kmin, kmax; auto s1 = make_seed(HT_P, 1, rng, kmin, kmax);
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(1, kmin, 1, s1, rng, 0); n++;
        int kk = HT_P - 1;
        auto sf = make_seed(HT_P, kk, rng, kmin, kmax);
        int sl = kmax - kmin + 1;
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(kk, kmin, sl, sf, rng, sl > BM_WORDS * 32); n++;
    }
    // 门边界:span_len = BM_COLS 恰好(入)/+1(出)
    {
        int kmin, kmax;
        auto seed = make_seed(HT_P, 4, rng, kmin, kmax);
        int c0 = kmin;
        std::vector<int> s2(HT_P, -1);
        int placed = 0;
        for (int j = 0; placed < 4 && j < HT_P; j++) {   // 4 键:c0, c0+BM_COLS-1(边界入),2 个中间
            int key = c0 + (int)((j == 0) ? 0 : (j == 1 ? BM_WORDS * 32 - 1 : j * 7));
            unsigned slot = ((unsigned)key * 2654435761u) % (unsigned)HT_P;
            while (s2[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
            s2[slot] = key; placed++;
        }
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(4, c0, BM_WORDS * 32, s2, rng, 0); n++;       // 恰在门内
        bad += run_one<HT_P, LOG_NTHR, BM_WORDS>(4, c0, BM_WORDS * 32 + 1, s2, rng, 1); n++;   // 恰出门
    }
    printf("%s: HT_P=%d BD=%d BM_WORDS=%d(%d列) cases=%d bad=%d\n",
           bad ? "FAIL" : "ALL-PASS", HT_P, 1 << LOG_NTHR, BM_WORDS, BM_WORDS * 32, n, bad);
    return bad;
}

int main(int argc, char **argv)
{
    std::mt19937 rng(20261119);
    int bad = 0;
    if (argc == 5) {   // 单配置:HT_P BD BM_WORDS trials
        int ht = atoi(argv[1]), bd = atoi(argv[2]), bmw = atoi(argv[3]), t = atoi(argv[4]);
        if (ht == 512 && bd == 128) bad += run_cfg<512, 7, 2048>(t, rng);
        else if (ht == 2048 && bd == 256) bad += run_cfg<2048, 8, 2048>(t, rng);
        else if (ht == 4096 && bd == 512) bad += run_cfg<4096, 9, 2048>(t, rng);
        else if (ht == 8192 && bd == 1024) bad += run_cfg<8192, 10, 4096>(t, rng);
        else if (ht == 16253 && bd == 1024) bad += run_cfg<16253, 10, 8192>(t, rng);
        else { printf("未知配置\n"); return 2; }
    } else {           // 全五档
        bad += run_cfg<512, 7, 2048>(400, rng);
        bad += run_cfg<2048, 8, 2048>(400, rng);
        bad += run_cfg<4096, 9, 2048>(400, rng);
        bad += run_cfg<8192, 10, 4096>(400, rng);
        bad += run_cfg<16253, 10, 8192>(300, rng);
    }
    printf("== 总判定: %s ==\n", bad ? "FAIL" : "ALL-GREEN");
    return bad ? 1 : 0;
}
