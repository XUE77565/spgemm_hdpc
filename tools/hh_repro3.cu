// hh_repro3.cu — Gv5a 载体融合提取 微复现器(study/69 刀一)
// 用法: ./hh_repro3 [HT_P blockDim trials]  默认:2048 256 300(bins4/5 形态)
// 现状(翻 bin 人口):无序写 tmp → 独立载体内核(读回+CAP 直方图+排序+再写,一趟 DRAM 往返)。
// Gv5a:排序融进 HH 内核提取——表走一遍收集有效键值到 smem → 块内位偶排序(键值对)
// → 有序直写 tmp。物理量 O(k·log²k),与 span 无关(补位图家族宽 span 盲区)。
// 判据:输出恰为升序键序列;row_nnz==k;哨兵检验双射无空洞。
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define SENTINEL_KEY 0x7fffffff

// N_POW2 = 排序网络宽度(编译期 2 的幂,≥ MAXK);k≤MAXK,空位填 SENTINEL_KEY
template <int HT_P, int LOG_NTHR, int MAXK, int N_POW2>
__global__ __launch_bounds__((1 << LOG_NTHR), (2048 >> LOG_NTHR)) void hh_repro3_kernel(
    const int *d_seed_col, unsigned long long *tmp_key, double *tmp_val, int *row_nnz, long long base)
{
    extern __shared__ int smem[];
    int  *sh_col = smem;                        // [N_POW2]
    double *sh_val = (double *)(smem + N_POW2); // [N_POW2]
    const int tid = threadIdx.x;
    // 初始化:前 k 由收集填,其余填哨兵
    for (int i = tid; i < N_POW2; i += blockDim.x) { sh_col[i] = SENTINEL_KEY; sh_val[i] = 0.0; }
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    for (int s = tid; s < HT_P; s += blockDim.x) {
        const int c = d_seed_col[s];
        if (c != -1) { const int idx = atomicAdd(&cnt, 1); sh_col[idx] = c; sh_val[idx] = 1.0; }
    }
    __syncthreads();
    const int k = cnt;
    if (tid == 0) row_nnz[0] = k;
    // 位偶排序(N_POW2 全网;哨兵自然沉底)。方向:i&p==0 → 升
    for (int p = 2; p <= N_POW2; p <<= 1) {
        for (int d = p >> 1; d >= 1; d >>= 1) {
            __syncthreads();
            for (int i = tid; i < N_POW2; i += blockDim.x) {
                const int j = i ^ d;
                if (j > i) {
                    const int a = sh_col[i], b = sh_col[j];
                    const bool asc = ((i & p) == 0);
                    if (asc ? (a > b) : (a < b)) {
                        sh_col[i] = b; sh_col[j] = a;
                        const double v = sh_val[i]; sh_val[i] = sh_val[j]; sh_val[j] = v;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < k; i += blockDim.x) {
        tmp_key[base + i] = (unsigned)sh_col[i];
        tmp_val[base + i] = sh_val[i];
    }
}

template <int HT_P, int LOG_NTHR, int MAXK, int N_POW2>
int run_one(int k, const std::vector<int> &seed, std::mt19937 &rng)
{
    static int *d_seed = nullptr; static unsigned long long *d_key; static double *d_val; static int *d_nnz;
    if (!d_seed) {
        CK(cudaMalloc(&d_seed, HT_P * 4)); CK(cudaMalloc(&d_key, (size_t)MAXK * 8));
        CK(cudaMalloc(&d_val, (size_t)MAXK * 8)); CK(cudaMalloc(&d_nnz, 4));
    }
    const size_t smem = (size_t)N_POW2 * 4 + (size_t)N_POW2 * 8;   // 键+值(不 含表——播种表走 const)
    CK(cudaMemset(d_key, 0xAA, (size_t)k * 8));
    CK(cudaMemset(d_nnz, 0, 4));
    CK(cudaMemcpy(d_seed, seed.data(), HT_P * 4, cudaMemcpyHostToDevice));
    hh_repro3_kernel<HT_P, LOG_NTHR, MAXK, N_POW2><<<1, (1 << LOG_NTHR), smem>>>(d_seed, d_key, d_val, d_nnz, 0LL);
    CK(cudaGetLastError());
    int nnz; CK(cudaMemcpy(&nnz, d_nnz, 4, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> hk(std::max(k, 1));
    CK(cudaMemcpy(hk.data(), d_key, (size_t)k * 8, cudaMemcpyDeviceToHost));
    std::vector<int> keys;
    for (int s = 0; s < HT_P; s++) if (seed[s] != -1) keys.push_back(seed[s]);
    std::sort(keys.begin(), keys.end());
    if ((int)keys.size() != k || nnz != k) { printf("  TRIAL k=%d: nnz=%d 种子数=%zu 不符\n", k, nnz, keys.size()); return 1; }
    for (int t = 0; t < k; t++) {
        const unsigned long long v = hk[t];
        if (v == 0xAAAAAAAAAAAAAAAAull) { printf("  TRIAL k=%d: pos %d 空洞\n", k, t); return 1; }
        if ((int)(v & 0xffffffffu) != keys[t]) { printf("  TRIAL k=%d: pos %d 错序(%d≠%d)\n", k, t, (int)(v & 0xffffffffu), keys[t]); return 1; }
    }
    (void)rng; return 0;
}

static std::vector<int> make_seed(int HT_P, int k, std::mt19937 &rng)
{
    std::vector<int> seed(HT_P, -1);
    std::uniform_int_distribution<int> kd(0, 1 << 21);
    std::vector<int> keys;
    while ((int)keys.size() < k) {
        int key = kd(rng);
        bool dup = false;
        for (int e : keys) if (e == key) { dup = true; break; }
        if (!dup) keys.push_back(key);
    }
    for (int key : keys) {
        unsigned slot = ((unsigned)key * 2654435761u) % (unsigned)HT_P;
        while (seed[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
        seed[slot] = key;
    }
    return seed;
}

template <int HT_P, int LOG_NTHR, int MAXK>
int run_cfg(int trials, std::mt19937 &rng)
{
    constexpr int N_POW2 = (MAXK <= 2048) ? 2048 : 4096;
    int bad = 0, n = 0;
    for (int t = 0; t < trials; t++) {
        const int k = 1 + (int)(rng() % MAXK);
        auto seed = make_seed(HT_P, k, rng);
        bad += run_one<HT_P, LOG_NTHR, MAXK, N_POW2>(k, seed, rng); n++;
    }
    {   // 边界:k=1、k=MAXK、聚簇键
        auto s1 = make_seed(HT_P, 1, rng);
        bad += run_one<HT_P, LOG_NTHR, MAXK, N_POW2>(1, s1, rng); n++;
        auto sk = make_seed(HT_P, MAXK, rng);
        bad += run_one<HT_P, LOG_NTHR, MAXK, N_POW2>(MAXK, sk, rng); n++;
        std::vector<int> seed(HT_P, -1);
        for (int j = 0; j < 32; j++) {
            unsigned slot = ((unsigned)(1000 + j) * 2654435761u) % (unsigned)HT_P;
            while (seed[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
            seed[slot] = 1000 + j;
        }
        bad += run_one<HT_P, LOG_NTHR, MAXK, N_POW2>(32, seed, rng); n++;
    }
    printf("%s: HT_P=%d BD=%d MAXK=%d cases=%d bad=%d\n", bad ? "FAIL" : "ALL-PASS", HT_P, 1 << LOG_NTHR, MAXK, n, bad);
    return bad;
}

int main(int argc, char **argv)
{
    std::mt19937 rng(20261121);
    int bad = 0;
    bad += run_cfg<512, 7, 512>(300, rng);      // bin4 形态
    bad += run_cfg<2048, 8, 2048>(300, rng);    // bin5 形态(刀一 v1 主靶)
    printf("== 总判定: %s ==\n", bad ? "FAIL" : "ALL-GREEN");
    return bad ? 1 : 0;
}
