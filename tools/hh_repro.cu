// hh_repro.cu — Gv3v2 有序直写提取逻辑的独立微复现器(bcsstk17 空洞取证)
// 用法: ./hh_repro [HT_P blockDim ntrials]   默认 512 128 1000
// 忠实拷贝 src/spgemm_kernel_hash.cu:4557-4603 的 Pass A/前缀/Pass B,
// 跳过累积相,宿主侧直接播种 k 个键进虚拟表(同 Knuth 散列+线性探查布槽)。
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ int g_hh_sorted_d = 1;

template <int HT_P, int LOG_NTHR>
__global__ __launch_bounds__((1 << LOG_NTHR), (2048 >> LOG_NTHR)) void hh_repro_kernel(
    const int *d_seed_col,     // HT_P 槽的播种表(-1 = 空)
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz,
    int *overflow_flag, int *row_ovf, long long base, int cap,
    int *dbg)   // 诊断:[0,HT_P)=每槽最终 r(-2 空/-3 超帽);[HT_P..]=seg_pre 后缀
{
    extern __shared__ int smem_hh[];
    int *sh_col = smem_hh;
    double *sh_val = (double *)(smem_hh + ((HT_P + 1) & ~1));
    const int tid = threadIdx.x;
    for (int s = tid; s < HT_P; s += blockDim.x) { sh_col[s] = d_seed_col[s]; sh_val[s] = 1.0; }
    if (dbg && tid < HT_P) dbg[tid] = -2;
    __syncthreads();

    // ==== 以下逐行拷贝自生产内核 4557-4603(Gv3v2) ====
    if (g_hh_sorted_d) {
        const int lane5 = tid & 31, wid = tid >> 5, nw = (int)(blockDim.x >> 5);
        const int nseg = (HT_P + 31) >> 5;
        __shared__ int seg_pre[(HT_P + 31) >> 5 > 1 ? (HT_P + 31) >> 5 : 1];
        for (int seg = wid; seg < nseg; seg += nw) {
            const int lo = (seg << 5) + lane5;
            const int c = (lo < HT_P) ? sh_col[lo] : -1;
            const unsigned valid = __ballot_sync(0xffffffffu, c != -1);
            if (lane5 == 0) seg_pre[seg] = __popc(valid);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0;
            for (int s = 0; s < nseg; s++) { const int t = seg_pre[s]; seg_pre[s] = acc; acc += t; }
            row_nnz[0] = acc;
            if (dbg) for (int s = 0; s < nseg; s++) dbg[HT_P + s] = seg_pre[s];
        }
        __syncthreads();
        for (int seg = wid; seg < nseg; seg += nw) {
            const int lo = (seg << 5) + lane5;
            const int c = (lo < HT_P) ? sh_col[lo] : -1;
            unsigned lt = 0;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int other = __shfl_xor_sync(0xffffffffu, c, off);
                if (other != -1 && other < c) lt |= (1u << (lane5 ^ off));
            }
            if (c != -1) {
                const int r = seg_pre[seg] + __popc(lt);
                if (r < cap) {
                    tmp_key[base + r] = (unsigned int)c;
                    tmp_val[base + r] = sh_val[lo];
                    if (dbg) dbg[lo] = r;
                } else {
                    atomicExch(overflow_flag, 1);
                    atomicExch(&row_ovf[0], 1);
                    if (dbg) dbg[lo] = -3;
                }
            }
        }
    }
}

template <int HT_P, int LOG_NTHR>
int run_trial(int k, std::mt19937 &rng)
{
    const size_t smem = ((size_t)((HT_P + 1) & ~1)) * sizeof(int) + (size_t)HT_P * sizeof(double);
    std::vector<int> seed(HT_P, -1);
    // 播种:k 个随机列,Knuth 散列+线性探查(与生产累积同布槽)
    std::uniform_int_distribution<int> keyd(0, 1000000);
    std::vector<int> keys;
    for (int t = 0; t < k; t++) {
        int key = keyd(rng);
        while (true) { // 去重
            bool dup = false;
            for (int e : keys) if (e == key) dup = true;
            if (!dup) break; key = keyd(rng);
        }
        keys.push_back(key);
        unsigned slot = ((unsigned)key * 2654435761u) % (unsigned)HT_P;
        while (seed[slot] != -1) slot = (slot + 1 >= (unsigned)HT_P) ? 0 : slot + 1;
        seed[slot] = key;
    }
    int *d_seed; unsigned long long *d_key; double *d_val; int *d_nnz, *d_ovf, *d_rovf, *d_dbg;
    CK(cudaMalloc(&d_seed, HT_P * sizeof(int)));
    CK(cudaMalloc(&d_key, (size_t)k * 8));
    CK(cudaMalloc(&d_val, (size_t)k * 8));
    CK(cudaMalloc(&d_nnz, sizeof(int)));
    CK(cudaMalloc(&d_ovf, sizeof(int)));
    CK(cudaMalloc(&d_rovf, sizeof(int)));
    CK(cudaMalloc(&d_dbg, (HT_P + 256) * sizeof(int)));
    CK(cudaMemset(d_key, 0xAA, (size_t)k * 8));   // 残留哨兵:未写位置可见
    CK(cudaMemset(d_nnz, 0, 4)); CK(cudaMemset(d_ovf, 0, 4)); CK(cudaMemset(d_rovf, 0, 4));
    CK(cudaMemcpy(d_seed, seed.data(), HT_P * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(hh_repro_kernel<HT_P, LOG_NTHR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    hh_repro_kernel<HT_P, LOG_NTHR><<<1, (1 << LOG_NTHR), smem>>>(d_seed, d_key, d_val, d_nnz, d_ovf, d_rovf, 0LL, k, d_dbg);
    CK(cudaGetLastError());
    int nnz, ovf, rovf;
    std::vector<int> hdbg(HT_P + 256);
    CK(cudaMemcpy(&nnz, d_nnz, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ovf, d_ovf, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&rovf, d_rovf, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdbg.data(), d_dbg, (HT_P + 256) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> hk(k);
    CK(cudaMemcpy(hk.data(), d_key, (size_t)k * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_seed); cudaFree(d_key); cudaFree(d_val); cudaFree(d_nnz); cudaFree(d_ovf); cudaFree(d_rovf);

    int bad = 0;
    if (nnz != k) { printf("  TRIAL k=%d: row_nnz=%d != k!\n", k, nnz); bad = 1; }
    std::vector<int> got;
    int holes = 0;
    for (int t = 0; t < k; t++) {
        unsigned long long v = hk[t];
        if (v == 0xAAAAAAAAAAAAAAAAull) { holes++; continue; }   // 从未被写
        got.push_back((int)(v & 0xffffffffu));
    }
    std::sort(keys.begin(), keys.end());
    std::sort(got.begin(), got.end());
    if (holes || got != keys) {
        printf("  TRIAL k=%d: nnz=%d 空洞=%d 写入=%d", k, nnz, holes, (int)got.size());
        if (got != keys) printf(" 且键集合错");
        printf("  (ovf=%d rovf=%d)\n", ovf, rovf);
        // 诊断:宿主公式 rank vs kernel rank,逐槽对照
        const int nseg = (HT_P + 31) >> 5;
        std::vector<int> segcnt(nseg, 0);
        for (int s = 0; s < HT_P; s++) if (seed[s] != -1) segcnt[s >> 5]++;
        int shown = 0;
        for (int s = 0; s < HT_P && shown < 8; s++) {
            if (seed[s] == -1) continue;
            int seg = s >> 5;
            int below = 0, smaller = 0;
            for (int q = 0; q < seg; q++) below += segcnt[q];
            for (int q = seg << 5; q < ((seg + 1) << 5) && q < HT_P; q++)
                if (seed[q] != -1 && seed[q] < seed[s]) smaller++;
            int exp_r = below + smaller;
            int got_r = hdbg[s];
            if (got_r != exp_r) {
                printf("    槽%d seg%d 键%d: 公式r=%d kernel_r=%d seg_pre[%d]=%d(期望%d) 段内更小=%d\n",
                       s, seg, seed[s], exp_r, got_r, seg, hdbg[HT_P + seg], below, smaller);
                shown++;
            }
        }
        if (shown == 0) printf("    (公式 r 全对但仍有空洞 → 写入丢失/覆盖)\n");
        bad = 1;
    }
    return bad;
}

int main(int argc, char **argv)
{
    const int HT_P = argc > 1 ? atoi(argv[1]) : 512;
    const int BD   = argc > 2 ? atoi(argv[2]) : 128;
    const int T    = argc > 3 ? atoi(argv[3]) : 1000;
    std::mt19937 rng(12345);
    int bad = 0;
    if (HT_P == 512 && BD == 128)      { for (int t = 0; t < T; t++) bad += run_trial<512, 7>(1 + (int)(rng() % 400), rng); }
    else if (HT_P == 2048 && BD == 256) { for (int t = 0; t < T; t++) bad += run_trial<2048, 8>(1 + (int)(rng() % 1600), rng); }
    else if (HT_P == 4096 && BD == 512) { for (int t = 0; t < T; t++) bad += run_trial<4096, 9>(1 + (int)(rng() % 3200), rng); }
    else if (HT_P == 8192 && BD == 1024){ for (int t = 0; t < T; t++) bad += run_trial<8192, 10>(1 + (int)(rng() % 6400), rng); }
    else { printf("未支持组合\n"); return 2; }
    printf("%s: HT_P=%d BD=%d trials=%d bad=%d\n", bad ? "FAIL" : "ALL-PASS", HT_P, BD, T, bad);
    return bad ? 1 : 0;
}
