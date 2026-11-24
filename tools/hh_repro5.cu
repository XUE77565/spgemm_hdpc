// hh_repro5.cu — 刀三·想法二(位图标记-累积两遍式)生产例程的微复现器(study/69 §二十六)
// 与 hh_repro4 的区别:本件 #include 生产头 src/hh_bm2.cuh,跑的就是 hash_heavy_kernel<...,BM2>
// 里那同一个 hh_bm2_row / hh_bm2_fits / hh_smem_words,不是手抄的同构件。
// 构建: nvcc -O3 -arch=sm_90 -std=c++14 -Isrc tools/hh_repro5.cu -o hh_repro5
// 用法: ./hh_repro5 [trials_per_cfg=40] [seed=42]
// 覆盖:块宽 128/256/512/1024 × 生产各 bin 的 SMEM 字预算(位图形态与 fused-legacy 形态)× B16 开/关
//       × 值区模式 1(SMEM)/2(全局直写);行人口含 dup 1-60×、聚簇列、空 B 行、span 贴 32 边界、
//       est 低估(n>cap,应注册 ovf 且不写输出)、SMEM 装不下(应原样跳过)。
// 判据(逐行):row_nnz==n;键恰为真值升序;值相对误差≤1e-9(按该列 Σ|乘积| 定标);[n,cap) 与
//       非本行区域哨兵未动(抓越界写);ovf 行在 ovf_rows 里恰出现一次;跳过行零副作用。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <random>
#include <cuda_runtime.h>
#include "hh_bm2.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static const unsigned long long KEY_SENT = 0xAAAAAAAAAAAAAAAAull;
static const double VAL_SENT = -7.25e300;

// 与生产 hash_heavy_kernel 的 BM2 分流块同式:跨度→词数、槽宽、SMEM 判定、调用实参次序
template<int LNT, int B16ON, int VMODE>
__global__ void repro5_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val, const uint4 *B16,
    const int *row_ids, const long long *d_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz,
    int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *d_flop, const int *d_maxbl, const int *d_span_lo, const int *d_span_len,
    const int totw)
{
    extern __shared__ int smem_hh[];
    const int i2 = row_ids[blockIdx.x];
    const int sl2 = d_span_len[i2];
    const int cw2 = (int)(((unsigned)sl2 + 31u) >> 5);
    const int cap2 = (int)(d_off[i2 + 1] - d_off[i2]);
    const long long fl2 = (long long)d_flop[i2];
    const bool ok2 = sl2 > 0 && hh_bm2_fits(cw2, cap2, totw, VMODE);
    if (!ok2) { if (threadIdx.x == 0) row_nnz[i2] = -2; return; }
    hh_bm2_row<LNT, B16ON, VMODE>(
        A_rp, A_ci, A_val, B_rp, B_ci, B_val, B16,
        i2, d_span_lo[i2], cw2, cap2, d_off[i2], fl2, d_maxbl[i2],
        tmp_key, tmp_val, row_nnz, overflow_flag, ovf_rows, ovf_cnt, row_ovf, smem_hh);
}

struct Case {
    int R = 0, nB = 0;
    std::vector<int> A_rp, A_ci; std::vector<double> A_val;
    std::vector<int> B_rp, B_ci; std::vector<double> B_val;
    std::vector<uint4> B16;
    std::vector<int> flop, maxbl, span_lo, span_len, cap, row_ids;
    std::vector<long long> off;
    std::vector<std::vector<int> > tcol;
    std::vector<std::vector<double> > tval, tabs;
};

static void gen_case(std::mt19937 &rng, int span_max, int cap_room, Case &c)
{
    std::normal_distribution<double> vd(1.0, 0.75);
    const int col_lo = (int)(rng() % 200000u);
    int span = 1 + (int)(rng() % (unsigned)span_max);
    if (rng() % 5 == 0) span = ((span + 31) / 32) * 32;                // 贴 32 边界
    if (rng() % 9 == 0) span = 1 + (int)(rng() % 70u);                 // 极窄
    c.nB = 1 + (int)(rng() % 96u);
    c.R = 48;
    c.B_rp.assign(1, 0); c.B_ci.clear(); c.B_val.clear();
    static const int LMAX[5] = {4, 16, 64, 256, 1024};
    const int lmax = LMAX[rng() % 5];
    const bool clustered = (rng() % 3 == 0);
    for (int k = 0; k < c.nB; k++) {
        int len = (rng() % 20 == 0) ? 0 : 1 + (int)(rng() % (unsigned)lmax);
        int w_lo = 0, w_len = span;
        if (clustered) { w_lo = (int)(rng() % (unsigned)span); w_len = std::min(span - w_lo, 1 + (int)(rng() % 160u)); }
        std::vector<int> pk;
        for (int t = 0; t < len; t++) pk.push_back(col_lo + w_lo + (int)(rng() % (unsigned)w_len));
        std::sort(pk.begin(), pk.end());
        pk.erase(std::unique(pk.begin(), pk.end()), pk.end());
        for (size_t t = 0; t < pk.size(); t++) { c.B_ci.push_back(pk[t]); c.B_val.push_back(vd(rng)); }
        c.B_rp.push_back((int)c.B_ci.size());
    }
    c.B16.resize(c.B_ci.size() ? c.B_ci.size() : 1);
    for (size_t q = 0; q < c.B_ci.size(); q++) {
        unsigned long long bits; std::memcpy(&bits, &c.B_val[q], 8);
        uint4 e4;                                                      // b_interleave_kernel 同布局:col|pad|val 低 32|val 高 32
        e4.x = (unsigned)c.B_ci[q]; e4.y = 0u; e4.z = (unsigned)(bits & 0xffffffffull); e4.w = (unsigned)(bits >> 32);
        c.B16[q] = e4;
    }
    c.A_rp.assign(1, 0); c.A_ci.clear(); c.A_val.clear();
    std::vector<int> perm(c.nB);
    for (int k = 0; k < c.nB; k++) perm[k] = k;
    for (int i = 0; i < c.R; i++) {
        const int al = 1 + (int)(rng() % (unsigned)std::min(c.nB, 64));
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int> ks(perm.begin(), perm.begin() + al);
        std::sort(ks.begin(), ks.end());
        for (int t = 0; t < al; t++) { c.A_ci.push_back(ks[t]); c.A_val.push_back(vd(rng)); }
        c.A_rp.push_back((int)c.A_ci.size());
    }
    c.flop.assign(c.R, 0); c.maxbl.assign(c.R, 0); c.span_lo.assign(c.R, 0); c.span_len.assign(c.R, 0);
    c.cap.assign(c.R, 1); c.off.assign(c.R + 1, 0); c.row_ids.resize(c.R);
    c.tcol.assign(c.R, std::vector<int>()); c.tval.assign(c.R, std::vector<double>()); c.tabs.assign(c.R, std::vector<double>());
    for (int i = 0; i < c.R; i++) {
        c.row_ids[i] = c.R - 1 - i;                                    // 行表非恒等(块号≠行号)
        int lo = 0x7fffffff, hi = -1, s = 0, mb = 0;
        std::map<int, std::pair<double, double> > acc;
        for (int p = c.A_rp[i]; p < c.A_rp[i + 1]; p++) {
            const int k = c.A_ci[p];
            const int b0 = c.B_rp[k], b1 = c.B_rp[k + 1];
            s += b1 - b0; mb = std::max(mb, b1 - b0);
            if (b0 < b1) { lo = std::min(lo, c.B_ci[b0]); hi = std::max(hi, c.B_ci[b1 - 1]); }   // count_flop_span_kernel 同式
            for (int q = b0; q < b1; q++) {
                const double v = c.A_val[p] * c.B_val[q];
                std::pair<double, double> &e = acc[c.B_ci[q]];
                e.first += v; e.second += std::fabs(v);
            }
        }
        c.flop[i] = s; c.maxbl[i] = mb;
        c.span_lo[i] = (lo == 0x7fffffff) ? 0 : lo;
        c.span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
        for (std::map<int, std::pair<double, double> >::iterator it = acc.begin(); it != acc.end(); ++it) {
            c.tcol[i].push_back(it->first); c.tval[i].push_back(it->second.first); c.tabs[i].push_back(it->second.second);
        }
        const int n = (int)c.tcol[i].size();
        int cap = n + ((rng() % 3 == 0) ? 0 : (int)(rng() % 8u));
        if (n >= 2 && rng() % 16 == 0) cap = std::max(1, n - 1 - (int)(rng() % 3u));   // est 低估 → 应 ovf
        if (rng() % 16 == 0) cap = n + cap_room;                                       // 超大槽宽 → VMODE=1 可能装不下
        if (cap < 1) cap = 1;
        c.cap[i] = cap;
    }
    for (int i = 0; i < c.R; i++) c.off[i + 1] = c.off[i] + c.cap[i];
}

template<class T> static T *up(const std::vector<T> &h)
{
    T *d = nullptr;
    const size_t nb = (h.size() ? h.size() : 1) * sizeof(T);
    CK(cudaMalloc(&d, nb));
    if (h.size()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct Tally { long long rows = 0, ok = 0, ovf = 0, skip = 0, bad = 0; };

template<int LNT, int B16ON, int VMODE>
static void run_cfg(const Case &c, const int totw, Tally &tl, const char *cfg, const int trial)
{
    const size_t smem = (size_t)totw * sizeof(int);
    CK(cudaFuncSetAttribute((const void *)repro5_kernel<LNT, B16ON, VMODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));   // 同 launch_csort 的写法
    const long long tot = c.off[c.R];
    std::vector<unsigned long long> hk((size_t)tot, KEY_SENT);
    std::vector<double> hv((size_t)tot, VAL_SENT);
    std::vector<int> hnnz(c.R, -7), hovf(c.R, 0), hlist(c.R, -1);
    int *dA_rp = up(c.A_rp), *dA_ci = up(c.A_ci), *dB_rp = up(c.B_rp), *dB_ci = up(c.B_ci);
    double *dA_val = up(c.A_val), *dB_val = up(c.B_val);
    uint4 *dB16 = up(c.B16);
    int *d_rows = up(c.row_ids), *d_flop = up(c.flop), *d_maxbl = up(c.maxbl), *d_slo = up(c.span_lo), *d_sln = up(c.span_len);
    long long *d_off = up(c.off);
    unsigned long long *d_key = up(hk); double *d_val = up(hv);
    int *d_nnz = up(hnnz), *d_rovf = up(hovf), *d_list = up(hlist);
    int *d_flag = nullptr, *d_cnt = nullptr;
    CK(cudaMalloc(&d_flag, sizeof(int))); CK(cudaMalloc(&d_cnt, sizeof(int)));
    CK(cudaMemset(d_flag, 0, sizeof(int))); CK(cudaMemset(d_cnt, 0, sizeof(int)));
    repro5_kernel<LNT, B16ON, VMODE><<<c.R, (1 << LNT), smem>>>(
        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, B16ON ? dB16 : nullptr,
        d_rows, d_off, d_key, d_val, d_nnz, d_flag, d_list, d_cnt, d_rovf,
        d_flop, d_maxbl, d_slo, d_sln, totw);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int h_flag = 0, h_cnt = 0;
    if (tot > 0) {
        CK(cudaMemcpy(hk.data(), d_key, (size_t)tot * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(), d_val, (size_t)tot * 8, cudaMemcpyDeviceToHost));
    }
    CK(cudaMemcpy(hnnz.data(), d_nnz, (size_t)c.R * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hovf.data(), d_rovf, (size_t)c.R * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hlist.data(), d_list, (size_t)c.R * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_flag, d_flag, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_cnt, d_cnt, 4, cudaMemcpyDeviceToHost));
    cudaFree(dA_rp); cudaFree(dA_ci); cudaFree(dB_rp); cudaFree(dB_ci); cudaFree(dA_val); cudaFree(dB_val); cudaFree(dB16);
    cudaFree(d_rows); cudaFree(d_flop); cudaFree(d_maxbl); cudaFree(d_slo); cudaFree(d_sln); cudaFree(d_off);
    cudaFree(d_key); cudaFree(d_val); cudaFree(d_nnz); cudaFree(d_rovf); cudaFree(d_list); cudaFree(d_flag); cudaFree(d_cnt);

    int exp_ovf = 0;
    for (int i = 0; i < c.R; i++) {
        tl.rows++;
        const int n = (int)c.tcol[i].size();
        const int cap = c.cap[i];
        const int cw = (int)(((unsigned)c.span_len[i] + 31u) >> 5);
        const bool fits = c.span_len[i] > 0 && hh_bm2_fits(cw, cap, totw, VMODE);
        const long long base = c.off[i];
        const char *why = nullptr;
        int untouched_from = 0;                                        // [untouched_from, cap) 须保持哨兵
        if (!fits) {
            tl.skip++;
            if (hnnz[i] != -2) why = "跳过行 row_nnz≠-2";
            else if (hovf[i] != 0) why = "跳过行被置 ovf";
        } else if (n > cap) {
            tl.ovf++; exp_ovf++;
            if (hovf[i] != 1) why = "低估行未置 row_ovf";
            else if (hnnz[i] != 0) why = "低估行 row_nnz≠0";
            else {
                int seen = 0;
                for (int t = 0; t < h_cnt && t < c.R; t++) seen += (hlist[t] == i);
                if (seen != 1) why = "低估行在 ovf_rows 中出现次数≠1";
            }
        } else {
            untouched_from = n;
            if (hovf[i] != 0) why = "正常行被置 ovf";
            else if (hnnz[i] != n) why = "row_nnz≠真值 distinct 数";
            else {
                for (int t = 0; t < n && !why; t++) {
                    const unsigned long long want = ((unsigned long long)(unsigned)i << 32) | (unsigned long long)(unsigned)c.tcol[i][t];
                    if (hk[(size_t)(base + t)] != want) why = "键错/乱序";
                    else if (std::fabs(hv[(size_t)(base + t)] - c.tval[i][t]) > 1e-9 * (c.tabs[i][t] + 1e-300)) why = "值错";
                }
            }
        }
        for (int t = untouched_from; t < cap && !why; t++)
            if (hk[(size_t)(base + t)] != KEY_SENT || hv[(size_t)(base + t)] != VAL_SENT) why = "槽外/未授权写(哨兵被动)";
        if (why) {
            tl.bad++;
            if (tl.bad <= 12)
                printf("  BAD %s trial=%d row=%d: %s (n=%d cap=%d span=%d flop=%d a_len=%d got_nnz=%d ovf=%d)\n",
                       cfg, trial, i, why, n, cap, c.span_len[i], c.flop[i], c.A_rp[i + 1] - c.A_rp[i], hnnz[i], hovf[i]);
        } else if (fits && n <= cap) tl.ok++;
    }
    if (h_cnt != exp_ovf || (exp_ovf > 0) != (h_flag != 0)) {
        tl.bad++;
        if (tl.bad <= 12) printf("  BAD %s trial=%d: ovf 计数/总旗不符 (ovf_cnt=%d 期望=%d flag=%d)\n", cfg, trial, h_cnt, exp_ovf, h_flag);
    }
}

template<int LNT>
static long long run_geom(std::mt19937 &rng, const int totw, const int trials, const char *geom)
{
    // 跨度上限取 SMEM 可容的 ~1.25 倍:多数行装得下,少数走"装不下原样跳过"分支
    const int span_max = std::max(64, (int)(16LL * totw * 5 / 4));
    Tally t10, t11, t20, t21;
    char nm[96];
    for (int tr = 0; tr < trials; tr++) {
        Case c;
        gen_case(rng, span_max, totw, c);
        snprintf(nm, sizeof(nm), "%s/B16=0/V=1", geom); run_cfg<LNT, 0, 1>(c, totw, t10, nm, tr);
        snprintf(nm, sizeof(nm), "%s/B16=1/V=1", geom); run_cfg<LNT, 1, 1>(c, totw, t11, nm, tr);
        snprintf(nm, sizeof(nm), "%s/B16=0/V=2", geom); run_cfg<LNT, 0, 2>(c, totw, t20, nm, tr);
        snprintf(nm, sizeof(nm), "%s/B16=1/V=2", geom); run_cfg<LNT, 1, 2>(c, totw, t21, nm, tr);
    }
    const Tally *ts[4] = {&t10, &t11, &t20, &t21};
    static const char *tn[4] = {"B16=0/V=1", "B16=1/V=1", "B16=0/V=2", "B16=1/V=2"};
    long long bad = 0;
    for (int x = 0; x < 4; x++) {
        printf("%-34s %-10s rows=%lld ok=%lld ovf=%lld skip=%lld bad=%lld\n", geom, tn[x],
               ts[x]->rows, ts[x]->ok, ts[x]->ovf, ts[x]->skip, ts[x]->bad);
        bad += ts[x]->bad;
    }
    return bad;
}

int main(int argc, char **argv)
{
    const int trials = argc > 1 ? atoi(argv[1]) : 40;
    const unsigned seed = argc > 2 ? (unsigned)atoi(argv[2]) : 42u;
    std::mt19937 rng(seed);
    long long bad = 0;
    // (块宽, SMEM 字预算) = 生产 HH_LAUNCH 各 bin:hh_smem_words(HT_P, BM_WORDS, FUSED, LEG)
    bad += run_geom<7>(rng, hh_smem_words(512, 2048, 1, 0), trials, "bin4 128T bitmap(3584w)");
    bad += run_geom<7>(rng, hh_smem_words(512, 2048, 1, 1), trials, "bin4 128T fused-leg(2688w)");
    bad += run_geom<8>(rng, hh_smem_words(2048, 2048, 1, 0), trials, "bin5 256T bitmap(8192w)");
    bad += run_geom<8>(rng, hh_smem_words(2048, 2048, 1, 1), trials, "bin5 256T fused-leg(7296w)");
    bad += run_geom<9>(rng, hh_smem_words(2719, 2048, 0, 0), trials, "bin6 512T bitmap(10206w)");
    bad += run_geom<9>(rng, hh_smem_words(4096, 2048, 0, 0), trials, "bin7 512T bitmap(14336w)");
    bad += run_geom<9>(rng, hh_smem_words(4096, 2048, 0, 1), trials, "bin7 512T legacy(12288w)");
    bad += run_geom<10>(rng, hh_smem_words(8192, 4096, 0, 0), trials, "bin8 1024T bitmap(28672w)");
    bad += run_geom<10>(rng, hh_smem_words(16253, 8192, 0, 0), trials, "bin9 1024T bitmap(56952w)");
    printf("%s: trials/cfg=%d seed=%u bad=%lld\n", bad ? "FAIL" : "ALL-PASS", trials, seed, bad);
    return bad ? 1 : 0;
}
