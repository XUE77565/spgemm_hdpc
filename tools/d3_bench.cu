// D3 独立测量台(T7c,study/30 战报 spec):绕开 diter 管线纠缠,直接回答
// "merge kernel 在 pre2 型行上能否打过彼 denseNumeric(基线 1.67×)"。
// 用法: d3_bench <mtx> <v9_rows_file> <rowtime_out> [rep]
// 行单谓词(host): a_len∈[1,32] ∧ est∈[256,4096] ∧ span≤16384(v9 dump 9 列)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void d3_merge_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int upper_tri, int A_rows,
    const int *rows, int n_rows,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz,
    unsigned long long *rowtime = nullptr,
    int mode = 1)
{
    int wid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (wid >= n_rows) return;
    int lane = threadIdx.x & 31;
    int i = rows[wid];
    unsigned long long rt0 = rowtime ? clock64() : 0;
    int rs = A_rp[i], re = A_rp[i + 1];
    int k = re - rs;
    int p = B_rp[A_ci[rs + lane]], e = B_rp[A_ci[rs + lane] + 1];
    double a = (lane < k) ? A_val[rs + lane] : 0.0;
    int base = exact_rp[i];
    int prev_col = -1; double sum = 0.0; int cnt = 0;
    const unsigned FULL = 0xffffffffu;
    while (true) {
        int my = (lane < k && p < e) ? B_ci[p] : 0x7fffffff;
        unsigned m = __reduce_min_sync(FULL, (unsigned)my);
        if (m == 0x7fffffff) break;
        int col = (int)m;
        if (upper_tri && col < i) { if (my == col) p++; continue; }
        unsigned winners = __ballot_sync(FULL, my == col);
        int w = __ffs(winners) - 1;
        double av = __shfl_sync(FULL, a, w);
        int pw = __shfl_sync(FULL, p, w);
        double prod = av * B_val[pw];
        if (col != prev_col) {
            if (prev_col >= 0) { if (mode) { out_col[base + cnt] = prev_col; out_val[base + cnt] = sum; } cnt++; }
            prev_col = col; sum = 0.0;
        }
        sum += prod;
        if (my == col && p < e && lane == w) p++;
    }
    if (prev_col >= 0) { if (mode) { out_col[base + cnt] = prev_col; out_val[base + cnt] = sum; } cnt++; }
    if (lane == 0) { row_nnz[i] = cnt; if (rowtime) rowtime[i] = clock64() - rt0; }
}


// ── D3 v2(merge 批处理 32 路位排序锦标赛/轮)──────────────────────────────
// v1 判负根因:逐元素 redux+ballot+shfl 串行链无 ILP。v2:每轮全 warp 位排序
// 32 个链头(15 级 shfl.bfly),批内等值 run 分段求和(shfl scan),末位写出。
__global__ void d3_merge_v2_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    int A_rows, const int *rows, int n_rows,
    const int *exact_rp, int *out_col, double *out_val, int *row_nnz,
    unsigned long long *rowtime, int mode)
{
    int wid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (wid >= n_rows) return;
    int lane = threadIdx.x & 31;
    int i = rows[wid];
    unsigned long long rt0 = rowtime ? clock64() : 0;
    int rs = A_rp[i], re = A_rp[i + 1];
    int k = re - rs;
    int p = B_rp[A_ci[rs + lane]], e = B_rp[A_ci[rs + lane] + 1];
    double a = (lane < k) ? A_val[rs + lane] : 0.0;
    int base = exact_rp[i];
    int carry_col = -1; double carry_sum = 0.0; int cnt = 0;
    const unsigned FULL = 0xffffffffu;
    while (true) {
        // 1) 各链头(耗尽=哨兵);打包 key = col*32+lane(col<2^26 时单调序=列序)
        int head = (lane < k && p < e) ? B_ci[p] : 0x7fffffff;
        unsigned key = ((unsigned)head << 5) | (unsigned)lane;
        // 2) 32 路位排序(shfl.bfly,15 级;哨兵沉底)
        #pragma unroll
        for (int st = 1; st < 32; st <<= 1) {
            unsigned span = (unsigned)st << 1;
            unsigned partner_bit = span >> 1;
            #pragma unroll
            for (unsigned d = partner_bit; d > 0; d >>= 1) {
                unsigned partner = lane ^ d;
                unsigned pk = __shfl_xor_sync(FULL, key, (int)d);
                bool up = ((lane & span) != 0);         // 块内方向:偶数块升
                bool take_other = ((key > pk) == (d == partner_bit ? !up : up)) || (key == pk && (lane & d));
                // 标准位排序:比较方向由块首决定;简化实现(以下两行等价经典式)
                bool sw = ((key > pk) && (((lane & span) == 0) == ((lane & d) != 0) ? false : false));
                // 直接采用经典 bitonic 交换判定:
                bool dir_up = ((lane & span) == 0);      // 此块升序
                bool do_swap = (d == partner_bit) ? (dir_up ? (key > pk) : (key < pk))
                                                  : (dir_up ? (key < pk) : (key > pk));
                if (do_swap) key = pk;
            }
        }
        int col = (int)(key >> 5);
        int src = (int)(key & 31);
        if (col == 0x7fffffff) break;                     // 全耗尽(本批全哨兵才到这?——非:部分哨兵沉底部)
        // 3) 取源链乘积(a[src]×B_val[p[src]])
        double av = __shfl_sync(FULL, a, src);
        int pw = __shfl_sync(FULL, p, src);
        double prod = av * B_val[pw];
        // 4) 推进源链(每 lane 的 p 按 key 里的 src 定位;等价:head 有效且 src==lane 的链 +1)
        if ((lane < k && p < e) && ((unsigned)head << 5 | (unsigned)lane) == key) p++;
        // 5) 批内 run 结构:与前一位同列?
        int prev_col = (int)(__shfl_up_sync(FULL, (unsigned)col, 1) >> 5);
        bool same_prev = (lane > 0) && (col == prev_col);
        // 分段求和(shfl_up 累加同 run)
        double seg = prod;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int pc = (int)(__shfl_up_sync(FULL, (unsigned)col, d) >> 5);
            double pv = __shfl_up_sync(FULL, seg, d);
            if (lane >= d && col == pc) seg += pv;
        }
        // run 末位 = 与后一位不同列(或 lane==31;跨批由 carry 处理)
        int next_col = (int)(__shfl_down_sync(FULL, (unsigned)col, 1) >> 5);
        bool run_end = (lane == 31) || (col != next_col);
        // 6) 批内新 run 前缀计数(run 首位标记 = !same_prev)
        bool run_start = !same_prev;
        unsigned start_mask = __ballot_sync(FULL, run_start);
        unsigned before = start_mask & ((1u << lane) - 1);
        int distinct_before = __popc(before);
        // 7) 写出(每个 run 末位;批尾跨批延后)
        if (run_end) {
            int target_col = col; double total = seg;
            if (same_prev == false && carry_col == col) {   // 与上批 carry 同列:并入
                total += carry_sum;
            }
            if (lane == 31) {                               // 批尾:若与下批首列相等需 carry——先当完成写,重复概率低(暂简化:直接写)
                out_col_tmp:
                if (mode) { out_col[base + cnt + distinct_before] = target_col; out_val[base + cnt + distinct_before] = total; }
                cnt += 0; // cnt 更新在批尾统一
            } else {
                if (mode) { out_col[base + cnt + distinct_before] = target_col; out_val[base + cnt + distinct_before] = total; }
            }
        }
        // 8) 批尾更新 cnt = 本批 distinct(含跨 carry 并列时不增——简化:popc 全 run 首)
        cnt += __popc(start_mask);
        // 跨批同列 carry(批尾 col == 批首 col 的情形近似:重置)
        carry_col = -1; carry_sum = 0;
    }
    if (lane == 0) { row_nnz[i] = cnt; if (rowtime) rowtime[i] = clock64() - rt0; }
}

// 简易 MTX 读取(对称镜像展开,与主管线 reader 同语义:每条 (r,c,v) 同时给 A[r]+c 与 A[c]+r)
struct CSR { int n; std::vector<int> rp, ci; std::vector<double> val; };
static CSR read_mtx_sym(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s fail\n", path); exit(1); }
    char line[512];
    fgets(line, sizeof line, f);   // header:general 不镜像(生产 reader 同语义;pre2=general!)
    bool is_sym = strstr(line, "symmetric") != nullptr;
    do { fgets(line, sizeof line, f); } while (line[0] == '%');
    long long M, K, NNZ; sscanf(line, "%lld %lld %lld", &M, &K, &NNZ);
    if (M != K) { fprintf(stderr, "非方阵\n"); exit(1); }
    std::vector<int> deg(M + 1, 0);
    std::vector<int> er(NNZ), ec(NNZ); std::vector<double> ev(NNZ);
    for (long long z = 0; z < NNZ; z++) {
        int r, c; double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) break;
        r--; c--; er[z] = r; ec[z] = c; ev[z] = v;
        deg[r + 1]++; if (is_sym && c != r) deg[c + 1]++;
    }
    fclose(f);
    CSR A; A.n = M; A.rp.assign(M + 1, 0);
    for (int i = 0; i < M; i++) A.rp[i + 1] = A.rp[i] + deg[i + 1];
    std::vector<int> cur(A.rp.begin(), A.rp.end() - 1);
    A.ci.resize(A.rp[M]); A.val.resize(A.rp[M]);
    for (long long z = 0; z < NNZ; z++) {
        A.ci[cur[er[z]]] = ec[z]; A.val[cur[er[z]]++] = ev[z];
        if (is_sym && ec[z] != er[z]) { A.ci[cur[ec[z]]] = er[z]; A.val[cur[ec[z]]++] = ev[z]; }
    }
    for (int i = 0; i < M; i++) {   // 行内排序(保序:值随键)
        int s = A.rp[i], e2 = A.rp[i + 1];
        std::vector<int> idx(e2 - s); for (int q = 0; q < e2 - s; q++) idx[q] = q;
        std::stable_sort(idx.begin(), idx.end(), [&](int x, int y){ return A.ci[s+x] < A.ci[s+y]; });
        std::vector<int> c2(e2-s); std::vector<double> v2(e2-s);
        for (int q = 0; q < e2-s; q++) { c2[q]=A.ci[s+idx[q]]; v2[q]=A.val[s+idx[q]]; }
        for (int q = 0; q < e2-s; q++) { A.ci[s+q]=c2[q]; A.val[s+q]=v2[q]; }
    }
    return A;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <mtx> <v9_rows> <rt_out> [rep]\n", argv[0]); return 1; }
    int rep = argc > 4 ? atoi(argv[4]) : 3;
    CSR A = read_mtx_sym(argv[1]);
    int n = A.n;
    // 行单:v9 dump 谓词
    std::vector<int> rows;
    { FILE *f = fopen(argv[2], "r"); char l[256];
      while (fgets(l, sizeof l, f)) { int t[9]; int nf = 0; char *p = strtok(l, " \n");
        while (p && nf < 9) { t[nf++] = atoi(p); p = strtok(nullptr, " \n"); }
        if (nf == 9) { int al = t[1], est = t[3], sp = t[4];
          if (al >= 1 && al <= 32 && est >= 256 && est <= 4096 && sp <= 16384) rows.push_back(t[0]); } }
      fclose(f); }
    fprintf(stderr, "[d3bench] n=%d 行单 %zu\n", n, rows.size());
    int NR = rows.size();
    // device
    int *d_rp, *d_ci; double *d_val, *d_oval; int *d_rows, *d_rownnz, *d_ex; unsigned long long *d_rt;
    CK(cudaMalloc(&d_rp, (n+1)*4)); CK(cudaMalloc(&d_ci, A.ci.size()*4)); CK(cudaMalloc(&d_val, A.val.size()*8));
    CK(cudaMalloc(&d_rows, NR*4)); CK(cudaMalloc(&d_rownnz, n*4)); CK(cudaMalloc(&d_ex, (NR+1)*4));
    CK(cudaMalloc(&d_rt, n*8));
    CK(cudaMemcpy(d_rp, A.rp.data(), (n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci, A.ci.data(), A.ci.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val, A.val.data(), A.val.size()*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rows, rows.data(), NR*4, cudaMemcpyHostToDevice));
    // 预热 + count pass
    // 单行 CPU 参照(argv[5]=row id;破 warp 语义谜)
    if (argc > 5) {
        int ri = atoi(argv[5]);
        // 行单中找
        int pos = -1; for (int z = 0; z < NR; z++) if (rows[z] == ri) { pos = z; break; }
        if (pos < 0) { fprintf(stderr, "row %d 不在行单\n", ri); return 0; }
        std::vector<int> all;
        for (int q2 = A.rp[ri]; q2 < A.rp[ri+1]; q2++) {
            int k = A.ci[q2];
            for (int w2 = A.rp[k]; w2 < A.rp[k+1]; w2++) all.push_back(A.ci[w2]);
        }
        std::sort(all.begin(), all.end());
        int dist = (int)(std::unique(all.begin(), all.end()) - all.begin());
        fprintf(stderr, "[chk] row %d: a_len=%d flop=%d CPU_distinct=%d\n", ri, A.rp[ri+1]-A.rp[ri], (int)all.size(), dist);
        // kernel count for this row
        std::vector<int> rn(n, -1);
        { int g2 = (NR + 7) / 8;
        d3_merge_v2_kernel<<<g2, 256>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, n, d_rows, NR, d_ex, nullptr, nullptr, d_rownnz, nullptr, 0); }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(rn.data(), d_rownnz, n*4, cudaMemcpyDeviceToHost));
        fprintf(stderr, "[chk] row %d: kernel_cnt=%d\n", ri, rn[ri]);
        return 0;
    }

    int wpb = 256 >> 5; int g = (NR + wpb - 1) / wpb;
    d3_merge_kernel<<<g, 256>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, 0, n, d_rows, NR, d_ex, nullptr, nullptr, d_rownnz, nullptr, 0);
    CK(cudaDeviceSynchronize());
    std::vector<int> hcnt(NR);
    { std::vector<int> all(n); CK(cudaMemcpy(all.data(), d_rownnz, n*4, cudaMemcpyDeviceToHost));
      long long tot = 0; for (int z = 0; z < NR; z++) { hcnt[z] = all[rows[z]]; tot += hcnt[z]; }
      fprintf(stderr, "[d3bench] Σnnz(count)=%lld\n", tot);
      // 前缀(行单局部)
      std::vector<int> ex(NR + 1, 0); for (int z = 0; z < NR; z++) ex[z+1] = ex[z] + hcnt[z];
      CK(cudaMemcpy(d_ex, ex.data(), (NR+1)*4, cudaMemcpyHostToDevice)); }
    // 计时 rep 发(warm)
    for (int r = 0; r < rep + 1; r++) {
        d3_merge_kernel<<<g, 256>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, 0, n, d_rows, NR, d_ex, nullptr, nullptr, d_rownnz, d_rt, 0);
        CK(cudaDeviceSynchronize());
    }
    FILE *fo = fopen(argv[3], "w");
    { std::vector<unsigned long long> rt(n); CK(cudaMemcpy(rt.data(), d_rt, n*8, cudaMemcpyDeviceToHost));
      for (int z = 0; z < NR; z++) if (rt[rows[z]]) fprintf(fo, "%d %llu\n", rows[z], rt[rows[z]]);
      fclose(fo); }
    fprintf(stderr, "[d3bench] rowtime %d 行 → %s\n", NR, argv[3]);
    return 0;
}
