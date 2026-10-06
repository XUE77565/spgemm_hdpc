// kernel₇ 独立测量台(计划 v5 第二步):双 pass 顺序直写,dup≈1 人口专用。
// 机理:Pass1 全产品只 atomicOr 位图(32-bit 字原子,免 F64)→ popcount = 精确 nnz;
//      Pass2 重走产品流,atomicExch 认领(32-bit)→ 首写者普通写值到已知终位,
//      dup 败者 atomicAdd 补。输出按位序天然有序,零估计器依赖、零表、零 compact+sort。
// 对拍:同一批行 vs 主管线 HASH_ROWTIME 的逐行 cycles(host join)。
// 用法: k7_bench <mtx> <rep> [rowtime_out]
// 行单谓词(host 精确): a_len∈[1,32] ∧ span≤32768 ∧ distinct≥0.6×flop(dup≤1.67)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

#define K7_SPAN_MAX   32768          // 位图 span 上限(4KB mark + 4KB claim + cum)
#define K7_DUP_MIN    0.60           // distinct/flop 下限(=dup 上限 1.67)
#define K7_BLOCK      128            // 一行一 block
#define K7_FIX_MAX    512            // 每行 fixup 表容量(碰撞数=count 证书 flop−distinct)
#define K7_FIX_OFF    (512 + 1024 + 2)   // smem 字偏移:[fvl 1024w][frk 512w][cnt+pad 2w] 后是 mark(1538)

__global__ void k7_count_kernel(
    const int *A_rp, const int *A_ci,
    const int *B_rp, const int *B_ci,
    const int *rows, int n_rows,
    const int *span_words,           // 每行位图字数
    const int *col_base,             // 每行最小列(位图基)
    int *row_nnz)
{
    extern __shared__ unsigned bmp[];          // span_words 个字
    int i = blockIdx.x;
    if (i >= n_rows) return;
    int W = span_words[i];
    for (int w = threadIdx.x; w < W; w += blockDim.x) bmp[w] = 0u;
    __syncthreads();
    int rs = A_rp[rows[i]], re = A_rp[rows[i] + 1];
    int base = col_base[i];
    int ak = re - rs, tg = blockDim.x / (ak > 0 ? ak : 1); if (tg < 1) tg = 1;
    int k_idx = threadIdx.x / tg, t_in = threadIdx.x % tg;   // 2D 切分:a_len 小则全 block 摊 B 流
    if (k_idx < ak) {
        int j = A_ci[rs + k_idx];
        for (int p = B_rp[j] + t_in; p < B_rp[j + 1]; p += tg) {
            int c = B_ci[p] - base;
            atomicOr(&bmp[c >> 5], 1u << (c & 31));
        }
    }
    __syncthreads();
    unsigned cnt = 0;
    for (int w = threadIdx.x; w < W; w += blockDim.x) cnt += __popc(bmp[w]);
    // block 归约
    __shared__ unsigned red[K7_BLOCK];
    red[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_nnz[i] = (int)red[0];
}

// 正式认领-直写核(上面草稿的修正:认领 = atomicOr 返回旧值,败者 atomicAdd)
__global__ void k7_numeric2_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    const int *rows, int n_rows,
    const int *span_words, const int *col_base,
    const int *exact_rp,
    int *out_col, double *out_val,
    int *row_nnz,
    unsigned long long *rowtime, int mode)
{
    extern __shared__ unsigned smem[];
    int i = blockIdx.x;
    if (i >= n_rows) return;
    unsigned long long rt0 = rowtime ? clock64() : 0;
    int W = span_words[i];
    unsigned *mark = smem + K7_FIX_OFF, *claim = mark + W, *cum = mark + 2 * W;
    for (int w = threadIdx.x; w < W; w += blockDim.x) { mark[w] = 0u; claim[w] = 0u; }
    if (threadIdx.x == 0) *(smem + 1536) = 0u;                // fixup 计数器清零(fvl 后)
    __syncthreads();
    int rs = A_rp[rows[i]], re = A_rp[rows[i] + 1];
    int base = col_base[i];
    int ak = re - rs, tg = blockDim.x / (ak > 0 ? ak : 1); if (tg < 1) tg = 1;
    int k_idx = threadIdx.x / tg, t_in = threadIdx.x % tg;
    if (k_idx < ak) {
        int j = A_ci[rs + k_idx];
        for (int p = B_rp[j] + t_in; p < B_rp[j + 1]; p += tg) {
            int c = B_ci[p] - base;
            atomicOr(&mark[c >> 5], 1u << (c & 31));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int w = 0; w < W; w++) { cum[w] = acc; acc += __popc(mark[w]); }
        cum[W] = acc;
        if (row_nnz) row_nnz[i] = acc;
    }
    __syncthreads();
    int obase = exact_rp[i];
    if (mode == 3) {
        // 快道 v2(fixup 协议):认领→胜者普通写,败者入 SMEM 修复表;__syncthreads 后
        // 统一 atomicAdd —— 屏障保证块内全局写可见 ⇒ 败者的加必然落在胜者写之后(次序由
        // 屏障定,免 atomicExch 的"两原子次序仍任意"缺陷)。容量由 count 证书背书:
        // flop−distinct ≤ K7_FIX_MAX(host 行单已滤);溢出走 racy 兜底由值门兜底报错。
        // 布局 v5(修 10-06 重叠 bug):fvl 必须 1024 连续字且不与 mark(K7_FIX_OFF 起)
        // 重叠 —— 旧版 fvl=1024..2047 与 mark≥1538 重叠,pre2 中 dup 人口 slot≥257 踩位图
        // → rank 越界 → illegal instruction(wedge #8 根因);bloweybq 碰撞≈1 只用 slot0 故绿
        double *fvl = (double *)(smem);                       // words 0..1023(8B 对齐)
        int *frk = (int *)(smem + 1024);                      // words 1024..1535
        volatile int *fcnt = (volatile int *)(smem + 1536);
        if (k_idx < ak) {
            int j = A_ci[rs + k_idx];
            double a = A_val[rs + k_idx];
            for (int p = B_rp[j] + t_in; p < B_rp[j + 1]; p += tg) {
                int c = B_ci[p] - base;
                unsigned bit = 1u << (c & 31);
                int word = c >> 5;
                unsigned old = atomicOr(&claim[word], bit);
                int rank = (int)cum[word] + __popc(mark[word] & (bit - 1));
                double prod = a * B_val[p];
                if (!(old & bit)) {
                    out_col[obase + rank] = c + base;         // 胜者:纯普通写(免一切值原子)
                    out_val[obase + rank] = prod;
                } else {
                    int slot = atomicAdd((int *)fcnt, 1);
                    if (slot < K7_FIX_MAX) { frk[slot] = rank; fvl[slot] = prod; }
                    else atomicAdd(&out_val[obase + rank], prod);   // 证书外兜底(不应达)
                }
            }
        }
        __syncthreads();
        int nf = *fcnt;
        for (int s = threadIdx.x; s < nf && s < K7_FIX_MAX; s += blockDim.x)
            atomicAdd(&out_val[obase + frk[s]], fvl[s]);      // 屏障后:确定性追加
        __syncthreads();
    } else if (k_idx < ak) {
        int j = A_ci[rs + k_idx];
        double a = A_val[rs + k_idx];
        for (int p = B_rp[j] + t_in; p < B_rp[j + 1]; p += tg) {
            int c = B_ci[p] - base;
            unsigned bit = 1u << (c & 31);
            int word = c >> 5;
            unsigned old = atomicOr(&claim[word], bit);      // 认领:旧值无此位 = 首写
            int rank = (int)cum[word] + __popc(mark[word] & (bit - 1));
            double prod = a * B_val[p];
            if (mode) {
                if (!(old & bit)) {                          // 首写者:col 普通写;值 atomicExch
                    out_col[obase + rank] = c + base;        // 与败者 atomicAdd 同为原子,免丢更新
                    atomicExch(reinterpret_cast<unsigned long long*>(&out_val[obase + rank]), __double_as_longlong(prod));
                } else {                                     // dup 败者:原子补
                    atomicAdd(&out_val[obase + rank], prod);
                }
            }
        }
    }
    __syncthreads();
    if (rowtime && threadIdx.x == 0) rowtime[i] = clock64() - rt0;
}

// 简易 MTX 读取(对称镜像展开,与 d3_bench/主管线 reader 同语义)
struct CSR { int n; std::vector<int> rp, ci; std::vector<double> val; };
static CSR read_mtx_sym(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s fail\n", path); exit(1); }
    char line[512];
    fgets(line, sizeof line, f);
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
    if (argc < 3) { fprintf(stderr, "usage: %s <mtx> <rep> [rowtime_out] [--verify]\n", argv[0]); return 1; }
    int rep = atoi(argv[2]);
    bool verify = argc > 4 && !strcmp(argv[4], "--verify");
    CSR A = read_mtx_sym(argv[1]);
    int n = A.n;
    // host 精确行单:a_len≤32 ∧ span≤K7_SPAN_MAX ∧ distinct≥0.6×flop
    std::vector<int> rows, span_w, cbase, cnnz;
    for (int i = 0; i < n; i++) {
        int rs = A.rp[i], re = A.rp[i + 1];
        int al = re - rs;
        if (al < 1 || al > 32) continue;
        std::vector<int> all;
        for (int q = rs; q < re; q++) {
            int j = A.ci[q];
            for (int w2 = A.rp[j]; w2 < A.rp[j + 1]; w2++) all.push_back(A.ci[w2]);
        }
        if (all.empty()) continue;
        int mn = *min_element(all.begin(), all.end());
        int mx = *max_element(all.begin(), all.end());
        if (mx - mn + 1 > K7_SPAN_MAX) continue;
        std::sort(all.begin(), all.end());
        int dist = (int)(std::unique(all.begin(), all.end()) - all.begin());
        if ((double)dist < K7_DUP_MIN * (double)all.size()) continue;
        if ((int)all.size() - dist > K7_FIX_MAX && !getenv("K7_KEEP_DUP")) continue;   // 快道合法域:碰撞数 ≤ fixup 容量(count 证书)
        rows.push_back(i);
        span_w.push_back((mx - mn + 1 + 31) / 32);
        cbase.push_back(mn);
        cnnz.push_back(dist);   // 真值 nnz(verify 用)
    }
    int NR = rows.size();
    fprintf(stderr, "[k7bench] n=%d 行单 %zu(Σnnz=%lld)\n", n, rows.size(),
            (long long)std::accumulate(cnnz.begin(), cnnz.end(), 0));
    if (!NR) return 0;
    // device 分配
    int *d_rp, *d_ci; double *d_val; int *d_rows, *d_rownnz, *d_ex;
    int *d_sw, *d_cb; unsigned long long *d_rt;
    CK(cudaMalloc(&d_rp, (n+1)*4)); CK(cudaMalloc(&d_ci, A.ci.size()*4)); CK(cudaMalloc(&d_val, A.val.size()*8));
    CK(cudaMalloc(&d_rows, NR*4)); CK(cudaMalloc(&d_rownnz, NR*4)); CK(cudaMalloc(&d_ex, (NR+1)*4));
    CK(cudaMalloc(&d_sw, NR*4)); CK(cudaMalloc(&d_cb, NR*4)); CK(cudaMalloc(&d_rt, NR*8));
    long long totnnz = std::accumulate(cnnz.begin(), cnnz.end(), 0);
    int *d_ocol; double *d_ov2;
    CK(cudaMalloc(&d_ocol, totnnz*4)); CK(cudaMalloc(&d_ov2, totnnz*8));
    CK(cudaMemcpy(d_rp, A.rp.data(), (n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci, A.ci.data(), A.ci.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val, A.val.data(), A.val.size()*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rows, rows.data(), NR*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sw, span_w.data(), NR*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cb, cbase.data(), NR*4, cudaMemcpyHostToDevice));
    std::vector<int> ex(NR+1, 0); for (int z = 0; z < NR; z++) ex[z+1] = ex[z] + cnnz[z];
    CK(cudaMemcpy(d_ex, ex.data(), (NR+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_rt, 0, NR*8));
    int wmax = *std::max_element(span_w.begin(), span_w.end());
    size_t smem_cnt = wmax * 4;
    size_t smem_num = (size_t)(K7_FIX_OFF + 3 * wmax + 2) * 4;
    // count 核(计时 rep+1 发)
    std::vector<int> hnnz(NR);
    for (int r = 0; r < rep + 1; r++) {
        k7_count_kernel<<<NR, K7_BLOCK, smem_cnt>>>(d_rp, d_ci, d_rp, d_ci, d_rows, NR, d_sw, d_cb, d_rownnz);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpy(hnnz.data(), d_rownnz, NR*4, cudaMemcpyDeviceToHost));
    long long mism = 0; for (int z = 0; z < NR; z++) if (hnnz[z] != cnnz[z]) mism++;
    fprintf(stderr, "[k7bench] count:nnz 精确 %lld/%d%s\n", NR - mism, NR, mism ? " ✗MISMATCH" : " ✓");
    // numeric 计时(先 1 发 mode=0 热身再 rep 发)
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    k7_numeric2_kernel<<<NR, K7_BLOCK, smem_num>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, d_rows, NR, d_sw, d_cb, d_ex, d_ocol, d_ov2, nullptr, nullptr, 0);
    CK(cudaDeviceSynchronize());
    { float c0; CK(cudaEventRecord(t0));
      for (int q = 0; q < 3; q++) k7_count_kernel<<<NR, K7_BLOCK, smem_cnt>>>(d_rp, d_ci, d_rp, d_ci, d_rows, NR, d_sw, d_cb, d_rownnz);
      CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&c0, t0, t1));
      fprintf(stderr, "[k7bench] count: 3 发 %.3f ms(%.1f ns/行)\n", c0, c0 * 1e6 / 3 / NR); }
    float ms_best = 1e9f;
    for (int r = 0; r < rep; r++) {
        CK(cudaEventRecord(t0));
        k7_numeric2_kernel<<<NR, K7_BLOCK, smem_num>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, d_rows, NR, d_sw, d_cb, d_ex, d_ocol, d_ov2, nullptr, nullptr, 3);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < ms_best) ms_best = ms;
    }
    fprintf(stderr, "[k7bench] numeric: %d 行 best %.3f ms(%.1f ns/行)\n", NR, ms_best, ms_best*1e6/NR);
    // 值门:mode=1 一发 + host 逐行真值比对
    k7_numeric2_kernel<<<NR, K7_BLOCK, smem_num>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, d_rows, NR, d_sw, d_cb, d_ex, d_ocol, d_ov2, nullptr, nullptr, 3);
    CK(cudaDeviceSynchronize());
    std::vector<int> hcol(totnnz); std::vector<double> hval(totnnz);
    CK(cudaMemcpy(hcol.data(), d_ocol, totnnz*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hval.data(), d_ov2, totnnz*8, cudaMemcpyDeviceToHost));
    long long bad_pos = 0, bad_val = 0, bad_ord = 0;
    if (verify) {
        for (int z = 0; z < NR; z++) {
            int i = rows[z];
            std::vector<std::pair<int,double>> ref;
            for (int q = A.rp[i]; q < A.rp[i+1]; q++) {
                int j = A.ci[q]; double a = A.val[q];
                for (int w2 = A.rp[j]; w2 < A.rp[j+1]; w2++) ref.push_back({A.ci[w2], a * A.val[w2]});
            }
            std::sort(ref.begin(), ref.end());
            std::vector<std::pair<int,double>> out;
            for (int q2 = ex[z]; q2 < ex[z+1]; q2++) out.push_back({hcol[q2], hval[q2]});
            std::sort(out.begin(), out.end());
            // 归并真值
            std::vector<std::pair<int,double>> merged;
            for (auto &pr : ref) {
                if (!merged.empty() && merged.back().first == pr.first) merged.back().second += pr.second;
                else merged.push_back(pr);
            }
            if ((int)out.size() != (int)merged.size()) { bad_pos++; continue; }
            for (int q2 = 0; q2 < (int)out.size(); q2++) {
                if (out[q2].first != merged[q2].first) { bad_pos++; break; }
                double d = fabs(out[q2].second - merged[q2].second);
                if (d > 1e-9 * (1.0 + fabs(merged[q2].second))) { bad_val++; break; }
            }
            // 有序检查(输出应天然位序)
            for (int q2 = ex[z]+1; q2 < ex[z+1]; q2++) if (hcol[q2-1] >= hcol[q2]) { bad_ord++; break; }
        }
        fprintf(stderr, "[k7bench] 值门:列错 %lld / 值错 %lld / 无序 %lld %s\n", bad_pos, bad_val, bad_ord,
                (bad_pos||bad_val||bad_ord) ? "✗ FAIL" : "✓ PASS");
    }
    // rowtime 输出(argv[3])
    if (argc > 3) {
        k7_numeric2_kernel<<<NR, K7_BLOCK, smem_num>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_val, d_rows, NR, d_sw, d_cb, d_ex, d_ocol, d_ov2, nullptr, d_rt, 3);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> rt(NR);
        CK(cudaMemcpy(rt.data(), d_rt, NR*8, cudaMemcpyDeviceToHost));
        FILE *fo = fopen(argv[3], "w");
        for (int z = 0; z < NR; z++) fprintf(fo, "%d %llu\n", rows[z], rt[z]);
        fclose(fo);
        fprintf(stderr, "[k7bench] rowtime %d 行 → %s\n", NR, argv[3]);
    }
    return 0;
}
