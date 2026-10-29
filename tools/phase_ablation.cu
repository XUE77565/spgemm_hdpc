// 相位消融实验(study/60 S3 前置):batched 核的 clear/insert/extract 三段占比
// 独立文件,不进生产路径。用法:./phase_ablation <mtx> [variant]
// variant: 0=全核(基线) 1=只清表 2=清+插入 3=清+提取(跳过插入)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>

#define HT 128
#define WPB 8
#define WARP 32

__global__ void ablate_kernel(
    const int *A_rp, const int *A_ci, const double *A_val,
    const int *B_rp, const int *B_ci, const double *B_val,
    const int *rows, int n_rows,
    long long *row_off, int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int variant)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    int i = rows[warp];
    __shared__ int smem[WPB][5 * HT];
    __shared__ int wcnt[WPB];
    int *t_col = smem[warp & (WPB-1)];
    double *t_val = (double*)(smem[warp & (WPB-1)] + HT);
    int *pack_col = smem[warp & (WPB-1)] + 3*HT;
    int *pack_slot = smem[warp & (WPB-1)] + 4*HT;

    // Phase 1: clear
    if (lane == 0) wcnt[warp & (WPB-1)] = 0;
    for (int s = lane; s < HT; s += 32) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();
    if (variant == 1) return;  // 只清表

    // Phase 2: hash insert
    if (variant != 3) {  // variant 3 skips insert
        int rs = A_rp[i], re = A_rp[i+1];
        for (int p = rs; p < re; p++) {
            int k = A_ci[p]; double a = A_val[p];
            int ks = B_rp[k], ke = B_rp[k+1];
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_ci[q]; double v = a * B_val[q];
                unsigned slot = ((unsigned)(j * 2654435761u)) & (HT-1);
                int probes = 0;
                while (true) {
                    int old = atomicCAS(&t_col[slot], -1, j);
                    if (old == -1 || old == j) { atomicAdd(&t_val[slot], v); break; }
                    slot = (slot + 1) & (HT-1);
                    if (++probes >= HT) break;
                }
            }
        }
    }
    __syncwarp();
    if (variant == 2) return;  // 清+插入,不提取

    // Phase 3: extract (pack + O(n²) rank + write)
    long long base = row_off[i];
    int cap = est[i];
    for (int s = lane; s < HT; s += 32) {
        int c = t_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&wcnt[warp & (WPB-1)], 1);
            if (pos < HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = wcnt[warp & (WPB-1)];
    if (n > HT) n = HT;
    for (int idx = lane; idx < n; idx += 32) {
        int c = pack_col[idx];
        int rank = 0;
        for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
        if (rank < cap) {
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = t_val[pack_slot[idx]];
        }
    }
    __syncwarp();
    if (lane == 0) { row_nnz[i] = (n <= cap) ? n : cap; }
}

// 简化 mtx 读取(CSR)
struct CSR { int n, nnz; int *rp, *ci; double *val; };
static CSR read_mtx(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    char line[256]; do { fgets(line, sizeof(line), f); } while (line[0] == '%');
    int nr, nc, nnz; sscanf(line, "%d %d %d", &nr, &nc, &nnz);
    CSR c; c.n = nr; c.nnz = nnz;
    c.rp = new int[nr+1]; c.ci = new int[nnz]; c.val = new double[nnz];
    int *coo_r = new int[nnz], *coo_c = new int[nnz]; double *coo_v = new double[nnz];
    for (int i = 0; i < nnz; i++) fscanf(f, "%d %d %lf", &coo_r[i], &coo_c[i], &coo_v[i]);
    fclose(f);
    memset(c.rp, 0, (nr+1)*sizeof(int));
    for (int i = 0; i < nnz; i++) c.rp[coo_r[i]]++;
    for (int i = 0, s = 0; i <= nr; i++) { int t = c.rp[i]; c.rp[i] = s; s += t; }
    int *pos = new int[nr]; memcpy(pos, c.rp, nr*sizeof(int));
    for (int i = 0; i < nnz; i++) { int r = coo_r[i]-1; c.ci[pos[r]] = coo_c[i]-1; c.val[pos[r]] = coo_v[i]; pos[r]++; }
    delete[] coo_r; delete[] coo_c; delete[] coo_v; delete[] pos;
    return c;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <mtx> [variant 0-3] [reps]\n", argv[0]); return 1; }
    CSR A = read_mtx(argv[1]);
    int variant = argc > 2 ? atoi(argv[2]) : 0;
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    int n = A.n;
    // Device copies
    int *d_rp, *d_ci; double *d_val, *d_Bval;
    cudaMalloc(&d_rp, (n+1)*sizeof(int)); cudaMemcpy(d_rp, A.rp, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_ci, A.nnz*sizeof(int)); cudaMemcpy(d_ci, A.ci, A.nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_val, A.nnz*sizeof(double)); cudaMemcpy(d_val, A.val, A.nnz*sizeof(double), cudaMemcpyHostToDevice);
    // B = A (self-product)
    d_Bval = d_val;
    // 行集(全部行)
    int *d_rows; cudaMalloc(&d_rows, n*sizeof(int));
    int *h_rows = new int[n]; for (int i = 0; i < n; i++) h_rows[i] = i;
    cudaMemcpy(d_rows, h_rows, n*sizeof(int), cudaMemcpyHostToDevice);
    // est/flop:粗算 = 每行 Σ B行长度
    int *h_est = new int[n]; long long *h_off = new long long[n+1]; h_off[0] = 0;
    for (int i = 0; i < n; i++) {
        int s = 0;
        for (int p = A.rp[i]; p < A.rp[i+1]; p++) s += A.rp[A.ci[p]+1] - A.rp[A.ci[p]];
        h_est[i] = s > 64 ? 64 : s; h_off[i+1] = h_off[i] + h_est[i];
    }
    long long tot_est = h_off[n];
    int *d_est; long long *d_off;
    cudaMalloc(&d_est, n*sizeof(int)); cudaMemcpy(d_est, h_est, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_off, (n+1)*sizeof(long long)); cudaMemcpy(d_off, h_off, (n+1)*sizeof(long long), cudaMemcpyHostToDevice);
    unsigned long long *d_tk; double *d_tv; int *d_rn;
    cudaMalloc(&d_tk, tot_est*sizeof(unsigned long long));
    cudaMalloc(&d_tv, tot_est*sizeof(double));
    cudaMalloc(&d_rn, n*sizeof(int));
    int block = WPB * WARP;
    int grid = (n + WPB - 1) / WPB;
    // warmup
    for (int w = 0; w < 3; w++) ablate_kernel<<<grid, block>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_Bval, d_rows, n, d_off, d_est, d_tk, d_tv, d_rn, variant);
    cudaDeviceSynchronize();
    // timing
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(start);
        ablate_kernel<<<grid, block>>>(d_rp, d_ci, d_val, d_rp, d_ci, d_Bval, d_rows, n, d_off, d_est, d_tk, d_tv, d_rn, variant);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }
    const char *names[] = {"全核(clear+insert+extract)", "只清表(clear)", "清+插入(clear+insert)", "清+提取(跳过插入)"};
    printf("variant=%d [%s]: %.3f ms (n=%d, nnz=%d)\n", variant, names[variant], best, n, A.nnz);
    return 0;
}
