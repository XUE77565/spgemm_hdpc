// A1 标定扫描台(study/55 A线 + study/57 §三十五):hash 核成本曲面
//   维度:d(行 distinct)梯度 × 配置(我方 ht×块宽 / ocean 表×块宽)
//   输出:CSV(kernel,ht,block,d_target,ms,mean_d,gprod_s,cyc_prod,ovf)
//   用法:编译见 scripts/build_calib.sh;运行直接出 /tmp/calib_ladder.csv
// 口径注:我方 extract2=0(legacy rank,与 ocean 的 O(n²) 同型——头对头公平);
//   生产 CS2 差异记 caveat;曲面相对形状服务行→配置分配(A3)。
#include "../src/spgemm_kernel_hash.cu"
#include "../src/ocean_hash_port.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static int NROWS = 400000, ADIM = 4000000, ALEN = 44;

int main() {
    srand(42);
    int *d_arp, *d_aci, *d_brp, *d_bci, *d_rows, *d_rnz, *d_ovf, *d_ovfr, *d_ovfc, *d_rovf, *d_flop, *d_maxbl;
    double *d_aval, *d_bval, *d_tval;
    unsigned long long *d_tkey; long long *d_off;
    CHK(cudaMalloc(&d_arp, ((long long)NROWS + 1) * 4)); CHK(cudaMalloc(&d_aci, (long long)NROWS * ALEN * 4));
    CHK(cudaMalloc(&d_aval, (long long)NROWS * ALEN * 8));
    CHK(cudaMalloc(&d_brp, ((long long)ADIM + 1) * 4)); CHK(cudaMalloc(&d_bci, (long long)ADIM * 50 * 4)); CHK(cudaMalloc(&d_bval, (long long)ADIM * 50 * 8));
    CHK(cudaMalloc(&d_rows, NROWS * 4)); CHK(cudaMalloc(&d_rnz, NROWS * 4));
    CHK(cudaMalloc(&d_ovf, 4)); CHK(cudaMalloc(&d_ovfr, NROWS * 4)); CHK(cudaMalloc(&d_ovfc, 4)); CHK(cudaMalloc(&d_rovf, NROWS * 4));
    CHK(cudaMalloc(&d_flop, NROWS * 4)); CHK(cudaMalloc(&d_maxbl, NROWS * 4));
    CHK(cudaMalloc(&d_tkey, (size_t)NROWS * 2100 * 8)); CHK(cudaMalloc(&d_tval, (size_t)NROWS * 2100 * 8));
    CHK(cudaMalloc(&d_off, ((long long)NROWS + 1) * 8));
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,0,0>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,10,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,11,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    FILE *csv = fopen("/tmp/calib_ladder.csv", "w");
    fprintf(csv, "kernel,ht,block,d_target,ms,mean_d,gprod_s,cyc_prod,ovf\n");

    const int DT[] = {64, 128, 180, 256, 384, 512, 768, 1024};
    for (int dt : DT) {
        int pool = (dt <= 512) ? dt : (int)(dt * 1.25);
        // ── 人口生成(a_len=44,B 行长 40-50,列池=pool)──
        std::vector<int> arp(NROWS + 1, 0), aci; std::vector<double> aval;
        std::vector<int> brp(ADIM + 1, 0), bci; std::vector<double> bval;
        std::vector<int> flop(NROWS), maxbl(NROWS, 50);
        for (int k = 0; k < ADIM; k++) {
            int len = 40 + rand() % 11; brp[k + 1] = brp[k] + len;
            for (int q = 0; q < len; q++) { bci.push_back(rand() % pool); bval.push_back((rand() % 1000) / 1000.0); }
        }
        for (int i = 0; i < NROWS; i++) {
            arp[i + 1] = arp[i] + ALEN;
            long long f = 0;
            for (int q = 0; q < ALEN; q++) { int k = rand() % ADIM; aci.push_back(k); aval.push_back((rand() % 1000) / 1000.0); f += brp[k + 1] - brp[k]; }
            flop[i] = (int)f;
        }
        CHK(cudaMemcpy(d_arp, arp.data(), (NROWS + 1) * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_aci, aci.data(), (size_t)aci.size() * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_aval, aval.data(), (size_t)aval.size() * 8, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_brp, brp.data(), (ADIM + 1) * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_bci, bci.data(), (size_t)bci.size() * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_bval, bval.data(), (size_t)bval.size() * 8, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_flop, flop.data(), NROWS * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_maxbl, maxbl.data(), NROWS * 4, cudaMemcpyHostToDevice));
        { std::vector<int> rows(NROWS); std::vector<long long> off(NROWS + 1);
          for (int i = 0; i < NROWS; i++) { rows[i] = i; off[i] = (long long)i * 2100; } off[NROWS] = (long long)NROWS * 2100;
          CHK(cudaMemcpy(d_rows, rows.data(), NROWS * 4, cudaMemcpyHostToDevice));
          CHK(cudaMemcpy(d_off, off.data(), (NROWS + 1) * 8, cudaMemcpyHostToDevice)); }
        const double tot_prod_g = (double)flop[0] * 0 + [&]{ double s = 0; for (int f : flop) s += f; return s; }() / 1e9;

        // ── 我方:<LOG_HT> × {128T, 256T} ──
        struct Ours { int lht, ht, blk; } ours[] = {{7,128,128},{7,128,256},{8,256,128},{8,256,256},{9,512,128},{9,512,256},{10,1024,256},{10,1024,512},{11,2048,256},{11,2048,512}};
        for (auto& c : ours) {
            size_t smem = (size_t)c.ht * 12 + 12 * 1024;
            float best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                CHK(cudaMemset(d_ovf, 0, 4)); CHK(cudaMemset(d_ovfc, 0, 4)); CHK(cudaMemset(d_rovf, 0, NROWS * 4));
                cudaEventRecord(t0);
                if (c.lht == 7)       hash_spa_kernel<0,1,0,0,7,0,0,0><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, c.ht, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
                else if (c.lht == 8)  hash_spa_kernel<0,1,0,0,8,0,0,0><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, c.ht, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
                else if (c.lht == 9)  hash_spa_kernel<0,1,0,0,9,0,0,0><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, c.ht, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
                else if (c.lht == 10) hash_spa_kernel<0,1,0,0,10,0,0,0><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, c.ht, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
                else                  hash_spa_kernel<0,1,0,0,11,0,0,0><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, c.ht, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
                cudaEventRecord(t1); CHK(cudaEventSynchronize(t1));
                float ms; cudaEventElapsedTime(&ms, t0, t1); if (ms < best) best = ms;
            }
            std::vector<int> rnz(NROWS); CHK(cudaMemcpy(rnz.data(), d_rnz, NROWS * 4, cudaMemcpyDeviceToHost));
            double md = 0; int ovf = 0; for (int r : rnz) { md += r; if (r >= c.ht) ovf++; } md /= NROWS;
            double cps = tot_prod_g / best * 1000.0;
            fprintf(csv, "ours,%d,%d,%d,%.3f,%.1f,%.1f,%.2f,%d\n", c.ht, c.blk, dt, best, md, cps, best * 1e-3 * 132 * 1.76e9 / (tot_prod_g * 1e9), ovf);
        }
        // ── ocean:<表> × {64T, 128T} ──
        struct Ocn { int ln, ht, blk; } ocn[] = {{6,331,64},{6,331,128},{7,673,64},{7,673,128},{8,1327,128}};
        for (auto& c : ocn) {
            size_t smem = 16 + (size_t)(((c.ht + 1) / 2) * 2) * 4 + (size_t)c.ht * 8 + 16;
            float best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                CHK(cudaMemset(d_ovf, 0, 4)); CHK(cudaMemset(d_ovfc, 0, 4)); CHK(cudaMemset(d_rovf, 0, NROWS * 4)); CHK(cudaMemset(d_rnz, 0, NROWS * 4));
                cudaEventRecord(t0);
                if (c.ln == 6)      ocean_hash_kernel<6, 331, true, long long><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, NROWS, ADIM, 0, d_brp, d_bci, d_bval, ADIM, ADIM, 0, d_flop, d_maxbl, d_rows, NROWS, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf, 0);
                else if (c.ln == 7) ocean_hash_kernel<7, 673, true, long long><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, NROWS, ADIM, 0, d_brp, d_bci, d_bval, ADIM, ADIM, 0, d_flop, d_maxbl, d_rows, NROWS, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf, 0);
                else                ocean_hash_kernel<8, 1327, true, long long><<<NROWS, c.blk, smem>>>(d_arp, d_aci, d_aval, NROWS, ADIM, 0, d_brp, d_bci, d_bval, ADIM, ADIM, 0, d_flop, d_maxbl, d_rows, NROWS, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf, 0);
                cudaEventRecord(t1); CHK(cudaEventSynchronize(t1));
                float ms; cudaEventElapsedTime(&ms, t0, t1); if (ms < best) best = ms;
            }
            std::vector<int> rnz(NROWS); CHK(cudaMemcpy(rnz.data(), d_rnz, NROWS * 4, cudaMemcpyDeviceToHost));
            double md = 0; int ovf = 0; for (int r : rnz) { md += r; if (r >= c.ht) ovf++; } md /= NROWS;
            double cps = tot_prod_g / best * 1000.0;
            fprintf(csv, "ocn,%d,%d,%d,%.3f,%.1f,%.1f,%.2f,%d\n", c.ht, c.blk, dt, best, md, cps, best * 1e-3 * 132 * 1.76e9 / (tot_prod_g * 1e9), ovf);
        }
        fflush(csv);
        printf("d_target=%d 完成\n", dt);
    }
    fclose(csv);
    printf("→ /tmp/calib_ladder.csv\n");
    return 0;
}