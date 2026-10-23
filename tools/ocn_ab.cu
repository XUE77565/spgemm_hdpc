// 刀E6b:独立微基准——我方 hash_spa vs ocean 字面移植,同数据同内存背靠背(study/57 §三十三)
// 编译:nvcc -O3 -arch=sm_90 -std=c++14 -I src -I include tools/ocn_ab.cu -o /tmp/ocn_ab
// 无管线、无 est、无共驻:合成 Bump 形态行(a_len=44,B 行长 45,dup~10×),只比 kernel 本体。
#include "../src/spgemm_kernel_hash.cu"      // 我方 hash_spa(全依赖随附)
#include "../src/ocean_hash_port.cuh"
#include <cstdio>

// 链接桩:全部由真实 TU 提供(mempool.cu / kernel_struct.cu),此处不再重复定义

#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main() {
    // ── 合成 Bump-形态人口:N=855k 行,a_len=44,B 方阵 4M 行,行长 ~45,值随机 ──
    const int NROWS = 855000, ADIM = 4000000, ALEN = 44;
    srand(42);
    std::vector<int> h_arp(NROWS + 1), h_aci; std::vector<double> h_aval;
    // B:固定行长带(40-50 随机,cols 随机 → dup 高)
    std::vector<int> h_brp(ADIM + 1); std::vector<int> h_bci; std::vector<double> h_bval;
    h_brp[0] = 0;
    for (int k = 0; k < ADIM; k++) {
        int len = 40 + rand() % 11;
        h_brp[k + 1] = h_brp[k] + len;
        for (int q = 0; q < len; q++) { h_bci.push_back(rand() % 180); h_bval.push_back((rand() % 1000) / 1000.0); }
    }
    h_arp[0] = 0;
    for (int i = 0; i < NROWS; i++) {
        h_arp[i + 1] = h_arp[i] + ALEN;
        for (int q = 0; q < ALEN; q++) { h_aci.push_back(rand() % ADIM); h_aval.push_back((rand() % 1000) / 1000.0); }
    }
    const int ANZ = h_arp[NROWS], BNZ = h_bci.size();
    int *d_arp, *d_aci, *d_brp, *d_bci, *d_rows, *d_rnz, *d_ovf, *d_ovfr, *d_ovfc, *d_rovf, *d_flop, *d_maxbl;
    double *d_aval, *d_bval, *d_tval;
    unsigned long long *d_tkey; long long *d_off;
    CHK(cudaMalloc(&d_arp, h_arp.size() * 4)); CHK(cudaMalloc(&d_aci, ANZ * 4)); CHK(cudaMalloc(&d_aval, ANZ * 8));
    CHK(cudaMalloc(&d_brp, h_brp.size() * 4)); CHK(cudaMalloc(&d_bci, BNZ * 4)); CHK(cudaMalloc(&d_bval, BNZ * 8));
    CHK(cudaMalloc(&d_rows, NROWS * 4)); CHK(cudaMalloc(&d_rnz, NROWS * 4));
    CHK(cudaMalloc(&d_ovf, 4)); CHK(cudaMalloc(&d_ovfr, NROWS * 4)); CHK(cudaMalloc(&d_ovfc, 4)); CHK(cudaMalloc(&d_rovf, NROWS * 4));
    CHK(cudaMalloc(&d_flop, NROWS * 4)); CHK(cudaMalloc(&d_maxbl, NROWS * 4));
    CHK(cudaMalloc(&d_tkey, (size_t)NROWS * 600 * 8)); CHK(cudaMalloc(&d_tval, (size_t)NROWS * 600 * 8));
    CHK(cudaMalloc(&d_off, (NROWS + 1) * 8));
    CHK(cudaMemcpy(d_arp, h_arp.data(), h_arp.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_aci, h_aci.data(), ANZ * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_aval, h_aval.data(), ANZ * 8, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_brp, h_brp.data(), h_brp.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_bci, h_bci.data(), BNZ * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_bval, h_bval.data(), BNZ * 8, cudaMemcpyHostToDevice));
    // rows = 0..N-1;off = est 槽 600/行;d_flop 近似精确(a_len×45);maxbl=50
    {
        std::vector<int> rows(NROWS), fl(NROWS), mb(NROWS, 50); std::vector<long long> off(NROWS + 1);
        for (int i = 0; i < NROWS; i++) { rows[i] = i; fl[i] = ALEN * 45; off[i] = (long long)i * 600; }
        off[NROWS] = (long long)NROWS * 600;
        CHK(cudaMemcpy(d_rows, rows.data(), NROWS * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_flop, fl.data(), NROWS * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_maxbl, mb.data(), NROWS * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_off, off.data(), (NROWS + 1) * 8, cudaMemcpyHostToDevice));
    }
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 24 * 1024);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;
    const int HT = 512;
    size_t smem_ours = (size_t)HT * 12 + 3 * 1024 * 4;   // bin4 形态:表+hist(生产同款)
    size_t smem_ocn  = 4 * 4 + (size_t)(((673 + 1) / 2 * 2)) * 4 + (size_t)673 * 8 + 16;

    // 预热+正确性:双方先各跑一遍收 row_nnz
    CHK(cudaMemset(d_ovf, 0, 4)); CHK(cudaMemset(d_ovfc, 0, 4)); CHK(cudaMemset(d_rovf, 0, NROWS * 4));
    hash_spa_kernel<0,1,0,0,9,0,0,0><<<NROWS, 256, smem_ours>>>(
        d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, HT, d_off,
        d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
    CHK(cudaGetLastError()); CHK(cudaDeviceSynchronize());
    std::vector<int> rnz_ours(NROWS); CHK(cudaMemcpy(rnz_ours.data(), d_rnz, NROWS * 4, cudaMemcpyDeviceToHost));
    long long s1 = 0; for (int r : rnz_ours) s1 += r;

    CHK(cudaMemset(d_ovf, 0, 4)); CHK(cudaMemset(d_ovfc, 0, 4)); CHK(cudaMemset(d_rovf, 0, NROWS * 4));
    CHK(cudaMemset(d_rnz, 0, NROWS * 4));
    ocean_hash_kernel<7, 673, true, long long><<<NROWS, 128, smem_ocn>>>(
        d_arp, d_aci, d_aval, NROWS, ADIM, 0, d_brp, d_bci, d_bval, ADIM, ADIM, 0,
        d_flop, d_maxbl, d_rows, NROWS, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf, 0);
    CHK(cudaGetLastError()); CHK(cudaDeviceSynchronize());
    std::vector<int> rnz_ocn(NROWS); CHK(cudaMemcpy(rnz_ocn.data(), d_rnz, NROWS * 4, cudaMemcpyDeviceToHost));
    long long s2 = 0; for (int r : rnz_ocn) s2 += r;
    int mism = 0; for (int i = 0; i < NROWS; i++) if (rnz_ours[i] != rnz_ocn[i]) mism++;
    printf("nnz校验: ours=%lld ocn=%lld mismatch行=%d %s\n", s1, s2, mism, (s1 == s2 && !mism) ? "PASS" : "FAIL");

    // 计时:交错 ×3
    float best_o = 1e9, best_c = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        hash_spa_kernel<0,1,0,0,9,0,0,0><<<NROWS, 256, smem_ours>>>(
            d_arp, d_aci, d_aval, d_brp, d_bci, d_bval, 0, d_rows, NROWS, HT, d_off,
            d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf);
        cudaEventRecord(t1); CHK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1);
        if (ms < best_o) best_o = ms;
        cudaEventRecord(t0);
        ocean_hash_kernel<7, 673, true, long long><<<NROWS, 128, smem_ocn>>>(
            d_arp, d_aci, d_aval, NROWS, ADIM, 0, d_brp, d_bci, d_bval, ADIM, ADIM, 0,
            d_flop, d_maxbl, d_rows, NROWS, d_off, d_tkey, d_tval, d_rnz, d_ovf, d_ovfr, d_ovfc, d_rovf, 0);
        cudaEventRecord(t1); CHK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1);
        if (ms < best_c) best_c = ms;
    }
    printf("微基准(855k 行,a_len=44,B~45,dup 高,各自最优配置):\n");
    printf("  我方 hash_spa<9>  256T: %.2f ms\n", best_o);
    printf("  ocean <7,673>     128T: %.2f ms\n", best_c);
    printf("  比: %.2f×\n", best_o / best_c);
    return 0;
}