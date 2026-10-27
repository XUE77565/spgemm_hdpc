// A2 真实行重放头对头(study/57 §三十七):bin5 带 KERNEL-vs-KERNEL 终裁
//   人口:从 /tmp/rt_bump_full.txt 采样真实 bin5 行的 (a_len, flop, d) 三元组
//   B 行:长度=真实均值,列池按 Newton 解 d=pool(1-e^{-P/pool}) 逐行配
//   配置:ours{1024@256(生产),2048@512(A3α),2048@256} × ocn{673@128,1327@128}
//   判决:同人口同内存,谁快——bin5 的 kernel 有无罪
#include "../src/spgemm_kernel_hash.cu"
#include "../src/ocean_hash_port.cuh"
#include <cstdio>
#include <vector>
#include <cmath>
#include <numeric>

#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    const char* spec = (argc > 1) ? argv[1] : "/tmp/rt_bump_full.txt";
    int want_bin = (argc > 2) ? atoi(argv[2]) : 5;
    int NMAX = (argc > 3) ? atoi(argv[3]) : 400000;
    // ── 读真实行:cols = r a_len flop est span ovf rt c_nnz bkid ──
    FILE* f = fopen(spec, "r");
    if (!f) { printf("无 %s\n", spec); return 1; }
    std::vector<int> r_alen, r_d; long long flsum = 0; int adim = 0;
    int r_, al, fl, es, sp, ov, cn, bk; unsigned long long rt; int cr, bk2;
    long long cnt = 0;
    while (fscanf(f, "%d %d %d %d %d %d %llu %d %d", &r_, &al, &fl, &es, &sp, &ov, &rt, &cn, &bk2) == 9) {
        cnt = r_ + 1; adim = cnt;
        if (bk2 == want_bin && al > 0 && cn > 0 && NMAX > (int)r_alen.size()) { r_alen.push_back(al); r_d.push_back(cn); flsum += fl; }
    }
    fclose(f);
    int N = r_alen.size();
    if (!N) { printf("bin%d 无行\n", want_bin); return 1; }
    double mean_blen = (double)flsum / N / 44.0;   // B 行长近似(flop/(a_len≈44))
    printf("回放人口:bin%d %d 行,均值 d=%.0f,B行≈%.1f\n", want_bin, N,
           (double)std::accumulate(r_d.begin(), r_d.end(), 0LL)/N, mean_blen);
    // ── 生成:A 行(a_len 真实),B 行(长=mean_blen,池按行解)──
    srand(42);
    std::vector<int> arp(N + 1, 0), aci; std::vector<double> aval;
    std::vector<int> brp(adim + 1, 0), bci; std::vector<double> bval;
    std::vector<int> flop(N), maxbl(N, (int)mean_blen + 1);
    int BL = (int)mean_blen + 1;
    for (int k = 0; k < adim; k++) { brp[k + 1] = brp[k] + BL; for (int q = 0; q < BL; q++) { bci.push_back(rand() % 4000000); bval.push_back((rand()%1000)/1000.0); } }
    for (int i = 0; i < N; i++) {
        arp[i + 1] = arp[i] + r_alen[i]; flop[i] = r_alen[i] * BL;
        double P = flop[i], dt = r_d[i];           // Newton 解 pool
        double lo = dt * 0.5, hi = dt * 12;
        for (int it = 0; it < 40; it++) { double mid = (lo + hi) / 2; double dd = mid * (1 - exp(-P / mid)); if (dd < dt) lo = mid; else hi = mid; }
        int pool = (int)((lo + hi) / 2) + 1;
        for (int q = 0; q < r_alen[i]; q++) { int k = rand() % adim; aci.push_back(k); aval.push_back((rand()%1000)/1000.0); }
        // 该行的 B 内容按池重造:把该行引用的 k 的 B 行改为池内列 —— 简化:全局 B 已随机,
        // 池约束改为行内 j 域过滤:直接放弃逐行池(复杂),改全局池=均值 d×k'
        (void)pool;
    }
    // 简化口径:全局池(所有行共用)按均值 d 解一次 —— 与真实 d 分布有差,但对拍两核公平(同数据)
    { double P = (double)44 * BL, dt = (double)std::accumulate(r_d.begin(), r_d.end(), 0LL) / N;
      double lo = dt*0.5, hi = dt*12;
      for (int it = 0; it < 40; it++) { double mid = (lo+hi)/2; double dd = mid*(1-exp(-P/mid)); if (dd < dt) lo = mid; else hi = mid; }
      int pool = (int)((lo+hi)/2) + 1;
      for (auto& c : bci) c = rand() % pool;
      printf("全局池=%d(目标均值 d=%.0f)\n", pool, dt); }
    // ── 上卡 ──
    int *d_arp,*d_aci,*d_brp,*d_bci,*d_rows,*d_rnz,*d_ovf,*d_ovfr,*d_ovfc,*d_rovf,*d_flop,*d_maxbl;
    double *d_aval,*d_bval,*d_tval; unsigned long long *d_tkey; long long *d_off;
    size_t ANZ = aci.size(), BNZ = bci.size();
    CHK(cudaMalloc(&d_arp,(N+1)*4)); CHK(cudaMalloc(&d_aci,ANZ*4)); CHK(cudaMalloc(&d_aval,ANZ*8));
    CHK(cudaMalloc(&d_brp,(adim+1)*4)); CHK(cudaMalloc(&d_bci,BNZ*4)); CHK(cudaMalloc(&d_bval,BNZ*8));
    CHK(cudaMalloc(&d_rows,N*4)); int *d_rows_rev; CHK(cudaMalloc(&d_rows_rev,N*4));
    { std::vector<int> rv(N); for(int i=0;i<N;i++) rv[i]=N-1-i; CHK(cudaMemcpy(d_rows_rev,rv.data(),N*4,cudaMemcpyHostToDevice)); } CHK(cudaMalloc(&d_rnz,N*4)); CHK(cudaMalloc(&d_ovf,4));
    CHK(cudaMalloc(&d_ovfr,N*4)); CHK(cudaMalloc(&d_ovfc,4)); CHK(cudaMalloc(&d_rovf,N*4));
    CHK(cudaMalloc(&d_flop,N*4)); CHK(cudaMalloc(&d_maxbl,N*4));
    long long SLOT = 4200; CHK(cudaMalloc(&d_tkey,(size_t)N*SLOT*8)); CHK(cudaMalloc(&d_tval,(size_t)N*SLOT*8));
    CHK(cudaMalloc(&d_off,(N+1)*8));
    CHK(cudaMemcpy(d_arp,arp.data(),(N+1)*4,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_aci,aci.data(),ANZ*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_aval,aval.data(),ANZ*8,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_brp,brp.data(),(adim+1)*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_bci,bci.data(),BNZ*4,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_bval,bval.data(),BNZ*8,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_flop,flop.data(),N*4,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_maxbl,maxbl.data(),N*4,cudaMemcpyHostToDevice));
    { std::vector<int> rows(N); std::vector<long long> off(N+1);
      for (int i=0;i<N;i++){rows[i]=i; off[i]=(long long)i*SLOT;} off[N]=(long long)N*SLOT;
      CHK(cudaMemcpy(d_rows,rows.data(),N*4,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_off,off.data(),(N+1)*8,cudaMemcpyHostToDevice)); }
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,10,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,11,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    double gprod = (double)flsum / N * N / 1e9; // ≈ Σflop… flop 数组本身即 Σ
    { double s=0; for(int i=0;i<N;i++) s+=flop[i]; gprod = s/1e9; }
    struct Cfg { const char* name; int ours, lht, ht, blk; } cfgs[] = {
        {"ours ht1024@256T(生产)", 1, 10, 1024, 256},
        {"ours ht2048@512T(A3α)",  1, 11, 2048, 512},
        {"ours ht2048@256T",       1, 11, 2048, 256},
        {"ocn 673@128T",           0, 7, 673, 128},
        {"ocn 1327@128T",          0, 8, 1327, 128},
        {"ours ht1024@256T 行反序", 2, 10, 1024, 256},
    };
    double ref_nnz = -1;
    for (auto& c : cfgs) {
        size_t smem = c.ours ? (size_t)c.ht*12 + 12*1024 : 16 + (size_t)(((c.ht+1)/2)*2)*4 + (size_t)c.ht*8 + 16;
        float best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            CHK(cudaMemset(d_ovf,0,4)); CHK(cudaMemset(d_ovfc,0,4)); CHK(cudaMemset(d_rovf,0,N*4));
            cudaEventRecord(t0);
            if (c.ours) {
                const int *rwp = (c.ours==2) ? d_rows_rev : d_rows;
                if (c.lht==10) hash_spa_kernel<0,1,0,0,10,0,0,0><<<N,c.blk,smem>>>(d_arp,d_aci,d_aval,d_brp,d_bci,d_bval,0,rwp,N,c.ht,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf);
                else          hash_spa_kernel<0,1,0,0,11,0,0,0><<<N,c.blk,smem>>>(d_arp,d_aci,d_aval,d_brp,d_bci,d_bval,0,d_rows,N,c.ht,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf);
            } else if (c.lht==7) ocean_hash_kernel<7,673,true,long long><<<N,c.blk,smem>>>(d_arp,d_aci,d_aval,N,adim,0,d_brp,d_bci,d_bval,adim,adim,0,d_flop,d_maxbl,d_rows,N,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
            else                ocean_hash_kernel<8,1327,true,long long><<<N,c.blk,smem>>>(d_arp,d_aci,d_aval,N,adim,0,d_brp,d_bci,d_bval,adim,adim,0,d_flop,d_maxbl,d_rows,N,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
            cudaEventRecord(t1); CHK(cudaEventSynchronize(t1));
            float ms; cudaEventElapsedTime(&ms,t0,t1); if (ms<best) best=ms;
        }
        std::vector<int> rnz(N); CHK(cudaMemcpy(rnz.data(),d_rnz,N*4,cudaMemcpyDeviceToHost));
        double s=0; int ovf=0; for(int r:rnz){s+=r;}
        int h_ovf=0; CHK(cudaMemcpy(&h_ovf,d_ovf,4,cudaMemcpyDeviceToHost)); ovf=h_ovf;
        if (ref_nnz<0) ref_nnz=s;
        printf("%-26s %8.2f ms  nnz=%lld %s ovf=%d\n", c.name, best, (long long)s, (fabs(s-ref_nnz)<1)?"✓":"✗", ovf);
    }
    return 0;
}