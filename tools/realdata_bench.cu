// E7-终章:真数据独立重放(study/57 §四十一)——真 Bump 字节,无管线无估计器,
//   两核背靠背。回答:harness 罚金(其核 22.2@家 vs 34.4@我处)住在哪。
//   输入:/tmp/bump_{rp,ci,val,rows,est,flop}.npy(np 标准头,小端)
#include "../src/spgemm_kernel_hash.cu"
#include "../src/ocean_hash_port.cuh"
// 终极对照:直接编译他们原始内核头(非移植)——排除移植差异的最后一丝可能
namespace theirs { 
#define NUM_HASH_INPLACE_SORT_THRESHOLD 673
#include "../ocean/kernels/AccumulatorHash.cuh"
}
#include <cstdio>
#include <vector>
#include <cstring>

#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

// 极简 npy 读取(假设 header 描述与实际一致,int32/float64 一维)
static void* npy_load(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("无 %s\n", path); exit(1); }
    char head[10]; fread(head, 1, 10, f);
    unsigned short hlen; memcpy(&hlen, head + 8, 2);      // little-endian header len
    char hdr[256]; fread(hdr, 1, hlen < 250 ? hlen : 250, f);
    hdr[hlen < 250 ? hlen : 250] = 0;
    int esize = strstr(hdr, "f8") ? 8 : 4;
    long datastart = 10 + hlen;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, datastart, SEEK_SET);
    long data = sz - datastart;
    void* p = malloc(data);
    if (fread(p, 1, data, f) != (size_t)data) { printf("读 %s 短\n", path); exit(1); }
    fclose(f);
    *n = data / esize;
    return p;
}

int main() {
    size_t nr, nc, nv, nrows, nest, nflop, nmb, ns1, ns2, ne1, ne2, nd;
    int* h_dn = (int*)npy_load("/tmp/bump_d.npy", &nd);
    int* h_rows_small = (int*)npy_load("/tmp/bump_rows_small.npy", &ns1);
    int* h_rows_big   = (int*)npy_load("/tmp/bump_rows_big.npy", &ns2);
    int* h_est_small  = (int*)npy_load("/tmp/bump_est_small.npy", &ne1);
    int* h_est_big    = (int*)npy_load("/tmp/bump_est_big.npy", &ne2);
    int* h_mb = (int*)npy_load("/tmp/bump_maxbl.npy", &nmb);
    int* h_rp = (int*)npy_load("/tmp/bump_rp.npy", &nr);
    int* h_ci = (int*)npy_load("/tmp/bump_ci.npy", &nc);
    double* h_val = (double*)npy_load("/tmp/bump_val.npy", &nv);
    int* h_rows = (int*)npy_load("/tmp/bump_rows.npy", &nrows);
    int* h_est = (int*)npy_load("/tmp/bump_est.npy", &nest);
    int* h_flop = (int*)npy_load("/tmp/bump_flop.npy", &nflop);
    int ADIM = 2911419;
    printf("真数据:%zu nnz,%zu 行(bin4+5)\n", nc, nrows);
    int N = (int)nrows;
    std::vector<long long> off(ADIM + 1, 0), dense_off(ADIM + 1, 0);
    long long dtot = 0; { std::vector<long long> dp(ADIM,-1); for(int i=0;i<N;i++){dp[h_rows[i]]=h_dn[i]+8;} for(int r=0;r<ADIM;r++) if(dp[r]>0) dtot+=dp[r]; }
// 按行号全长(核按 my_row 索引——此前按人口序是 bug,两侧核同染,已修)
    long long tot = 0;
    { std::vector<long long> pos(ADIM, -1);
      for (int i = 0; i < N; i++) pos[h_rows[i]] = tot, tot += (long long)h_est[i] * 2 + 64;
      for (int r = 0; r < ADIM; r++) off[r] = pos[r] < 0 ? 0 : pos[r];
      { std::vector<long long> p2(ADIM,-1); long long t2=0;
        for (int i=0;i<N;i++){ p2[h_rows[i]]=t2; t2+=h_dn[i]+8; }
        for (int r=0;r<ADIM;r++) dense_off[r] = p2[r]<0?0:p2[r]; dense_off[ADIM]=t2; }
      off[ADIM] = tot; }
    printf("tmp 槽总额:稀疏 %.2f G / 致密 %.2f G\n", tot / 1e9, dtot / 1e9);
    long long *d_off_dense; CHK(cudaMalloc(&d_off_dense,(ADIM+1)*8)); CHK(cudaMemcpy(d_off_dense,dense_off.data(),(ADIM+1)*8,cudaMemcpyHostToDevice));


    int *d_rp,*d_ci,*d_rows,*d_rnz,*d_ovf,*d_ovfr,*d_ovfc,*d_rovf,*d_flop,*d_maxbl;
    double *d_val,*d_tval; unsigned long long *d_tkey; long long *d_off;
    CHK(cudaMalloc(&d_rp,(ADIM+1)*4)); CHK(cudaMalloc(&d_ci,nc*4)); CHK(cudaMalloc(&d_val,nv*8));
    CHK(cudaMalloc(&d_rows,N*4)); CHK(cudaMalloc(&d_rnz,ADIM*4)); CHK(cudaMalloc(&d_ovf,4));
    CHK(cudaMalloc(&d_ovfr,ADIM*4)); CHK(cudaMalloc(&d_ovfc,4)); CHK(cudaMalloc(&d_rovf,ADIM*4));
    CHK(cudaMalloc(&d_flop,ADIM*4)); CHK(cudaMalloc(&d_maxbl,ADIM*4));
    CHK(cudaMalloc(&d_tkey,(size_t)tot*8)); CHK(cudaMalloc(&d_tval,(size_t)tot*8));
    CHK(cudaMalloc(&d_off,(ADIM+1)*8));
    CHK(cudaMemcpy(d_rp,h_rp,(ADIM+1)*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_ci,h_ci,nc*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_val,h_val,nv*8,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rows,h_rows,N*4,cudaMemcpyHostToDevice));
    { int* ff=(int*)calloc(ADIM,4); int* mm=(int*)calloc(ADIM,4);
  for(int i=0;i<N;i++){ff[h_rows[i]]=h_flop[i]; mm[h_rows[i]]=h_mb[i];}
  CHK(cudaMemcpy(d_flop,ff,ADIM*4,cudaMemcpyHostToDevice)); CHK(cudaMemcpy(d_maxbl,mm,ADIM*4,cudaMemcpyHostToDevice)); }
    CHK(cudaMemcpy(d_maxbl,h_mb,N*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_off,off.data(),(ADIM+1)*8,cudaMemcpyHostToDevice));
    cudaFuncSetAttribute(hash_spa_kernel<0,1,0,0,9,0,0,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    double gprod = 0; for (int i=0;i<N;i++) gprod += h_flop[i]; gprod /= 1e9;
    printf("总乘积 %.2f G\n", gprod);

    // 分层发射资源
    int *d_rs1,*d_rs2,*d_mb1,*d_mb2; long long *d_of1,*d_of2; int *d_fl1,*d_fl2;
    int N1=(int)ns1, N2=(int)ns2;
    CHK(cudaMalloc(&d_rs1,N1*4)); CHK(cudaMalloc(&d_rs2,N2*4));
    CHK(cudaMalloc(&d_of1,(ADIM+1)*8)); CHK(cudaMalloc(&d_of2,(ADIM+1)*8));
    CHK(cudaMalloc(&d_fl1,ADIM*4)); CHK(cudaMalloc(&d_fl2,ADIM*4));
    CHK(cudaMalloc(&d_mb1,ADIM*4)); CHK(cudaMalloc(&d_mb2,ADIM*4));
    CHK(cudaMemcpy(d_rs1,h_rows_small,N1*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rs2,h_rows_big,N2*4,cudaMemcpyHostToDevice));
    { std::vector<long long> o1(N1+1),o2(N2+1); long long t1=0,t2=0;
      for(int i=0;i<N1;i++){o1[i]=t1;t1+=(long long)h_est_small[i]*2+64;} o1[N1]=t1;
      for(int i=0;i<N2;i++){o2[i]=t2;t2+=(long long)h_est_big[i]*2+64;} o2[N2]=t2;
      CHK(cudaMemcpy(d_of1,off.data(),(ADIM+1)*8,cudaMemcpyHostToDevice));
      CHK(cudaMemcpy(d_of2,off.data(),(ADIM+1)*8,cudaMemcpyHostToDevice)); }
    CHK(cudaMemcpy(d_fl1,d_flop,ADIM*4,cudaMemcpyDeviceToDevice));
    CHK(cudaMemcpy(d_fl2,d_flop,ADIM*4,cudaMemcpyDeviceToDevice));
    CHK(cudaMemcpy(d_mb1,d_maxbl,ADIM*4,cudaMemcpyDeviceToDevice));
    CHK(cudaMemcpy(d_mb2,d_maxbl,ADIM*4,cudaMemcpyDeviceToDevice));
    { float best=1e9;
      for(int rep=0;rep<3;rep++){ CHK(cudaMemset(d_ovf,0,4)); CHK(cudaMemset(d_ovfc,0,4));
        cudaEventRecord(t0);
        ocean_hash_kernel<6,331,true,long long><<<N1,64,16+(size_t)332*4+(size_t)331*8+16>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_fl1,d_mb1,d_rs1,N1,d_of1,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
        ocean_hash_kernel<7,673,true,long long><<<N2,128,16+(size_t)674*4+(size_t)673*8+16>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_fl2,d_mb2,d_rs2,N2,d_of2,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
        cudaEventRecord(t1); CHK(cudaEventSynchronize(t1)); float ms; cudaEventElapsedTime(&ms,t0,t1); if(ms<best)best=ms; }
      printf("%-32s %8.1f ms  (%.2f ns/行)\n","ocn 分层 331@64T+673@128T(仿其家)",best,best*1e6/(N1+N2)); }
    struct Cfg { const char* name; int mode, lht, blk; size_t smem; } cfgs[] = {
        {"ours bin4核 全人口 ht512@256T",  1, 9,  256, (size_t)512*12+12*1024},
        {"ours bin5核 全人口 ht1024@256T", 1, 10, 256, (size_t)1024*12+12*1024},
        {"ocn 673@128T 全人口",            0, 7,  128, 16 + (size_t)674*4 + (size_t)673*8 + 16},
        {"ocn 673@128T 拆2发射(split2)",   2, 7,  128, 16 + (size_t)674*4 + (size_t)673*8 + 16},
        {"ocn 673@128T 致密槽",           3, 7,  128, 16 + (size_t)674*4 + (size_t)673*8 + 16},
        {"ours ht1024@256T 致密槽",       4, 10, 256, (size_t)1024*12+12*1024},
        {"★their 真身 673@128T(直编)", 5, 7, 128, 4096},
    };
    for (auto& c : cfgs) {
        float best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            CHK(cudaMemset(d_ovf,0,4)); CHK(cudaMemset(d_ovfc,0,4)); CHK(cudaMemset(d_rovf,0,N*4));
            cudaEventRecord(t0);
            if (c.mode == 2) {
                int half = N/2;
                ocean_hash_kernel<7,673,true,long long><<<half,c.blk,c.smem>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_flop,d_maxbl,d_rows,half,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
                ocean_hash_kernel<7,673,true,long long><<<N-half,c.blk,c.smem>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_flop,d_maxbl,d_rows+half,N-half,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
            } else if (c.mode == 3) {
                ocean_hash_kernel<7,673,true,long long><<<N,c.blk,c.smem>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_flop,d_maxbl,d_rows,N,d_off_dense,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
            } else if (c.mode == 4) {
                hash_spa_kernel<0,1,0,0,10,0,0,0><<<N,c.blk,c.smem>>>(d_rp,d_ci,d_val,d_rp,d_ci,d_val,0,d_rows,N,1024,d_off_dense,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf);
            } else if (c.mode == 5) {
                int *d_dummy; CHK(cudaMalloc(&d_dummy,16)); CHK(cudaMemset(d_dummy,0,16));
                theirs::somespgemm::hashNumericKernel<7,673,true,false,unsigned int><<<N,128,8192>>>(
                    (unsigned*)d_rp,(unsigned*)d_ci,d_val,ADIM,ADIM,(int)nc,
                    (unsigned*)d_rp,(unsigned*)d_ci,d_val,ADIM,ADIM,(int)nc,
                    (unsigned*)d_flop,(unsigned*)d_maxbl,(unsigned*)d_rows,(unsigned*)d_off,
                    (unsigned*)d_tkey,d_tval,(unsigned*)d_rnz,(unsigned*)d_dummy,(unsigned*)d_ovfr,(unsigned*)d_dummy,
                    nullptr,nullptr,0,0);
                CHK(cudaGetLastError()); CHK(cudaFree(d_dummy));
            } else if (c.mode) {
                if (c.lht==9)  hash_spa_kernel<0,1,0,0,9,0,0,0><<<N,c.blk,c.smem>>>(d_rp,d_ci,d_val,d_rp,d_ci,d_val,0,d_rows,N,512,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf);
                else           hash_spa_kernel<0,1,0,0,10,0,0,0><<<N,c.blk,c.smem>>>(d_rp,d_ci,d_val,d_rp,d_ci,d_val,0,d_rows,N,1024,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf);
            } else {
                ocean_hash_kernel<7,673,true,long long><<<N,c.blk,c.smem>>>(d_rp,d_ci,d_val,ADIM,ADIM,0,d_rp,d_ci,d_val,ADIM,ADIM,0,d_flop,d_maxbl,d_rows,N,d_off,d_tkey,d_tval,d_rnz,d_ovf,d_ovfr,d_ovfc,d_rovf,0);
            }
            cudaEventRecord(t1); CHK(cudaEventSynchronize(t1));
            float ms; cudaEventElapsedTime(&ms,t0,t1); if (ms<best) best=ms;
        }
        std::vector<int> rnz(N); CHK(cudaMemcpy(rnz.data(),d_rnz,N*4,cudaMemcpyDeviceToHost));
        long long s=0; int h0; CHK(cudaMemcpy(&h0,d_ovf,4,cudaMemcpyDeviceToHost));
        for(int r:rnz) s+=r;
        printf("%-32s %8.1f ms  (%.2f ns/行, %.1f Gprod/s)  nnzΣ=%lld ovf=%d\n",
               c.name, best, best*1e6/N, gprod/best*1000, s, h0);
    }
    return 0;
}