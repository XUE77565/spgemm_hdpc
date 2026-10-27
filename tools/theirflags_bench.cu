// 终测:他们的核 × 他们的 flags(compute_61 PTX→sm_90 SASS + fast_math)× 我们的数据
// 编译:nvcc -std=c++17 -O3 -gencode=arch=compute_61,code=sm_90 -use_fast_math ... (其 Makefile 原样)
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace theirs {
#define NUM_HASH_INPLACE_SORT_THRESHOLD 673
#include "../ocean/kernels/AccumulatorHash.cuh"
}

#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static void* npy_load(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("无 %s\n", path); exit(1); }
    char head[10]; fread(head, 1, 10, f);
    unsigned short hlen; memcpy(&hlen, head + 8, 2);
    char hdr[256]; fread(hdr, 1, hlen < 250 ? hlen : 250, f);
    hdr[hlen < 250 ? hlen : 250] = 0;
    int esize = strstr(hdr, "f8") ? 8 : 4;
    long datastart = 10 + hlen;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, datastart, SEEK_SET);
    long data = sz - datastart;
    void* p = malloc(data);
    if (fread(p, 1, data, f) != (size_t)data) { printf("短读\n"); exit(1); }
    fclose(f);
    *n = data / esize;
    return p;
}

int main() {
    size_t nr, nc, nv, nrows, nest, nflop, nmb;
    unsigned* h_rp = (unsigned*)npy_load("/tmp/bump_rp.npy", &nr);
    unsigned* h_ci = (unsigned*)npy_load("/tmp/bump_ci.npy", &nc);
    double* h_val = (double*)npy_load("/tmp/bump_val.npy", &nv);
    unsigned* h_rows = (unsigned*)npy_load("/tmp/bump_rows.npy", &nrows);
    int* h_est = (int*)npy_load("/tmp/bump_est.npy", &nest);
    int* h_flop32 = (int*)npy_load("/tmp/bump_flop.npy", &nflop);
    int* h_mb32 = (int*)npy_load("/tmp/bump_maxbl.npy", &nmb);
    int ADIM = 2911419; int N = (int)nrows;
    printf("人口 %d 行\n", N);
    std::vector<unsigned> flop(ADIM, 0), mb(ADIM, 60);
    for (int i = 0; i < N; i++) { flop[h_rows[i]] = (unsigned)h_flop32[i]; mb[h_rows[i]] = (unsigned)h_mb32[i]; }
    std::vector<unsigned> off(ADIM + 1, 0);
    unsigned tot = 0;
    { std::vector<int> pos(ADIM, -1);
      for (int i = 0; i < N; i++) pos[h_rows[i]] = (int)tot, tot += (unsigned)h_est[i] * 2 + 64;
      for (int r = 0; r < ADIM; r++) off[r] = pos[r] < 0 ? 0u : (unsigned)pos[r];
      off[ADIM] = tot; }
    unsigned *d_rp,*d_ci,*d_rows,*d_rnz,*d_ovfr,*d_flop,*d_maxbl,*d_off,*d_tk32,*d_dummy;
    double *d_val,*d_tval;
    CHK(cudaMalloc(&d_rp,(ADIM+1)*4)); CHK(cudaMalloc(&d_ci,nc*4)); CHK(cudaMalloc(&d_val,nv*8));
    CHK(cudaMalloc(&d_rows,N*4)); CHK(cudaMalloc(&d_rnz,ADIM*4)); CHK(cudaMalloc(&d_ovfr,ADIM*4));
    CHK(cudaMalloc(&d_flop,ADIM*4)); CHK(cudaMalloc(&d_maxbl,ADIM*4)); CHK(cudaMalloc(&d_off,(ADIM+1)*4));
    CHK(cudaMalloc(&d_tk32,(size_t)tot*4)); CHK(cudaMalloc(&d_tval,(size_t)tot*8));
    CHK(cudaMalloc(&d_dummy,16));
    CHK(cudaMemcpy(d_rp,h_rp,(ADIM+1)*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_ci,h_ci,nc*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_val,h_val,nv*8,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rows,h_rows,N*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_flop,flop.data(),ADIM*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_maxbl,mb.data(),ADIM*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_off,off.data(),(ADIM+1)*4,cudaMemcpyHostToDevice));
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    double gprod = 0; for (int i=0;i<N;i++) gprod += h_flop32[i]; gprod /= 1e9;
    float best = 1e9;
    for (int rep = 0; rep < 4; rep++) {
        CHK(cudaMemset(d_dummy,0,16));
        cudaEventRecord(t0);
        theirs::somespgemm::hashNumericKernel<7,673,true,false,unsigned int><<<N,128,8192>>>(
            d_rp,d_ci,d_val,ADIM,ADIM,(int)nc, d_rp,d_ci,d_val,ADIM,ADIM,(int)nc,
            d_flop,d_maxbl,d_rows,d_off, d_tk32,d_tval,d_rnz,d_dummy,d_ovfr,d_dummy,
            nullptr,nullptr,0,0);
        cudaEventRecord(t1); CHK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms,t0,t1); if (ms<best) best=ms;
    }
    std::vector<unsigned> rnz(N); CHK(cudaMemcpy(rnz.data(),d_rnz,N*4,cudaMemcpyDeviceToHost));
    long long s=0; for(unsigned r:rnz) s+=r;
    printf("their核 × their flags × 我们的数据:%.1f ms(%.2f ns/行,%.1f Gprod/s)nnzΣ=%lld\n",
           best, best*1e6/N, gprod/best*1000, s);
    return 0;
}