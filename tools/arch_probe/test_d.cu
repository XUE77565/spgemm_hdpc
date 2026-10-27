#include <cstdio>
#include <cuda_runtime.h>
static __global__ void skernel(int *out) { *out = 111; }   // static 非模板 kernel
__device__ int g_probe_var = 5;                            // 两 TU 同名 device 全局
void call_d(int *d) { skernel<<<1,1>>>(d); }
int main() {
    int *d, h = -1; cudaMalloc(&d, 4);
    call_d(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("D: static kernel = %d\n", h);
    extern void run_e(int *);
    run_e(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("after run_e = %d (222=E独立static=安全)\n", h);
    return 0;
}
