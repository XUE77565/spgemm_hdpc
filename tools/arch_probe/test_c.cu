#include <cstdio>
#include <cuda_runtime.h>
__global__ void only_a(int *out) { *out = 1; }
template<int T> __global__ void tk(int *out) { *out = 100 + T; }   // A 体(sm90)
void call_tk7_a(int *d) { tk<7><<<1,1>>>(d); }
int main() {
    int *d, h = -1; cudaMalloc(&d, 4);
    call_tk7_a(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("A: tk<7> = %d\n", h);
    extern void run_b(int *);
    run_b(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("after run_b = %d  (207=B独立=安全; 107/A体=stub弱合并=灾难)\n", h);
    return 0;
}
