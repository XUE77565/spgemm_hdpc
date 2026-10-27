#include <cstdio>
#include <cuda_runtime.h>
__global__ void shared_k(int *out) { *out = 111; }          // TU-A 体
template<int T> __global__ void tk(int *out) { *out = 100 + T; }   // TU-A 模板体(模拟 sm90 codegen)
void call_tk7_a(int *d) { tk<7><<<1,1>>>(d); }
int main() {
    int *d, h = -1; cudaMalloc(&d, 4);
    shared_k<<<1,1>>>(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("A: shared_k = %d\n", h);
    call_tk7_a(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("A: tk<7>   = %d\n", h);
    extern void run_b(int *);
    run_b(d); cudaDeviceSynchronize();
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("after run_b = %d  (207=B体赢=各自实例独立; 107=stub被合并=灾难)\n", h);
    return 0;
}
