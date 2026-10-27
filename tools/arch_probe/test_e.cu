#include <cuda_runtime.h>
static __global__ void skernel(int *out) { *out = 222; }   // E 侧同名 static
__device__ int g_probe_var = 9;                            // E 侧同名 device 全局
void run_e(int *d) { skernel<<<1,1>>>(d); }
