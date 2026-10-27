#include <cuda_runtime.h>
__global__ void b_only(int *out) { *out = 2; }
template<int T> __global__ void tk(int *out) { *out = 200 + T; }   // B 体(c61)
void run_b(int *d) { tk<7><<<1,1>>>(d); }
