#include "matrix_ops.cuh"

__global__ void kSub(const float *a, const float *b, float *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    
}