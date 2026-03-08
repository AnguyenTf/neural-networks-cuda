#include "activations.cuh"
#include <math.h>

__global__ void kSigmoid(const float *input, float *output, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += blockDim.x * gridDim.x){ 
        float x = int[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}