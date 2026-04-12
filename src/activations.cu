#include "activations.cuh"

// Sigmoid
__global__ void kSigmoid(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// Derivative sigmoid
__global__ void kSigmoid_d(const float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-x));
        output[i] = s * (1.0f - s);
    }
}
