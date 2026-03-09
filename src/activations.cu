#include "activations.cuh"
#include <math.h>

/*###################################################
#####                KERNELS                    #####
###################################################*/

/*
GPU Kernel kSigmoid
-------------------
This kernel applies the sigmoid function to each element of 'input'
*/
__global__ void kSigmoid(const float *input, float *output, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += blockDim.x * gridDim.x){ 
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}


__device__ void dSigmoid(const float *input, float *output, int rows, int cols){ 
    int n = rows * cols;

    int threads =256;
    int blocks = (n + threads - 1) / threads;

    kSigmoid<<<blocks, threads>>>(input, output, n);

    cudaDeviceSynchronize();
}

__global__ void kSigmoid_d(const float *input, float *output, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += blockDim.x * gridDim.x){
        float s = input[i];
        output[i] = s * (1.0f - s);
    }
}

__device__ void dSigmoid_d(const float *input, float *output, int rows, int cols){
    int n = rows * cols;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    kSigmoid_d<<<blocks, threads>>>(input, output, n);
    cudaDeviceSynchronize();
}