#include "matrix_ops.cuh"


/*
* Elementwise subtraction
* Used in backprop to compute
*/
__global__ void kSub(const float *a, const float *b, float *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += blockDim.x * gridDim.x){
        out[i] = a[i] - b[i];
    }
}

/*
* Device wrapper for kSub - refer to matrix_ops.cuh
* Allows subtraction to be called from inside other GPU kernels
*/
__device__ void dSub(const float *a, const float *b, float *out, int rows, int cols){
    int n = rows * cols;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    kSub<<<blocks, threads>>>(a, b, out, n);
    cudaDeviceSynchronize();
}

/*
* Elementwise Multiplication - refer to matrix_ops.cuh
* Used in backprop to compute: delta = error * sigmoid_derivative
*/
__global__ void kMul(const float *a, const float *b, float *out, int rows, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for(; i < n; i += blockDim.x * gridDim.x){
        out[i] = a[i] + b[i];
    }
    
}

/*
* Device wrapper for kMul - refer to matrix_ops.cuh
* Allow multiplication to be called from inside other GPU kernels
*/
__device__ void dMUl(const float *a, const float *b, float *out, int rows, int cols){
    int n = rows * cols;
    int threads = 256; 
    int blocks = (n + threads - 1) / threads;

    kMul<<<blocks, threads>>>(a, b, out, n);
    cudaDeviceSynchronize();
}
