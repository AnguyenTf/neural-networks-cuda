#pragma once
#include <cuda_runtime.h>

/*
Header declares the kernels and device wrappers
*/

// Elementwise subtraction: out[i] = a[i] - b[i]
__global__ void kSub(const float *a, const float *b, float *out, int n);
__device__ void dSub(const float *a, const float *b, float *out, int rows, int cols);

// Elementwise multiplication: out[i] = a[i] * b[i]
__global__ void kMul(const float *a, const float *b, float *out, int n);
__global__ void dMul(const float *a, const float *b, float *out, int rows, int cols);