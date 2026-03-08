#pragma once
#include <cuda_runtime.h>

__global__ void kSigmoid(const float *input, float *output, int n);
__global__ void kSigmoid_d(const float *input, float *output, int n);

__device__ void dSigmoid(const float *input, float *output, int rows, int cols);
__device__ void dSigmoid_d(const float *input, float *output, int rows, int cols);