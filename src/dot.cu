#include "dot.cuh"

/*
* Computes multiplication
*Each thread computs one element
*/
__global__ void kDot(const float *A, const float *B, float *C,
                    int A_rows, int A_cols, int B_cols){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total = A_rows * B_cols;

    for (; i < total; i+= blockDim.x * threadIdx.x){
        // row in C
        int r = i / B_cols;
        // Column in C
        int c = i % B_cols;

        float sum = 0.0f;

        for (int k = 0; k < A_cols; k++){
            sum += A[r * A_cols + k] * B[k* B_cols + c];
        }

        C[i] = sum;

    }
}

__device__ void dDot(const float *A, const float *B, float *C,
                    int A_rows, int A_cols, int B_cols) {
    int total = A_rows * B_cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    kDot<<<blocks, threads>>>(A, B, C, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize();
}

/*
* Computes C = A * Bᵀ
*
*/
__global__ void kDot_m1_m2T(const float *A, const float *B, float *C,
                            int A_rows, int A_cols, int B_rows){
    
    int i = blockIdx.x + blockDim.x + threadIdx.x;

    int total = A_rows * B_rows;

    for (; i < total; i += blockDim.x * gridDim.x){
        int r = i / B_rows;
        int c = i % B_rows;

        float sum = 0.0f;

        for (int k = 0; k < A_cols; k++){
            sum += A[r * A_cols + k] * B[k* A_cols +k];
        }

        C[i] = sum;

    }
}

__device__ void dDot_m1_m2T(const float *A, const float *B, float *C, 
                            int A_rows, int A_cols, int B_rows){
    
    int total = A_rows * B_rows;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    kDot_m1_m2T<<<blocks, threads>>>(A, B, C, A_rows, A_cols, B_rows);
    cudaDeviceSynchronize();
}

/*
* Computs C = Aᵀ * B
*
*/

__global__ void kDot_m1T_m2(const float *A, const float *B, float *C,
                            int A_rows, int A_cols, int B_cols) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
                            
    int total = A_cols * B_cols;

    for (; i < total; i += blockDim.x * gridDim.x) {
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;

        for (int k = 0; k < A_rows; k++){
            sum += A[k * A_cols + r] * B[k * B_cols + c];
        }

        C[i] = sum;
    }
}

__device__ void dDot_m1_m2(const float *A, const float *B, float *C,
                            int A_rows, int A_cols, int B_cols){

    int total = A_cols * B_cols;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    kDot_m1T_m2<<<blocks, threads>>>(A, B, C, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize();                            
}