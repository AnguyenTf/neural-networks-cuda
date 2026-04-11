#include "dot.cuh"

// Foward Pass
__global__ void kDot(const float *A, const float *B, float *C,
                     int A_rows, int A_cols, int B_cols){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = A_rows * B_cols;


    for (; i < total; i += stride){
        // r: rows, c: columns
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;

        for (int k = 0; k < A_cols; k++){
            float a_val = A[r * A_cols + k];
            float b_val = B[k * B_cols + c];

            sum += a_val * b_val;
        }

        C[i] = sum;

    }
}

// Backpropagating
__global__ void kDot_m1_m2T(const float *A, const float *B, float *C,
                            int A_rows, int A_cols, int B_rows){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = A_rows * B_rows;


    for (; i < total; i += stride){
        int r = i / B_rows;
        int c = i % B_rows;

        float sum = 0.0f;

        for (int k = 0; k < A_cols; k++){
            float a_val = A[r * A_cols + k];
            float b_val = B[c * A_cols + k];

            sum += a_val * b_val;
        }

        C[i] = sum;

    }
}

// Weight gradients
__global__ void kDot_m1T_m2(const float *A, const float *B, float *C,
                            int A_rows, int A_cols, int B_cols) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = gridDim.x * blockDim.x;
    int total = A_cols * B_cols;


    for (; i < total; i += stride) {
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;

        for (int k = 0; k < A_rows; k++){
            float a_val = A[k * A_cols + r];
            float b_val = B[k * B_cols + c];

            sum += a_val * b_val;
        }

        C[i] = sum;
    }
}
