#include "nn_train.cuh"
#include <cuda_runtime.h>

/*##################################################
#####              DEVICE HELPERS              #####
##################################################*/   

// A * B
__device__ void dDot(const float* A, const float* B, float* C,
                     int A_rows, int A_cols, int B_cols)
{
    int total = A_rows * B_cols;

    for (int i = 0; i < total; ++i) {
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k)
            sum += A[r * A_cols + k] * B[k * B_cols + c];

        C[i] = sum;
    }
}

// A * B^T
__device__ void dDot_m1_m2T(const float* A, const float* B, float* C,
                            int A_rows, int A_cols, int B_rows)
{
    int total = A_rows * B_rows;

    for (int i = 0; i < total; ++i) {
        int r = i / B_rows;
        int c = i % B_rows;

        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k)
            sum += A[r * A_cols + k] * B[c * A_cols + k];

        C[i] = sum;
    }
}

// A^T * B
__device__ void dDot_m1T_m2(const float* A, const float* B, float* C,
                            int A_rows, int A_cols, int B_cols)
{
    int total = A_cols * B_cols;

    for (int i = 0; i < total; ++i) {
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;
        for (int k = 0; k < A_rows; ++k)
            sum += A[k * A_cols + r] * B[k * B_cols + c];

        C[i] = sum;
    }
}

// elementwise subtraction
__device__ void dSub(const float* a, const float* b, float* out, int rows, int cols)
{
    int n = rows * cols;
    for (int i = 0; i < n; ++i)
        out[i] = a[i] - b[i];
}

// elementwise multiplication
__device__ void dMul(const float* a, const float* b, float* out, int rows, int cols)
{
    int n = rows * cols;
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
}

// sigmoid
__device__ void dSigmoid(const float* input, float* output, int rows, int cols)
{
    int n = rows * cols;
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// sigmoid derivative
__device__ void dSigmoid_d(const float* input, float* output, int rows, int cols)
{
    int n = rows * cols;
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-x));
        output[i] = s * (1.0f - s);
    }
}

/*##################################################
#####             TRAINING KERNEL              #####
##################################################*/  

__global__ void kFit(const float* X, int X_w, int X_h,
                     const float* y, int y_w,
                     float* l1, int l1_w, float* l1_delta,
                     float* pred, float* pred_delta,
                     float* W0,
                     float* W1,
                     float* buffer,
                     int iterations)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int iter = 0; iter < iterations; iter++)
        {
            // -------------------------
            // Forward pass
            // -------------------------
            dDot(X, W0, l1, X_h, X_w, l1_w);
            dSigmoid(l1, l1, X_h, l1_w);

            dDot(l1, W1, pred, X_h, l1_w, y_w);
            dSigmoid(pred, pred, X_h, y_w);

            // -------------------------
            // Output layer delta
            // -------------------------
            dSub(y, pred, pred_delta, X_h, y_w);
            dSigmoid_d(pred, buffer, X_h, y_w);
            dMul(pred_delta, buffer, pred_delta, X_h, y_w);

            // -------------------------
            // Hidden layer delta
            // -------------------------
            dDot_m1_m2T(pred_delta, W1, l1_delta, X_h, y_w, l1_w);
            dSigmoid_d(l1, buffer, X_h, l1_w);
            dMul(l1_delta, buffer, l1_delta, X_h, l1_w);

            // -------------------------
            // Weight updates with learning rate
            // -------------------------
            float lr = 0.1f;

            // W1 update: buffer size must handle l1_w * y_w
            dDot_m1T_m2(l1, pred_delta, buffer, X_h, l1_w, y_w);
            for (int i = 0; i < l1_w * y_w; i++)
                W1[i] += lr * buffer[i];

            // W0 update: buffer size must handle X_w * l1_w
            dDot_m1T_m2(X, l1_delta, buffer, X_h, X_w, l1_w);
            for (int i = 0; i < X_w * l1_w; i++)
                W0[i] += lr * buffer[i];
        }
    }
}

/*##################################################
#####         HOST TRAINING FUNCTION           #####
##################################################*/  

void trainBinaryClassifier(const float* d_X, int X_w, int X_h,
                           const float* d_y, int y_w,
                           float* d_W0, int L1_w,
                           float* d_W1,
                           int iterations)
{
    // sizes in elements
    int l1_elems    = X_h * L1_w;
    int pred_elems  = X_h * y_w;
    int w1grad_elems = L1_w * y_w;
    int w0grad_elems = X_w * L1_w;

    int max_elems = l1_elems;
    if (pred_elems  > max_elems) max_elems = pred_elems;
    if (w1grad_elems > max_elems) max_elems = w1grad_elems;
    if (w0grad_elems > max_elems) max_elems = w0grad_elems;

    size_t l1_size   = l1_elems   * sizeof(float);
    size_t pred_size = pred_elems * sizeof(float);
    size_t buffer_size = max_elems * sizeof(float);

    float *d_l1, *d_l1_delta, *d_pred, *d_pred_delta, *d_buffer;

    cudaMalloc(&d_l1,         l1_size);
    cudaMalloc(&d_l1_delta,   l1_size);
    cudaMalloc(&d_pred,       pred_size);
    cudaMalloc(&d_pred_delta, pred_size);
    cudaMalloc(&d_buffer,     buffer_size);

    cudaMemset(d_l1,         0, l1_size);
    cudaMemset(d_l1_delta,   0, l1_size);
    cudaMemset(d_pred,       0, pred_size);
    cudaMemset(d_pred_delta, 0, pred_size);
    cudaMemset(d_buffer,     0, buffer_size);

    kFit<<<1,1>>>(d_X, X_w, X_h,
                  d_y, y_w,
                  d_l1, L1_w, d_l1_delta,
                  d_pred, d_pred_delta,
                  d_W0,
                  d_W1,
                  d_buffer,
                  iterations);

    cudaDeviceSynchronize();

    cudaFree(d_l1);
    cudaFree(d_l1_delta);
    cudaFree(d_pred);
    cudaFree(d_pred_delta);
    cudaFree(d_buffer);
}

/*##################################################
#####            INFERENCE KERNEL              #####
##################################################*/ 

__global__ void forwardPass(const float* X, int X_w, int X_h,
                            const float* W0, int L1_w,
                            const float* W1,
                            float* l1,
                            float* pred)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dDot(X, W0, l1, X_h, X_w, L1_w);
        dSigmoid(l1, l1, X_h, L1_w);

        dDot(l1, W1, pred, X_h, L1_w, 1);
        dSigmoid(pred, pred, X_h, 1);
    }
}
