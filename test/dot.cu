#include <stdio.h>
#include <cuda_runtime.h>


/*##################################################
#####                kernel                    #####
##################################################*/

__global__ void kDot(const float *A, const float *B, float *C,
                     int A_rows, int A_cols, int B_cols){

    // A_rows: 2
    // A_cols: 3
    // B_cols: 2
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global Index: %d", i);
    // Total = 2 * 2 = 4
    int total = A_rows * B_cols;
    int stride = gridDim.x * blockDim.x;

    printf("Global Index: %d\nStride: %d \n-------------------------\n", i, stride);

    for (; i < total; i += stride){
        // r: rows, c: collumns
        int r = i / B_cols;
        int c = i % B_cols;

        float sum = 0.0f;

        for (int k = 0; k < A_cols; k++){
            float a_val = A[r * A_cols + k];
            float b_val = B[k * B_cols + c];
            printf("Thread %d: A[%d][%d]=%f  B[%d][%d]=%f  product=%f\n",
                   threadIdx.x, r, k, a_val, k, c, b_val, a_val * b_val);
            sum += a_val * b_val;
        }

        printf("Thread %d: C[%d][%d] = %f\n", threadIdx.x, r, c, sum);

        C[i] = sum;

    }
}

/*##################################################
#####                MAIN                      #####
##################################################*/

int main() {
    // A is 2x3
    float h_A[6] = {1,2,3,
                    4,5,6};

    // B is 3x2
    float h_B[6] = {7,8,
                    9,10,
                    11,12};

    // C is 2x2
    float h_C[4];

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, 6 * sizeof(float));
    cudaMalloc(&d_B, 6 * sizeof(float));
    cudaMalloc(&d_C, 4 * sizeof(float));

    cudaMemcpy(d_A, h_A, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, 32 threads (overkill but fine)
    kDot<<<1, 32>>>(d_A, d_B, d_C,
                    2, 3, 2); // A_rows=2, A_cols=3, B_cols=2

    cudaMemcpy(h_C, d_C, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C = A * B:\n");
    printf("%f %f\n", h_C[0], h_C[1]);
    printf("%f %f\n", h_C[2], h_C[3]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
