#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>


/*###################################################
#####                KERNELS                    #####
###################################################*/

<<<<<<< HEAD
__global__ void matMul(float* A, float* B, float* C, 
                            int M, int K, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float sum = 0.0f;
        
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
    C[row * N + col] = sum;
    }

=======
// Function to add the elements of two arrays
void add(int n, float *x, float *y){
    // 
    for (int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
>>>>>>> 31fdb8e (testing adding the element of two arrays on the CPU)
}

/*##################################################
#####                MAIN                      #####
##################################################*/

int main(void){
    // Make this 1 millions or 1M elements
    // N = 00000000 00001000 00000000 00000000
    int N = 1<<20;

    // Create 
    float *x = new float[N];
    float *y = new float[N];

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elemnets on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    
<<<<<<< HEAD
    int M = 3, K = 2, N = 4;

    float h_A[M*K] = {
        1, 2,
        3, 4,
        5, 6
    };

    float h_B[K*N] = {
        7, 8, 9, 10,
        11, 12, 13, 14
    };

    float h_C[M*N];

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N + 16)/16, (M+15)/16);

    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C:\n";

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            std::cout << h_C[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
    
=======

    // Free memory
    delete [] x;
    delete [] y;

    return 0;

>>>>>>> 31fdb8e (testing adding the element of two arrays on the CPU)
}