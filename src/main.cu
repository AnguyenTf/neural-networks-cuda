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

=======
// Function to add the elements of two arrays
__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        y[i] = x[i] + y[i];
    }
}
>>>>>>> a0e45e8 (adjusting template and removed uncessary files)

/*##################################################
#####                MAIN                      #####
##################################################*/

<<<<<<< HEAD
=======
int main(void){
    // Make this 1 millions or 1M elements
    // N = 00000000 00001000 00000000 00000000
    int N = 1<<20;

    float *x, *y;

    // Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elemnets on the CPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memroy
    cudaFree(x);
    cudaFree(y);

    return 0;

}
>>>>>>> a0e45e8 (adjusting template and removed uncessary files)
