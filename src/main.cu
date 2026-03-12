#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#inlcude "nn_train.cuh"

/*###################################################
#####                KERNELS                    #####
###################################################*/



/*##################################################
#####                MAIN                      #####
##################################################*/

int main() {
    // Training Data
    const int TRAINING_SIZE = 4;
    const int TRAINING_DIM = 4;
    const int L1_SIZE = 8;
    const int OUTPUT_DIM = 1;

    float h_X[TRAINING_SIZE * TRAINING_DIM] = {
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        6.2, 3.4, 5.4, 2.3,
        5.9, 3.0, 5.1, 1.8
    };

    float h_y[TRAINING_SIZE] = {0, 0, 1, 1};

    // Allocate device memory
    float *d_X, *d_Y;
    cudaMalloc(&d_Y, )
}