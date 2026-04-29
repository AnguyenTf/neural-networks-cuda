#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "nn_train.cuh"
#include "dot.cuh"
#include "activations.cuh"
#include "matrix_ops.cuh"

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
    float *d_X, *d_y;
    cudaMalloc(&d_X, TRAINING_SIZE * TRAINING_DIM * sizeof(float));
    cudaMalloc(&d_y, TRAINING_SIZE * sizeof(float));

    cudaMemcpy(d_X, h_X, TRAINING_SIZE * TRAINING_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, TRAINING_SIZE *sizeof(float), cudaMemcpyHostToDevice);

    // Initialize weights
    float *h_W0 = (float*)malloc(TRAINING_DIM * L1_SIZE * sizeof(float));
    float *h_W1 = (float*)malloc(L1_SIZE * OUTPUT_DIM * sizeof(float));

    for (int i = 0; i < TRAINING_DIM * L1_SIZE; i++) {
        h_W0[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }

    for (int i = 0; i < L1_SIZE * OUTPUT_DIM; i++) {
        h_W1[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }

    float *d_W0, *d_W1;
    cudaMalloc(&d_W0, TRAINING_DIM * L1_SIZE * sizeof(float));
    cudaMalloc(&d_W1, L1_SIZE * OUTPUT_DIM * sizeof(float));

    cudaMemcpy(d_W0, h_W0, TRAINING_DIM * L1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, L1_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Train the neural network

    int iterations = 50000;

    trainBinaryClassifier(
        d_X, TRAINING_DIM, TRAINING_SIZE,
        d_y, OUTPUT_DIM,
        d_W0, L1_SIZE,
        d_W1,
        iterations
    );

    // Compute predictions after training
    float h_pred[TRAINING_SIZE];
    float *d_pred;
    cudaMalloc(&d_pred, TRAINING_SIZE * sizeof(float));

    // Foward pass
    float *d_l1;
    cudaMalloc(&d_l1, TRAINING_SIZE *L1_SIZE * sizeof(float));

    forwardPass<<<1 , 1>>>(d_X, TRAINING_DIM, TRAINING_SIZE,
                        d_W0, L1_SIZE,
                        d_W1,
                        d_l1,
                        d_pred);
    cudaDeviceSynchronize();

    // remove one
    cudaMemcpy(h_pred, d_pred, TRAINING_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // print results
    printf("\nFinal Predictions:\n");
    for (int i = 0; i < TRAINING_SIZE; i ++){
        printf("Sample %d: pred = %.4f, true = %.1f\n", i, h_pred[i], h_y[i]);
    }

    // Clean up
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_pred);
    cudaFree(d_l1);

    free(h_W0);
    free(h_W1);

    return 0;
}
