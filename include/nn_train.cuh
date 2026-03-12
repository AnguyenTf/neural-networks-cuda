#pragma once 
#include <cuda_runtime.h>

// Trains 1 hidden layer binary classifier on the GPU
void trainBinaryClassifier(
    const float *d_X, int X_w, int X_h,
    const float *d_Y, int y_w,
    float *d_W0, int L1_w,
    float *d_W1,
    int iterations
);