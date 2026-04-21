# Neural Networks in CUDA (In-Progress)
*Insert NN architecture*

## Introducing Nueral Network in CUDA

In this project, I designed and implemented a fully GPU-accelerated 1 hidden layer nueral network. The network is learning binary prediction, but can also learn tasks like XOR, simple classification tasks, and small regression problems.  

The major operations i this network are matrix multiplication, activation functions, elementwise ops, and backpropagation

The goal of this project is to deeply understand how neural networks work under the hood while building real GPU programming experience.

---

### Highlights

- Fully custom CUDA implementation, no PyTorch, no TensorFlow
- GPU-accelerated forward pass, backpropagation, and weight updates
- Hand-written matrix multiplication, activation, and gradient kernels

---

### Project reflection/Help

Key things I learned were
- CUDA syntax
- Nueral Network Fundamentals
- Reading Documentations
- Interacting with CUDA communites
- CUDA Youtube Resources https://www.youtube.com/watch?v=2NgpYFdsduY&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU

Roughly around 1-2 months

---

### How it works

The neural network performs the full training loop on the GPU

1) Foward pass, uses kDot to compute weighted sums
2) Activation, uses kSigmoid to apply nonlinear activation
3) Compute error, uses kSub to compute prediction/target
4) Backpropagate, uses kDot_m1_m2t to propagate error backward
5) Activation Derivative, uses kSigmoid_d to compute σ'(x)
6) Weight gradient (Aᵀ × δ), uses kDot_m1T_m2 to compute gradients
7) Weight upade, uses KMul, kAdd, and kSub to update weights

---

### Kernel Explanations

Foward Pass - Kernel kDot in dot.su

Explanation
1) Each thread start with a global index i, which tells it which element of the output matrix C to compute. 
2) I compute total = A_rows * B_cols so I know how many output element exist. 
3) By using the grid stride loop, each thread can compute multiple elements if needed. This makes the kernel scalable regardless of matrix size or launch configuration. 
4) For each index i, I convert it into row and column using r = i / B_cols and c = i % B_cols. 
5) Then I compute the dot product between row r of A and column c of B by looping over k and accumlating A[r][k] * b[k][c].  
6) That gives me the value of C[r][c], which I store in C[i] since matrix in row-major order.

Example: *insert image*

Backpropagating - Kernel kDot_m1_m2T in dot.su

Explanation
1) Each thread start with a global index i.
2) I Compute total = A_rows * B_rows, because the output matrix C has A_rows rows B_rows columns when multiplying A * B 
3) Grid stride loop
4) For each index i, I convert it into row and column using r = i / B_rows and c = i % B_Rows.
5) Then I compute the dot product between row r of A dn row c of B (which corresponds to column c of B^T.
6) This gives me the value of C[r][c], which I stored in C[i] since the matrix is stored in row-major order.

Example: *insert image*

Weight Gradient - Kernel kDot_m1T_m2 in dot.su

Explanation
1) Each thread start with a global index i.
2) I compute total = A_cols * B_cols, because the output matrix C has A_cols rows and B_cols columns when multiplying A^T * B
3) Grid stride loop
4) For each index i, I convert it into row and column using r = i / B_cols and c = i % B_cols.
5) Then I compute the dot product betwen row r of A^t and column c of B. Since A is not explicitly transposed, I access A in transposed order using A[k][r], which in row major indexing is A[k * A_cols +r].
6) I loop over k and accumulate A[k][r] * B[k][c], which is the mathemeatical definition of A^T * B.
7) This gives me the value of C[r][c], which I store in C[i] since the matrix is stored in row major order.

Example: *insert image*

---

### Activation Functions (Sigmoid)

kSigmoid - Foward Activation
- Applies sigmoid element wise
- Used during forward pass

kSigmoid_d - Sigmoid Derivative
- Computes σ(x)(1 − σ(x))
- Used during backpropagation

---

### Matrix Operations

These kernels support the training loops:

kAdd - Add biases, accumulate gradients
kSub - Compute error, update weights
kMul - Scale gradients by learning rate or activation derivative
kTranspose - Used for wegith gradients and backpropagation

---

## Author
Andy Nguyen  
Insipring GPU & Parallel Computing Engineer
Email: anguyen57329@gmail.com

This project is part of my ongoing journey to master CUDA, GPU architecture, and low‑level neural network implementation.

---

## Why This Project Matters

This project demonstrates:
- Mastery of CUDA fundamentals
- Understanding of neural network math
- Ability to build ML systems from scratch
