# Neural Networks in CUDA
*Insert NN architecture*

## Introducing Nueral Network in CUDA

In this project, I designed and implemented a fully GPU-accelerated nueral network

**Author:** Andy Nguyen
**Email:** anguyen57329@gmail.com

---

### Project reflection

Key things I learned were
- CUDA syntax
- Nueral Network Fundamentals

Roughly around 1-2 months

This project is one of the many I want to tackle to improve my CUDA skills

---

##  Foward Pass

Explanation
1) Each thread start with a global index i, which tells it which element of the output matrix C to compute. 
2) I compute total = A_rows * B_cols so I know how many output element exist. 
3) By using the grid stride loop, each thread can compute multiple elements if needed. This makes the kernel scalable regardless of matrix size or launch configuration. 
4) For each index i, I convert it into row and column using r = i / B_cols and c = i % B_cols. 
5) Then I compute the dot product between row r of A and column c of B by looping over k and accumlating A[r][k] * b[k][c].  
6) That gives me the value of C[r][c], which I store in C[i] since matrix in row-major order.

Example:

## Backpropagating

Explanation
1) 
