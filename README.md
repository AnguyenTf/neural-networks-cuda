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
1) Each thread start with a global index i.
2) I Compute total = A_rows * B_rows, because the output matrix C has A_rows rows B_rows columns when multiplying A * B 
3) Grid stride loop
4) For each index i, I convert it into row and column using r = i / B_rows and c = i % B_Rows.
5) Then I compute the dot product between row r of A dn row c of B (which corresponds to column c of B^T.
6) This gives me the value of C[r][c], which I stored in C[i] since the matrix is stored in row-major order.

Example:

## Weight Gradient

Explanation
1) Each thread start with a global index i.
2) I compute total = A_cols * B_cols, because the output matrix C has A_cols rows and B_cols columns when multiplying A^T * B
3) Grid stride loop
4) For each index i, I convert it into row and column using r = i / B_cols and c = i % B_cols.
5) Then I compute the dot product betwen row r of A^t and column c of B. Since A is not explicitly transposed, I access A in transposed order using A[k][r], which in row major indexing is A[k * A_cols +r].
6) I loop over k and accumulate A[k][r] * B[k][c], which is the mathemeatical definition of A^T * B.
7) This gives me the value of C[r][c], which I store in C[i] since the matrix is stored in row major order.



















