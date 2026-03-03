### Linear Algebra Foundation

Neural networks are built on a small set of linear algebra operations. These operations define how data flows through a model, how predictions are computed, and how gradients are calculated during training.
 
### Data Structures/Shapes: 
CUDA sotres all data as a flat, contiguous block of memory(float*, double*). Higher dimensional structures (vectors, matrices, tensors) are logical views on top of this 1D memory.
- Vector (1D)  
  - A 1D array (e.g., a single input sample)
  - 𝑥 ∈ 𝑅^𝐷
  - float* x; // length D
- Matrix (2D)  
  - A 2D array (e.g., a batch of samples)
  - 𝑥 ∈ 𝑅^(B * D)
  - float* X; // length B x D
- Tensor
  - Tensor - an N-dimensional array (e.g., image data with channels).
  - 𝑇 ∈ 𝑅^(𝑑1 × 𝑑2 × ⋯ ×𝑑𝑛)
  - float* T; // length = d1 * d2 * ... * dn

### Row-Major Memory Layout:  
CUDA (like c/c++) uses row-major order.  
- Elements in the **same row** are stored next to each other.
- Increasing the **column index** moves to the next memory location.  

Example: a 3 X 4 matrix flattened in memory  
Index:
&nbsp;&nbsp;&nbsp;&nbsp;0
&nbsp;&nbsp;&nbsp;1
&nbsp;&nbsp;&nbsp;&nbsp;2
&nbsp;&nbsp;&nbsp;3
&nbsp;&nbsp;&nbsp;&nbsp;4
&nbsp;&nbsp;&nbsp;&nbsp;5
&nbsp;&nbsp;&nbsp;&nbsp;6
&nbsp;&nbsp;&nbsp;7
&nbsp;&nbsp;&nbsp;&nbsp;8
&nbsp;&nbsp;&nbsp;9
&nbsp;&nbsp;10
&nbsp;&nbsp;11    
Value: x00 x01 x02 x03 x10 x11 x12 x13 x20 x21 x22 x23  

### Converting between 2D and 1D indexing

CUDA kernels often assign each thread a single flat index:  
When working in CUDA, you often launch kernels where each thread is assigned a single flat index like this:  
  - int idx = blockIdx.x * blockDim.x + threadIdx.x;  
But neural network math is expessed in rows and columns, not flat indices. To apply operations correctly, each thread must recover its original matrix coordinates (B,D);

Find the flat index of element (i,j):
idx = i * D + j
- Each row has D = 4 elements
- So to reach row i, you skip i full rows -> i * 4
- Then you move to column j inside that row.

To recover (i,j) from a flat index:  
i = ⌊idx/D⌋ // Get the floor  
j = idx - i * D

### Example:  
Take a matrix with 2 rows and 4 columns:  
x = [ (a, b, c, d)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(e, f, &nbsp;g, h) ]  

Flattened in row-major order CUDA stores it like this:  
idx: &nbsp;0  &nbsp;1  &nbsp;2  &nbsp;3  &nbsp;4  &nbsp;5  &nbsp;6  &nbsp;7  
val: &nbsp;a  &nbsp;b  &nbsp;c  &nbsp;d  &nbsp;e  &nbsp;f  &nbsp;g  &nbsp;h  

Now to find the flat index of element (1,2)
- D = 4, die to how many columns are in a row
- idx = 1 * 4 + 2 = 6
- So the flattened array: index 6 is g
- i = ⌊6/4⌋ = ⌊1.5⌋ = 1
- j = 6 - 1 * 4 = 2



