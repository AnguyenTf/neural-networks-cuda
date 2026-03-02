Neural networks are built on a small set of linear algebra operations. These operations define how data flows through a model, how predictions are computed, and how gradients are calculated during training.

Data Structure:
- Vector
  - A 1D array (e.g., a single input sample)
- Matrix
  - A 2D array (e.g., a batch of samples)
- Tensor
  - Tensor - an N-dimensional array (e.g., image data with channels).
 
Shapes: 
How vectors, matrices, and tensors are stored in CUDA. Everything is stored as a flat, contiguous block of memory, usually a float* or double*.
X ∈ ℝ^{B × D}

