"""
This script is a short example demonstrating the use of numpy.flatnonzero
and array indexing with its results.

It showcases how to:
1. Define a boolean mask.
2. Use np.flatnonzero to get the indices of True elements in the mask.
3. Use these indices to access elements in another array (e.g., a 'depth' array).
"""
import numpy as np

# pretend the derivative's sign mask:
mask = np.array([False, True, True, False, True, True, True, False])

# Use np.flatnonzero to get the indices of the True elements in the mask.
# 'idx' will store the indices where 'mask' is True.
idx = np.flatnonzero(mask)     # Expected: array([1, 2, 4, 5, 6])

# Create a sample 'depth' array corresponding to the length of the mask.
# This array will be indexed using the results from np.flatnonzero.
depth = np.arange(len(mask)) * 10

# Example: Access an element in the 'depth' array using an index from 'idx'.
# idx[2] corresponds to the third True element's original index in 'mask', which is 4.
# So, depth[idx[2]] is equivalent to depth[4].
print("depth at idx[2] =", depth[idx[2]])   # Expected output: depth at idx[2] = 40

# For clarity, let's print the arrays:
print("\n--- Intermediate values for clarity ---")
print("Original mask:", mask)
print("Indices of True values (idx):", idx)
print("Depth array:", depth)
print(f"Value of idx[2]: {idx[2]}")
print(f"Depth at index {idx[2]} (depth[idx[2]]): {depth[idx[2]]}")
