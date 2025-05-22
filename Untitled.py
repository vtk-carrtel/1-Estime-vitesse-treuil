import numpy as np

# pretend the derivative's sign mask:
mask = np.array([False, True, True, False, True, True, True, False])

idx = np.flatnonzero(mask)     # array([1, 2, 4, 5, 6])

# use one of those indices on the *original* depth array
depth = np.arange(len(mask)) * 10
print("depth at idx[2] =", depth[idx[2]])   # → 40
