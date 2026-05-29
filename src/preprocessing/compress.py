import numpy as np

X = np.load("X_tensor_group2.npy")
np.savez_compressed("X_tensor_compressedgroup2.npz", X=X)
print("Done!")