import numpy as np
from utils import *

# Testing normalize() and unnormalize()
test_matrix1 = np.random.randn(3, 2)
mean, std = np.mean(test_matrix1, axis=0), np.std(test_matrix1, axis=0)
z = normalize(test_matrix1, mean, std)
x = unnormalize(z, mean, std)
assert np.allclose(test_matrix1, x) == True