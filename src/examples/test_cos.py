import sys, os

proj_path = os.path.dirname(__file__)
model_path = os.path.join(proj_path, '../model')
sys.path.append(model_path)

import numpy as np

import kernels
from kernels import KernelFunction
from optimizer import KernelMatcher


# Example using only sinusoidal kernels
f_target = KernelFunction(np.array([5, 5, 4, 3, 2, 0, 0, 0, 0, 1, 2]), smooth=True)
f_cos = KernelFunction(lambda x: kernels.cos(x, 2))

optimizer = KernelMatcher([f_cos for i in range(30)], f_target, res=0, max_gen=300)

t = optimizer.iterate()
print("Solution found: ")
print(t)