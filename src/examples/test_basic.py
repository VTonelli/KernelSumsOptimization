import sys, os

proj_path = os.path.dirname(__file__)
model_path = os.path.join(proj_path, '../model')
sys.path.append(model_path)

import numpy as np

import kernels
from kernels import KernelFunction
from optimizer import KernelMatcher


# Simple, basic example of the optimization process
f_target = KernelFunction(np.array([1, 1, 3, 1, 1, 0, 1, 2, 4, 2, 1]), smooth=True)
f_cos = KernelFunction(lambda x: kernels.cos(x, 2))
f_hill = KernelFunction(lambda x : kernels.hill(x, 4))

optimizer = KernelMatcher([f_hill for i in range(14)] + [f_cos for i in range(7)], f_target, res=0)

t = optimizer.iterate()
print("Solution found: ")
print(t)