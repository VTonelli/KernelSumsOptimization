import sys, os

proj_path = os.path.dirname(__file__)
model_path = os.path.join(proj_path, '../model')
sys.path.append(model_path)

import kernels
from kernels import KernelFunction

# Run this file to plot the example kernels
f_cos = KernelFunction(lambda x: kernels.cos(x, 2))
f_hill = KernelFunction(lambda x : kernels.hill(x, 4))
f_impulse = KernelFunction(lambda x : kernels.impulse(x, 0, 1))
f_triangle = KernelFunction(lambda x : kernels.triangle(x))

f_cos.plot()
f_hill.plot()
f_impulse.plot()
f_triangle.plot()