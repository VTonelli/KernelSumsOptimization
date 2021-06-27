import sys, os

proj_path = os.path.dirname(__file__)
model_path = os.path.join(proj_path, '../model')
sys.path.append(model_path)

import kernels
from kernels import KernelFunction
from optimizer import KernelMatcher

# Approximation of a function via impulses
# WARNING: Somewhat slow
f_cos = KernelFunction(lambda x: kernels.cos(x, 4))
f_impulse = KernelFunction(lambda x : kernels.impulse(x, 0, 1))
f_impulse_n = KernelFunction(lambda x : kernels.impulse(x, 0, -1))
f_target = f_cos

optimizer = KernelMatcher([f_impulse for i in range(75)] + [f_impulse_n for i in range(50)],
                          f_target, res=1, plot_res=1)

t = optimizer.iterate()
print("Solution found: ")
print(t)