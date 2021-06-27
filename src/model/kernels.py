import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# Class defining a function to be used either as the target or as one of the kernels to optimize
class KernelFunction:
    def __init__(self, points, smooth=True):
        # Max number of significant digits for function points
        self.precision_digits = 5
        # Number of function points at various resolutions
        self.res0_n = 11
        self.res1_n = 101
        self.res2_n = "continuous"
        # x-axis values at the resolution levels
        self.res0_x = np.linspace(0, 1, num=self.res0_n)
        self.res1_x = np.linspace(0, 1, num=self.res1_n)
        self.res2_x = np.array([0, 1])

        if (isinstance(points, np.ndarray)):        # Provided input is an array of points
            # Check that correct number of points is provided
            if (points.shape != (self.res0_n,) and points.shape != (self.res1_n,)):
                raise Exception("Expected points shape (11,) or (101,)! Found " + str(points.shape) + "!")
            # Number of provided points is low resolution
            if len(points) == self.res0_n:
                k = 2
                if not smooth:  # Set interpolation degree when working at low resolution
                    k = 1
                # y-axis values at the resolution levels, the higher res are obtained via interpolation
                self.res0_f = np.around(np.array(points, dtype=float), self.precision_digits)
                # Continuous function is interpolated from given resolution
                self.res2_f = InterpolatedUnivariateSpline(np.linspace(0, 1, num= self.res0_n),
                                                           self.res0_f, k=k, ext=1)
                self.res1_f = self.res2_f(self.res1_x)
                self.res1_f = np.around(self.res1_f, self.precision_digits)
                # Number of provided points is high resolution
            elif len(points) == self.res1_n:
                # Same as previous case, starting from resolution 1
                self.res1_f = np.around(np.array(points, dtype=float), self.precision_digits)
                self.res2_f = InterpolatedUnivariateSpline(np.linspace(0, 1, num= self.res1_n),
                                                           self.res1_f, k=2, ext=1)
                self.res0_f = self.res2_f(self.res1_x)
                self.res0_f = np.around(self.res0_f, self.precision_digits)
        elif (callable(points)):                    # Provided input is a callable function of 1 variable
            # Discrete resolutions are obtained directly from calling the provided function
            self.res1_f = points(self.res1_x)
            self.res1_f = np.around(self.res1_f, self.precision_digits)
            self.res0_f = points(self.res0_x)
            self.res0_f = np.around(self.res0_f, self.precision_digits)
            
            # Function here is rebuilt via interpolation, to avoid the function being non-zero outside
            # of range [0, 1] (eg. if sin(x) is the callable input passed here, it would produce non-zero
            # results when x-shifted)
            self.res2_f = InterpolatedUnivariateSpline(np.linspace(0, 1, num= self.res1_n),
                                                           self.res1_f, k=2, ext=1)
    
    # Helper function to get the function at any res in a single call
    def function(self, res=0):
        if res==2:
            return self.res2_f
        elif res==1:
            return self.res1_f
        elif res==0:
            return self.res0_f
    
    # Plots the function at the provided resolution and with the provided x-shift
    def plot(self, show=True, t=0, res=2):
        if res==2 or res==1:
            plt.plot(self.res1_x, self.res2_f(self.res1_x + t))
        elif res==0:
            plt.plot(self.res0_x, self.res2_f(self.res0_x + t))
        if show:
            plt.show()

    # Shifts the function on the x axis by the provided amount t, at provided resolution
    # NOTE: to avoid overhead memory allocation, returns an array, not a KernelFunction
    def x_shift(self, t, res):
        shifted_f = []
        if res==1 or res==2:
            shifted_f = self.res2_f(self.res1_x + t)
        elif res==0:
            shifted_f = self.res2_f(self.res0_x + t)
        shifted_f = np.around(shifted_f, self.precision_digits)
        return shifted_f

# Definition of some common kernel functions
def impulse(x, pos, A):
    return (np.isclose(x, pos, atol=1/101)) * A

def triangle(x):
    return np.where(x <= 0.5, x, -x+1)
    
def hill(x, f):
    return np.where(x > 1.0/f, 0, np.sin(x*f*np.pi))

def cos(x, f):
    return np.cos(x*f*3.141)
