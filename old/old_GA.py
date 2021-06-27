import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

class KernelFunction:
    def __init__(self, points):
        # Max precision
        self.precision_digits = 5
        # Number of points at various resolutions
        self.res0_n = 11
        self.res1_n = 501
        self.res2_n = "continuous"
        # x-axis values at the resolution levels
        self.res0_x = np.linspace(0, 1, num=self.res0_n)
        self.res1_x = np.linspace(0, 1, num=self.res1_n)
        self.res2_x = np.array([0, 1])
        if (isinstance(points, np.ndarray)):
            if (points.shape != (self.res0_n,) and points.shape != (self.res1_n,)):
                raise Exception("Expected points shape (11,) or (501,)! Found " + str(points.shape) + "!")
            if len(points) == self.res0_n:
                # y-axis values at the resolution levels
                # Resolution 2 is the callable function defined in all reals [0; 1]
                self.res0_f = np.around(np.array(points, dtype=float), self.precision_digits)
                res0_f_extension = np.append(self.res0_f, np.zeros(self.res0_n-1))
                res0_f_extension = np.insert(res0_f_extension, 0, np.zeros(self.res0_n-1))
                self.res2_f = InterpolatedUnivariateSpline(np.linspace(-1, 2, num= 3*self.res0_n -2),
                                                           res0_f_extension, k=1)
                self.res1_f = self.res2_f(self.res1_x)
                self.res1_f = np.around(self.res1_f, self.precision_digits)
            elif len(points) == self.res1_n:
                # y-axis values at the resolution levels
                # Resolution 2 is the callable function defined in all reals [0; 1]
                self.res1_f = np.around(np.array(points, dtype=float), self.precision_digits)
                res1_f_extension = np.append(self.res1_f, np.zeros(self.res1_n-1))
                res1_f_extension = np.insert(res1_f_extension, 0, np.zeros(self.res1_n-1))
                self.res2_f = InterpolatedUnivariateSpline(np.linspace(-1, 2, num= 3*self.res1_n -2),
                                                           res1_f_extension, k=1)
                self.res0_f = self.res2_f(self.res1_x)
                self.res0_f = np.around(self.res0_f, self.precision_digits)
        elif (callable(points)):
            self.res2_f = points
            
            self.res0_f = self.res2_f(self.res0_x)
            self.res0_f = np.around(self.res0_f, self.precision_digits)
            self.res1_f = self.res2_f(self.res1_x)
            self.res1_f = np.around(self.res1_f, self.precision_digits)
    
    def plot(self, t=0, res=2):
        if res==2 or res==1:
            plt.plot(self.res1_x, self.res2_f(self.res1_x + t))
        elif res==0:
            plt.plot(self.res0_x, self.res2_f(self.res0_x + t))
        plt.show()

    def x_shift(self, t, res):
        shifted_f = []
        if res==1 or res==2:
            shifted_f = self.res2_f(self.res1_x + t)
        elif res==0:
            shifted_f = self.res2_f(self.res0_x + t)
        shifted_f = np.around(shifted_f, self.precision_digits)
        return shifted_f

class KernelMatcher:
    def __init__(self, kernels, target, res):
        #Kernels should be provided as a Python list
        self.target = target
        self.kernels = kernels
        self.kernels_n = len(kernels)
        self.res = res
        
        self.res_n = 0
        if res==0: self.res_n = self.kernels[0].res0_n
        if res==1: self.res_n = self.kernels[0].res1_n
        
        self.out = 2/self.res_n
        
        # Evolutionary algorithm vars
        self.population_n = 150
        self.max_gen = 2000
        self.req_improvs = 40
        self.mutation_chance = 0.5
        self.mutation_range = [0.5, 0.01]
    
        self.population = [np.random.uniform(-1-self.out, 1+self.out, self.kernels_n) for i in range(self.population_n)]        
        self.curr_mut_range = self.mutation_range[0]
        self.curr_gen = 0
        self.curr_improvs = 0
        self.best_loss = float('inf')
        
    def kernels_sum(self, t):
        sum_f = np.zeros(self.res_n)
        for i in range(self.kernels_n):
            sum_f += self.kernels[i].x_shift(t[i], self.res)   
        return sum_f    
    
    def pseudo_correlate(self, t):
        sum_f = self.kernels_sum(t)
        target_f = np.array([])
        if self.res==0:
            target_f = self.target.res0_f
        elif self.res==1 or self.res==2:
            target_f = self.target.res1_f
            
        pseudocorrelation = np.zeros((2,self.res_n))
        target_f = np.append(target_f, np.zeros(self.res_n-1))
        sum_f = np.append(sum_f, np.zeros(self.res_n-1))
        
        for k in range(self.res_n):
            pseudocorrelation[0][k] = np.sum(
                np.where(sum_f[0:self.res_n] > target_f[k:k+self.res_n],
                         2*target_f[k:k+self.res_n] - sum_f[0:self.res_n],
                         sum_f[0:self.res_n]))
        for k in range(self.res_n):
            pseudocorrelation[1][k] = np.sum(
                np.where(sum_f[k:k+self.res_n] > target_f[0:self.res_n],
                         2*target_f[0:self.res_n] - sum_f[k:k+self.res_n],
                         sum_f[k:k+self.res_n]))
        return pseudocorrelation

    def adjust(self):
        for i in range(self.population_n):
            correlated = self.pseudo_correlate(self.population[i])
            global_shift = np.unravel_index(correlated.argmax(), correlated.shape)
            result = self.population[i] -1 * global_shift[0] * global_shift[1] / self.res_n
            result[result > 1+self.out] = 1+self.out
            result[result < -1-self.out] = -1-self.out
            self.population[i] = result

    def loss(self, t):
        sum_f = self.kernels_sum(t)
        target_f = np.array([])
        if self.res==0:
            target_f = self.target.res0_f
        elif self.res==1 or self.res==2:
            target_f = self.target.res1_f
        return np.sum(np.absolute(sum_f - target_f)) / self.res_n

    def mutate(self, t):
        delta = np.random.uniform(-self.curr_mut_range, self.curr_mut_range)
        index = np.random.randint(0, len(t))
        result = t
        result[index] += delta
        if result[index] > 1+self.out:
            result[index] -= 1+self.out
        elif result[index] < -1-self.out:
            result[index] += 1*self.out
        return result
        
    def crossover(self, t1, t2):
        split = np.random.randint(0, np.minimum(len(t1), len(t2)))
        t = np.ndarray((np.minimum(len(t1), len(t2))))
        t[:split] = t1[:split]
        t[split:] = t2[split:]
        return t

    def select(self):
        losses = [self.loss(self.population[i]) for i in range(self.population_n)]
        inds = np.argsort(losses)
        keep = np.zeros(inds.shape)
        keep[inds < len(inds) / 2] = 1
        return [self.population[i] for i in range(self.population_n) if keep[i]]

    def evolve(self):
        # Mutation step
        do_mutate = np.random.choice(2, self.population_n, p=[1-self.mutation_chance, self.mutation_chance])
        self.population = [self.mutate(self.population[i]) if do_mutate[i] else self.population[i] for i in range(self.population_n)]
        
        # Adjust via pseudocorrelation
        #self.adjust()
        # Selection step
        self.population = self.select()
        
        # Crossover step, parents chosen at random (possibly multiple times)
        
        children_n = self.population_n - len(self.population)
        for i in range(children_n):
            parents = np.random.randint(0,self.population_n-children_n, 2)
            self.population.append(self.crossover(self.population[parents[0]],
                                                  self.population[parents[1]]))
        
        # Loss computation and next iteration/stop
        self.curr_gen += 1
        losses = [self.loss(self.population[i]) for i in range(self.population_n)]
        loss = np.amin(losses)
        if self.curr_gen % 50 == 0:
            print("Loss: " + str(loss))
        final = False
        if self.curr_gen >= self.max_gen:
            final = True
        elif self.best_loss - loss > 0:
            self.curr_improvs += 1
            self.best_loss = loss
            if self.curr_improvs >= self.req_improvs:
                final = True
        else:
            self.curr_tol = 0
        self.curr_mut_range -= (self.mutation_range[0] - self.mutation_range[1]) / self.max_gen
        if final:
            print(self.population[np.argmin(losses)])
            print(self.kernels_sum(self.population[np.argmin(losses)]))
            return self.population[np.argmin(losses)]
        else:
            self.evolve()
        
def impulse(x, pos, A):
    return (np.isclose(x, pos, atol=1/501)) * A

f_impulse = KernelFunction(np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]))
f_impulse2 = KernelFunction(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
f_target =  KernelFunction(np.array([4, 4, 0, 1, 0, 10, 5, 3, 0, -3, -1]))
f_sin = KernelFunction(lambda x : np.sin(x*10))
optimizer = KernelMatcher([f_sin, f_impulse, f_sin, f_sin, f_impulse], f_target, res=0)
t = optimizer.evolve()
