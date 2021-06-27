import numpy as np
import matplotlib.pyplot as plt

from kernels import KernelFunction

# Class defining the optimizator for the problem of matching a sum of kernel functions
# to a target function. Works at a given resolution (lower is faster but rougher)
# NOTE: Cannot work in continuous resolution.
class KernelMatcher:
    def __init__(self, kernels, target, res, **kwargs):
        self.target = target
        self.res = res
        #Kernels should be provided as a Python list
        self.kernels = kernels
        self.kernels_n = len(kernels)
        
        # Function resolution expressed as a number
        self.res_n = 0
        if res==0: self.res_n = self.target.res0_n
        if res==1: self.res_n = self.target.res1_n
        
        # **kwargs here: hyperparameters for the PSO algorithm
        # The values assigned here are the default ones
        
        # PSO number of particles
        self.particle_n = 300 if not kwargs.get('particle_n') else kwargs.get('particle_n')
        # Number of generations to iterate for
        self.max_gen = 200 if not kwargs.get('max_gen') else kwargs.get('max_gen')
        # Multiplicative factor punishing optimizer for cutting kernels out of range [0, 1]
        self.cut_penalty = 1 if not kwargs.get('cut_penalty') else kwargs.get('cut_penalty')
        # PSO learning rate
        self.lr = 0.8 if not kwargs.get('lr') else kwargs.get('lr')
        # PSO w
        self.w = 0.8 if not kwargs.get('w') else kwargs.get('w')
        # PSO local and global phi
        self.phi_p = 0.8 if not kwargs.get('phi_p') else kwargs.get('phi_p')
        self.phi_g = 0.8 if not kwargs.get('phi_g') else kwargs.get('phi_g')
        # Whether to do local optimization at each step of PSO via correlation
        self.pseudo_corr = True if not kwargs.get('pseudo_corr') else kwargs.get('pseudo_corr')
        
        # Initial generation in which to perform local optimization
        # Correlation is not applied in the first generations to avoid a local optima where the kernels are
        # all cut from range [0, 1]
        self.min_gen_for_pseudo_corr = self.max_gen / 2
        # x-shift of kernels is in range [-1-out, 1+out] to allow kernels to be cut entirely
        # from contributing to target
        self.out = 2/self.res_n
        
        # Initial setup of parameters, as per PSO algorithm
        self.curr_gen = 0
        self.curr_positions = np.random.uniform(-1-self.out, 1+self.out, (self.particle_n, self.kernels_n))
        self.curr_velocities = np.random.uniform(-(2+2*self.out), 2+2*self.out, (self.particle_n, self.kernels_n))
        # Get best local in initial scenaro
        self.best_positions = np.copy(self.curr_positions)
        # Compute best global in initial scenario
        losses = [self.loss(self.curr_positions[i]) for i in range(self.particle_n)]
        best = np.argmin(losses)
        self.best_global = self.curr_positions[best]
    
    # Computes the sum of the kernel functions.
    # NOTE: to avoid overhead memory allocation, returns an array, not a KernelFunction
    def kernels_sum(self, t):
        sum_f = np.zeros(self.res_n)
        for i in range(self.kernels_n):
            sum_f += self.kernels[i].x_shift(t[i], self.res)   
        return sum_f

    # Performs local optimization step using correlation
    def adjust(self):
        # Correlation function, defined between two signals as follows:
        #   pseudocorr(f, g)(x) = sum_h(f(x) * g'(x+h)) where 
        #       { g'(x) = g(x)            if g(x) <= f(x)
        #       { g'(x) = 2*f(x) - g(x)   if g(x) > f(x)
        def _pseudo_correlate(t):
            sum_f = self.kernels_sum(t)                   # Kernel sums
            target_f = self.target.function(self.res)     # Target function
            
            pseudocorrelation = np.zeros((2,self.res_n))
            target_f = np.append(target_f, np.zeros(self.res_n-1))  # Zero padding to make pseudocorrelation works
            sum_f = np.append(sum_f, np.zeros(self.res_n-1))
            
            # Pseudocorrelation shifting the target to the left (equivalently, the kernel sum to the right)
            for k in range(self.res_n):
                pseudocorrelation[0][k] = np.sum(
                    np.where(sum_f[0:self.res_n] > target_f[k:k+self.res_n],
                             (2*target_f[k:k+self.res_n] - sum_f[0:self.res_n]) * target_f[k:k+self.res_n],
                             sum_f[0:self.res_n] * target_f[k:k+self.res_n]))
            # Pseudocorrelation shifting the kernel sum to the left
            for k in range(self.res_n):
                pseudocorrelation[1][k] = np.sum(
                    np.where(sum_f[k:k+self.res_n] > target_f[0:self.res_n],
                             (2*target_f[0:self.res_n] - sum_f[k:k+self.res_n]) * target_f[0:self.res_n],
                             sum_f[k:k+self.res_n] * target_f[0:self.res_n]))
            return pseudocorrelation
        
        # For each PSO particle...
        for i in range(self.particle_n):
            # Compute pseudocorrelation
            correlated = _pseudo_correlate(self.curr_positions[i])
            # Find x-shift which maximises pseudocorrelation
            global_shift = list(np.unravel_index(correlated.argmax(), correlated.shape))
            global_shift[0] = global_shift[0] * 2 - 1
            # Perform a global shift of all kernels by the found amount
            result = self.curr_positions[i] -1 * global_shift[0] * global_shift[1] / self.res_n
            # Limit the shift in range [-1-out; +1+out]
            result[result > 1+self.out] = 1+self.out
            result[result < -1-self.out] = -1-self.out
            # Set new particle position
            self.curr_positions[i] = result
    
    # Loss function for PSO optimization
    def loss(self, t, cut_penalty=True):
        cut_loss = 0
        sum_f = np.zeros(self.res_n)
        # Sum each kernel together
        for i in range(self.kernels_n):
            shift_f = self.kernels[i].x_shift(t[i], self.res)
            sum_f += shift_f
            # If the penalty for cutting a kernel outside range should be used, compute it
            if cut_penalty:
                orig_f = self.kernels[i].function(self.res)
                cut_loss += abs(np.sum(abs(shift_f)) - np.sum(abs(orig_f))) / np.sum(abs(orig_f)) / self.kernels_n
        # Get the target function
        target_f = self.target.function(self.res)
        if cut_penalty:
            # Loss with cut penalty is a weighted sum of terms, one which is given by how much the kernels are cut
            return self.cut_penalty * cut_loss + np.sum(np.square(sum_f - target_f)) / self.res_n
        else:
            # Default loss is difference of kernels sum and target
            return np.sum(np.square(sum_f - target_f)) / self.res_n
    
    # Perform a PSO iteration
    def iterate(self, verbose=True, ret_score=False, plot=True, plot_res=1):
        final = False
        # Inner loop
        while not final:
            # PSO iteration
            
            # Init random values
            r_p = np.random.uniform(0,1, (self.particle_n, self.kernels_n))
            r_g = np.random.uniform(0,1, (self.particle_n, self.kernels_n))
            
            # Compute new velocities
            self.curr_velocities = self.w * self.curr_velocities + \
                                    self.phi_p * r_p * (self.best_positions - self.curr_positions) + \
                                    self.phi_g * r_g * (self.best_global - self.curr_positions)
            
            # Compute new positions, pad them if they are outside of valid range
            self.curr_positions += self.lr * self.curr_velocities
            self.curr_positions = np.where(self.curr_positions > 1+self.out, -1-self.out+self.curr_positions, self.curr_positions)
            self.curr_positions = np.where(self.curr_positions < -1-self.out, 1+self.out+self.curr_positions, self.curr_positions)          
            
            # Adjust results (local improvement) via pseudocorrelation
            if self.pseudo_corr and self.curr_gen >= self.min_gen_for_pseudo_corr:
                self.adjust()
            
            # Update local bests and global best
            for i in range(self.particle_n):
                if self.loss(self.curr_positions[i]) < self.loss(self.best_positions[i]):
                    self.best_positions[i] = self.curr_positions[i]
                    if self.loss(self.best_positions[i]) < self.loss(self.best_global):
                        self.best_global = self.best_positions[i]
            
            # Print current loss, next iteration/stop
            if verbose and self.curr_gen % 25 == 0:
                loss = self.loss(self.best_global)
                print("Loss " + str(self.curr_gen) + ": " + str(loss))
            self.curr_gen += 1        
            if self.curr_gen >= self.max_gen:
                final = True
        
        # Get final loss
        loss = self.loss(self.best_global, cut_penalty=False)
        print("Final loss: " + str(loss))
        self.best_global = np.around(self.best_global, self.target.precision_digits)
        # If plotting is required, plot the kernel sum and the target together
        if plot:
            self.target.plot(show=False, res=plot_res)
            smooth=False
            if plot_res==1:
                smooth=True
            KernelFunction(self.kernels_sum(self.best_global), smooth=smooth).plot(show=False)
            plt.legend(['Target', 'Kernels Sum'])
            plt.show()
        # Return the x-shifts for the kernels and optionally the loss
        if ret_score:
            return self.best_global, loss
        else:
            return self.best_global