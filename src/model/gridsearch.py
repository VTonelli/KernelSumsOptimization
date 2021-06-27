import json
import numpy as np

from kernels import KernelFunction
import kernels
from optimizer import KernelMatcher

# Class used to perform a grid search on the space of the PSO parameters
class KernelCV:
    # subclass JSONEncoder
    class Encoder(json.JSONEncoder):
            def default(self, o):
                return o.__dict__
    
    # Constructor
    def __init__(self):
        self.losses = []
        self.curr_counters = {}
        # Fixed parameters
        parameters = {
            'cut_penalty': [0.3, 0.75, 1, 1.5],
            'w':[0.8, 0.5, 0.3],
            'phi_p':[0.8, 0.5, 0.3],
            'phi_g':[0.8, 0.5, 0.3]
        }
        # Read data from a file, to resume grid search from last checkpoint
        try:
            f = open("gridsearch.txt", "r")
        # If file could not be opened, set values to 0
        except OSError:
            self.curr_counters = {}
            for key in parameters.keys():
                self.curr_counters[key] = 0
        else:
            # If file has been read successfully, use JSON to load data
            string = f.read()
            f.close()
            self.curr_counters = json.loads(string)['curr_counters']
            self.losses = json.loads(string)['losses']
    
    # Perform grid search
    def grid_search(self):
        # Fixed parameters
        parameters = {
            'cut_penalty': [0.3, 0.75, 1, 1.5],
            'w':[0.8, 0.5, 0.3],
            'phi_p':[0.8, 0.5, 0.3],
            'phi_g':[0.8, 0.5, 0.3]
        }
        tries = 1
        counters = {}
        # Get number of values for each parameters and compute total number of combinations
        for key in parameters.keys():
            counters[key] = len(parameters[key])
            tries *= len(parameters[key])
        
        # Print completion percentage of grid search
        print("Done " + "{:.2f}".format(len(self.losses) / tries * 100) + "%")
        # Define a fixed target function
        f_target =  KernelFunction(np.array([1, 3, 2, 3, 6, 0, 0, 6, 5, 4, 2]), smooth=True)
        # Define types of kernel function
        f_cos = KernelFunction(lambda x: kernels.cos(x, 2))
        f_hill = KernelFunction(lambda x : kernels.hill(x, 4))
        f_impulse = KernelFunction(lambda x : kernels.impulse(x, 0, 1))
        final = False
        # Inner loop: try one combination of parameters
        try:
            while not final:
                    # If the counters for combination has reset and we have computed losses, end
                    if np.all(np.array(list(self.curr_counters.values())) == 0) and len(self.losses) != 0:
                        final = True
                        break
                    # Get values for parameters given current counters
                    iteration_dict = {}
                    for key in counters.keys():
                        iteration_dict[key] = parameters[key][self.curr_counters[key]]
                    print(iteration_dict)
                    avg_score = 0
                    # Perform the optimization 3 times, take the average loss
                    for i in range(3):
                        optimizer = KernelMatcher([f_hill for i in range(12)] + [f_impulse for i in range(4)] + [f_cos for i in range(8)], f_target, res=0,
                                                  **iteration_dict)
                        _, score = optimizer.iterate(verbose=False, ret_score=True, plot=False)
                        avg_score += score
                    avg_score /= 3.0
                    # Add the loss to the list
                    self.losses.append(avg_score)
                    # Update counters
                    for (key,value) in zip(self.curr_counters.keys(), self.curr_counters.values()):
                        if value < counters[key] - 1:
                            self.curr_counters[key] += 1
                            break
                        else:
                            self.curr_counters[key] = 0
        
            # Fetch the optimal combination of parameters from a flat index to an unravelled index to a dictionary
            counter_list = []
            for key in counters.keys():
                counter_list.append(counters[key])
            combination = np.unravel_index(np.array(self.losses).argmin(), np.array(counter_list))
            combination = np.flip(combination)
            combination_dict = {}
            i = 0
            for key in parameters.keys():
                combination_dict[key] = parameters[key][combination[i]]
                i += 1
            # Print the optimal parameters combination
            print("Optimal params: "+ str(combination_dict))
            print("Grid Search Finished.")
            data = json.dumps(self, indent=4, cls=self.Encoder)
            f = open("gridsearch.txt", "w")
            f.write(data)
            f.close()
        # On keyboard interrupt, dump current data as JSON into a file (which can then be read when running the function again)
        except KeyboardInterrupt:
            data = json.dumps(self, indent=4, cls=self.Encoder)
            f = open("gridsearch.txt", "w")
            f.write(data)
            f.close()

# Run this file to perform grid search
KernelCV().grid_search()