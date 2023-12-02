import numpy as np
import pandas as pd
import random
from typing import Union

class GA ():
    supported_int_types = Union[int, np.int8, np.int16, np.int32, np.int64,
                           np.uint, np.uint8, np.uint16, np.uint32, np.uint64]
    supported_float_types = Union[float, np.float16, np.float32, np.float64]
    def __init__(self, 
                X, 
                y,
                mod,
                max_iter: int,
                pop_size: int = None, 
                fitness_func = "AIC",
                starting_population = None,
                mutate_prob = 0.01,
                save_sols = False,
                random_seed = None,
                ):
        self.random_seed = random_seed
        if not random_seed:
            pass
        else:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        self.C = X.shape[1]    # CHECK: this is assuming intercept column

        if not pop_size:
            self.pop_size = int(1.5 * self.C)    # C < P < 2C
        else:
            self.pop_size = pop_size

        self.X = X
        self.y = y
        self.mod = mod
        self.max_iter = max_iter
        self.mutate_prob = mutate_prob
        self.fitness_func = fitness_func
        self.starting_population = starting_population
        self.current_population = None

        if save_sols == True:
            self.solutions_matrix = np.zeros((self.max_iter, self.C))    # Pre-specify matrix for storing solutions
        else:
            pass

    def initialize_pop(self):
        if not self.starting_population:    # Specify a starting pop
            rows = self.pop_size
            cols = self.C
            self.starting_population = np.random.choice([0, 1], size=(rows, cols))    # Complete random generation
            
            return self.starting_population
        else:
            return self.starting_population
 
    def calculate_fit(self):
        pass

    def choose_parents(self):
        pass

    def cross_over(self):
        pass

    def mutate(self):
        pass
    
    def select(self):
        pass
