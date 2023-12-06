import numpy as np
import pandas as pd
import random
from typing import Union

from utils import calculate_fit
from utils import parent_select
from utils import cross_over
from utils import mutate

class GA (calculate_fit.CalculateFit,
          parent_select.ParentSelection,
          cross_over.CrossOver,
          mutate.Mutation
         ):
    supported_int_types = Union[int, np.int8, np.int16, np.int32, np.int64,
                           np.uint, np.uint8, np.uint16, np.uint32, np.uint64]
    supported_float_types = Union[float, np.float16, np.float32, np.float64]
    def __init__(self, 
                X, 
                y,
                mod,
                max_iter: int,
                pop_size: int = None, 
                #fitness_func = "AIC",
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
        #self.fitness_func = fitness_func
        self.starting_population = starting_population
        self.current_population = None

        if save_sols == True:
            self.solutions_matrix = np.zeros((self.max_iter, self.C))    # Pre-specify matrix for storing solutions
        else:
            pass

    def initialize_pop(self):
        """
        Creates the starting population
        """
        if not isinstance(self.starting_population, np.ndarray):    # Specify a starting pop
            rows = self.pop_size
            if rows % 2 == 1:    # If pop_size is odd
                self.pop_size = self.pop_size + 1    # Only allow even number for population size
            
            cols = self.C
            self.starting_population = np.random.choice([0, 1], size=(rows, cols))    # Complete random generation
            
        else:
            pass
            
        self.starting_population = self.replace_zero_chromosome(self.starting_population)    # Replace chromosome of all zeros
        
        return self.starting_population

    def select(self, operator_list):
        """
        Runs variable selection based on a user-defined genetic operator sequence: operator_list
        """
        starting_pop = self.initialize_pop()
        current_pop = starting_pop.copy()

        for i in range(self.max_iter):
            # Calculates fitness and pairs parents
            chrom_ranked, fitness_val = self.calc_fit_sort_population(current_pop)
            parents = self.select_from_fitness_rank(chrom_ranked)
            current_pop = parents

            # Runs genetic operator sequence
            for method in operator_list:
                new_population = method(current_pop)
                current_pop = new_population
            # Check if any chromosome of zeros and replace the row
            current_pop = self.replace_zero_chromosome(current_pop)    
        
        final_pop = current_pop.copy()
        self.final_pop_sorted, self.final_fitness_val = self.calc_fit_sort_population(final_pop)
        
        return (self.final_pop_sorted[0], self.final_fitness_val[0])
    
    def replace_zero_chromosome(self, population):
        """
        Finds if any chromosome is all zeros, and replaces the zero rows with random 0,1s
        """
        while np.any((population == 0).all(axis=1)):
            # Find the indices of rows with all zeros
            zero_rows_indices = np.where((population == 0).all(axis=1))[0]

            # Replace each zero row with a randomly generated 0,1 row
            for row_index in zero_rows_indices:
                population[row_index] = np.random.randint(0, 2, self.C)
        
        return population
            

