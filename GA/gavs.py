import random
from functools import partial
from typing import Callable, List
import statsmodels.api

import numpy as np
from numpy import ndarray

from GA.utils import _CalculateFit, _CrossOver, _Mutation, _ParentSelection

class GA (_CalculateFit,
          _ParentSelection,
          _CrossOver,
          _Mutation
         ):
    def __init__(
        self, 
        X: ndarray,
        y: ndarray,
        mod: Callable,
        max_iter: int,
        pop_size: int = None,  # type: ignore
        # fitness_func = "AIC",
        starting_population: ndarray = None,  # type: ignore
        mutate_prob: float = 0.01,
        save_sols: bool = False,
        random_seed: int = None,  # type: ignore
        ):
        """
        parameters:
        --------
            X: design matrix (assuming no intercept column)
            y: outcome variable
            mod: regression model (statsmodels)
            max_iter: GA max iteration
            pop_size: GA population size
            starting_population: if set use it as initial GA population
            mutate_prob: GA mutation probability
            save_sols: ... TODO
            random_seed: random seed value

        examples:
        --------
   ...: from GA import *
   ...: import statsmodels.api as sm
   ...: import numpy as np
   ...: 
   ...: spector_data = sm.datasets.spector.load()
   ...: 
   ...: X = spector_data.exog
   ...: y = spector_data.endog
   ...: 
   ...: # Initialize GA class
   ...: ga_1 = GA(X=X, y=y, mod=sm.OLS, max_iter=10, random_seed=1)
   ...: 
   ...: # Run GA under default operators
   ...: final_pop, fit = ga_1.select()
   ...: print(final_pop, fit)

   ...: # Specify own operator, population size, and mutation probability
   ...: operator = [GA.random_mutate, GA.random_mutate, GA.split_and_glue_population]
   ...: ga_2 = GA(X=X, y=y, mod=sm.OLS, max_iter=10, pop_size = 4, mutate_prob=0.01, random_seed=12) TODO
   ...: final_pop, fit = ga_2.select(operator)
   ...: print(final_pop, fit)   
        """
        self.random_seed: int = random_seed
        if random_seed:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        self.C: int = X.shape[1]    # this is ASSUMING NO intercept column

        if pop_size is None:
            self.pop_size: int = int(1.5 * self.C)    # C < P < 2C
        else:
            self.pop_size: int = pop_size

        self.X: ndarray = X
        self.y: ndarray = y
        self.mod: Callable = mod
        self.max_iter: int = max_iter
        self.mutate_prob: float = mutate_prob
        # self.fitness_func = fitness_func
        self.starting_population: ndarray = starting_population
        self.current_population = None

        if save_sols == True:
            self.solutions_matrix = np.zeros((self.max_iter, self.C))    # Pre-specify matrix for storing solutions
        else:
            pass

    def initialize_pop(self):
        """
        Creates the starting population
        returns:
            starting_population: ndarray (random bool matrix used to sample self.X)
        """
        if not isinstance(self.starting_population, ndarray):    # Specify a starting pop
            if self.pop_size % 2 == 1:    # If pop_size is odd
                self.pop_size = self.pop_size + 1    # Only allow even number for population size
                print(f"Original pop_size is odd - new pop_size: {self.pop_size}")
            
            cols = self.C
            self.starting_population = np.random.choice([0, 1], size=(self.pop_size, cols))    # Complete random generation
            
        else:
            pass
            
        self.starting_population = self.replace_zero_chromosome(self.starting_population)    # Replace chromosome of all zeros
        
        return self.starting_population

    def select(self, operator_list: List[Callable] = None):
        """
        Runs variable selection based on a user-defined genetic operator sequence: operator_list            
        """
        """Set random seed"""
        random.seed(self.random_seed)

        # set default mutation methods
        # assigns user specified operator_list if its not None
        operator_list = operator_list or [
            self.split_and_glue_population,
            self.random_mutate,
            ]       
        #for i, f in enumerate(operator_list):
        #    if f.__name__ == "random_mutate":
        #        operator_list[i] = partial(f, mutate_prob=self.mutate_prob)
        print(f"Using genetic operators: {operator_list}.")

        """Prepare GA"""
        starting_pop = self.initialize_pop()
        current_pop = starting_pop.copy()

        for i in range(self.max_iter):
            """Calculates fitness and pairs parents"""
            # chrom_ranked: ordered bool matrix(current_pop) from the fittest to unfittest
            chrom_ranked, fitness_val = self.calc_fit_sort_population(current_pop)
            parents = self.select_from_fitness_rank(chrom_ranked)
            current_pop = parents    # update current_pop's chromosome
            print(f"[iteration {i+1}] score: {fitness_val[0]:3.4f} | {chrom_ranked[0]}")

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
            

