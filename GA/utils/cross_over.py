""" 
Module to perform crossover (genetic operator) to the current generation
"""
import numpy as np
from numpy import ndarray

class _CrossOver:
    
    def __init__(self):
        pass

    @staticmethod
    def split_and_glue_population(current_population: ndarray) -> ndarray:
        """
        Performs split-and-glue crossover to current population (assuming 1&2 is paired, 3&4, etc.)
       
        Inputs: Current population
        Outputs: Population of children (pairwise cross-over)
        """
        count = 0
        new_population = np.zeros(current_population.shape).astype(int)
        for pair in np.arange(int(current_population.shape[0]/2)):
            new_population[count], new_population[count+1] = _CrossOver.split_and_glue(current_population[count], current_population[count+1])
            count += 2
        return new_population

    @staticmethod
    def split_and_glue(parent1, parent2):
        """
        Crossover two parents to create two children. 
        The method used here is a simple split and glue approach. 
        The split position is randomly created.
        
        Inputs: Two parent organisms
        Outputs: Two child organisms (crossed-over)
        """
        cut_idx = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[0:cut_idx], parent2[cut_idx:]))
        child2 = np.concatenate((parent2[0:cut_idx], parent1[cut_idx:]))
        return child1, child2
    
    @staticmethod
    def random_allel_selection_population(current_population: ndarray) -> ndarray:
        """
        Performs random allel selection crossover to current population (assuming 1&2 is paired, 3&4, etc.)
       
        Inputs: Current population
        Outputs: Population of children (pairwise cross-over)
        """
        count = 0
        new_population = np.zeros(current_population.shape)
        for pair in np.arange(int(current_population.shape[0]/2)):
            new_population[count], new_population[count+1] = _CrossOver.random_allel_selection(current_population[count], current_population[count+1])
            count += 2
        return new_population
    
    @staticmethod
    def random_allel_selection(parent1, parent2):
        """
        Crossover two parents to create two children. 
        The method randomly selects an allel from one of the parents per loci.
        
        Inputs: Two parent organisms
        Outputs: Two child organisms (crossed-over)
        """
        rng = np.random.default_rng()
        
        allel_selector = rng.binomial(1, 0.5, size = parent1.shape[0])
        allel_selector_reversed = 1-allel_selector
        
        child1 = allel_selector*parent1 + allel_selector_reversed*parent2
        child2 = allel_selector*parent2 + allel_selector_reversed*parent1
        return child1, child2