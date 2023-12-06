""" 
Mutation module with genetic operator "mutation"
"""

import numpy as np

class Mutation:
    
    def __init__():
        pass

    def random_mutate(self, current_population):
        
        """
        Randomly switches genes (bit switch) in generation with probability mutate_prob

        Inputs: Generation of organisms - Size: (pop_size x C (number of predictors))
        Outputs: Generation of mutated organisms of same size
        
        """

        # initialize random generator
        rng = np.random.default_rng()
        
        population_new = current_population.copy()
        mutation_locations = rng.binomial(1, self.mutate_prob, size = current_population.shape)
        mask = mutation_locations == 1
        population_new[mask] = 1 - population_new[mask] # flip bits using the mask
        return population_new
