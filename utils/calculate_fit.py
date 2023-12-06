""" 
Module to calculate the fitness of the current generation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

class CalculateFit:
    
    def __init__():
        pass
    
    def calc_fit_sort_population(self, current_population):
        """
        Calculated fitness of organisms and sorts population based on fitness score (AIC). From low AIC (best) to high.
        
        Inputs: Current population
        Outputs: Sorted population, sorted fitness scores
        """
        
        fitness_scores = self.calculate_fit_of_population(current_population)
        return self.sort_population(current_population, fitness_scores)
    
    def sort_population(self, current_population, fitness_scores):
        """
        Sorts population based on fitness score (AIC). From low AIC (best) to high.
        
        Inputs: Current population, Fitness scores per organism
        Outputs: Sorted population, sorted fitness scores
        """
        
        sort_index = np.argsort(fitness_scores)
        
        return current_population[sort_index], fitness_scores[sort_index]
    
    
    def calculate_fit_of_population(self, current_population):
        """
        Calculates fitness of all organism in generation.
        
        Inputs: Current population
        Outputs: Fitness score per organism
        """
        fitness_scores = []
        for organism in current_population:
            X_trimmed = self.select_features(organism)
            fitness_scores.append(self.calculate_fit_per_organism(X_trimmed))
        return np.array(fitness_scores)
            
    def calculate_fit_per_organism(self, X_trimmed):
        """
        Calculates fitness of one organism based on trimmed data according to its allels.
        
        Inputs: Trimmed data
        Outputs: Fitness score of organism
        """
        mod_fitted = self.mod(self.y, X_trimmed).fit() 
        return mod_fitted.aic
        
    def select_features(self, organism):
        """
        Drops non-relevant features from data based on allels of an organism.
        
        Inputs: Single organism - Size: (1 x C (number of predictors))
        Outputs: Data to be used for fitness calculation of this organism
        """
      
        X_trimmed = self.X.drop(columns=self.X.columns[organism == 0], axis=1)
        return X_trimmed