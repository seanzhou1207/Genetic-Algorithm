""" 
Module to calculate the fitness of the current generation
"""
from typing import List, Tuple
from numpy import ndarray
import numpy as np
from statsmodels.regression.linear_model import RegressionModel
import statsmodels.api as sm

class _CalculateFit:
    
    def __init__(self):
        pass
    
    def calc_fit_sort_population(self, 
        current_population: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        Calculated fitness of organisms and sorts population based on fitness score (AIC). From low AIC (best) to high.
        
        Inputs: Current population
        Outputs: Sorted population, sorted fitness scores
        """
        
        fitness_scores: ndarray = self.calculate_fit_of_population(current_population)
        return self.sort_population(current_population, fitness_scores)
    
    def sort_population(self, 
        current_population: ndarray, fitness_scores
        )-> Tuple[ndarray, ndarray]:
        """
        Sorts population based on fitness score (AIC). From low AIC (best) to high.
        
        Inputs: Current population, Fitness scores per organism
        Outputs: Sorted population, sorted fitness scores
        """
        
        sort_index = np.argsort(fitness_scores)
        
        return current_population[sort_index], fitness_scores[sort_index]
    
    
    def calculate_fit_of_population(self, 
        current_population: ndarray
        ) -> ndarray:
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
            
    def calculate_fit_per_organism(self, X_trimmed: ndarray) -> float:
        """
        Calculates fitness of one organism based on trimmed data according to its allels.
        
        Inputs: Trimmed data
        Outputs: Fitness score of organism
        """
        X_trimmed_w_intercept = sm.add_constant(X_trimmed)
        mod = self.mod(self.y, X_trimmed_w_intercept)

        # Check if the model is an instance of RegressionModel
        if not isinstance(mod, RegressionModel):
            raise TypeError(f"The model must be an instance of a statsmodels linear regression model. Instead it is {type(mod)}")
        
        #print(mod.fit().params)

        aic = mod.fit().aic
        return aic
        
    def select_features(self, organism: ndarray) -> ndarray:
        """
        Drops non-relevant features from data based on allels of an organism.
        
        Inputs: Single organism - Size: (1 x C (number of predictors))
        Outputs: Data to be used for fitness calculation of this organism
        """
      
        X_trimmed = self.X.drop(columns=self.X.columns[organism == 0], axis=1)
        #X_trimmed = self.X[:, organism != 0]

        return X_trimmed