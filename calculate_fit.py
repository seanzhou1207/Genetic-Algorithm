""" 
Module to calculate the fitness of the current generation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

class calculate_fit:
    
    def __init__():
        pass
    
    

    def calculate_fit_per_organism(self, X_trimmed):
         """
        Calculates fitness of one organism based on trimmed data according to its allels.
        
        Inputs: Trimmed data
        Outputs: Fitness score of organism
        
        """
        
        mod_fitted = self.mod(self.y, X_trimmed).fit()
        
        if self.fitness_func == "AIC":
            fitness = mod_fitted.aic
        else:
            print("Fitness function is not AIC")
            pass
            
        return fitness
        
        
    def select_features(self, organism):
        """
        Drops non-relevant features from data based on allels of an organism.
        
        Inputs: Single organism - Size: (1 x C (number of predictors))
        Outputs: Data to be used for fitness calculation of this organism
        
        """
      
    X_trimmed = self.X.drop(columns=X.columns[organism == 0], axis=1)
    
    return X_trimmed