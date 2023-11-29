import numpy
import random

class GA ():
    def __init__(self, 
                pop_size, 
                num_generations,
                model,
                fitness_func = "AIC",
                starting_population = None
                ):
        self.Y = model.endog
        pass

    def initialize_pop(self):
        pass

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
