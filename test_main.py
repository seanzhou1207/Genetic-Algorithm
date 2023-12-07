import pytest
import statsmodels.api as sm
import numpy as np
from gavs import GA



def test_initialization():
    """
    Test to see if the class GA works with given parameters
    """
    X = np.random.rand(100, 10)  # Example dataset with 100 samples, 10 Variables
    y = np.random.rand(100)      # Dependant Variable
    mod = sm.OLS                 # OLS Regression

    ga = GA(X, y, mod, max_iter=100, pop_size=20, mutate_prob=0.01)

    assert ga.pop_size == 20
    assert ga.max_iter == 100
    assert ga.mutate_prob == 0.01
    assert ga.X.shape == (100, 10)
    assert ga.y.shape == (100,)

def test_population_initialization():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    mod = sm.OLS

    ga = GA(X, y, mod, max_iter=100, pop_size=20)
    initial_pop = ga.initialize_pop()

    assert initial_pop.shape == (20, 10) 
    assert not np.any((initial_pop == 0).all(axis=1))  # No array should be all zeros

def test_selection_process():
    """
    Tests select method
    """
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    mod = sm.OLS

    ga = GA(X, y, mod, max_iter=10, pop_size=20)
    best_solution, best_fitness = ga.select([ga.random_mutate])  

    assert isinstance(best_solution, np.ndarray)
    assert isinstance(best_fitness, (float, int))


def test_mutation():
    """
    Tests random_mutate method
    """
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    mod = sm.OLS

    ga = GA(X, y, mod, max_iter=100, pop_size=20, mutate_prob=0.1)
    initial_pop = ga.initialize_pop()
    mutated_pop = ga.random_mutate(initial_pop)

    assert mutated_pop.shape == initial_pop.shape
    

def simulate_dataset(num_samples=100, num_features=100):
    """
    Simulate a dataset where the first half of the variables are relevant and the second half are not.
    """
    X = np.random.rand(num_samples, num_features)  
    y = np.sum(X[:, :num_features // 2], axis=1)   # Target depends on the first half of the features
    return X, y

def test_ga_feature_selection():
    num_samples, num_features = 100, 100
    X, y = simulate_dataset(num_samples, num_features)
    
    # OLS Regression
    mod = lambda y, X: sm.OLS(y, X)  

    ga = GA(X, y, mod, max_iter=50, pop_size=30, mutate_prob=0.01)

    # Run GA to get best solution
    # Store best solution in a list 
    best_solution, _ = ga.select([])

    did_ga_favor_first_half = best_solution[:num_features // 2].sum() > best_solution[num_features // 2:].sum()
    assert did_ga_favor_first_half, "GA did not favor the first half of the features"


