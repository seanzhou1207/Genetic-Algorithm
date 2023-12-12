"""
To execute the TEST
pytest test_main.py -v
"""
import os
import random
from typing import Tuple

import numpy as np
import pytest
import statsmodels.api as sm
from numpy import ndarray

from GA import GA
import pandas as pd


@pytest.fixture(autouse=True, scope="module", params=[_ for _ in range(41, 71)])
def seed_everything(request):
    """
    test on different random seeds
    """
    seed = request.param
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_initialization():
    """
    Test to see if the class GA works with given parameters
    """
    X = np.random.rand(100, 10)  # Example dataset with 100 samples, 10 Variables
    y = np.random.rand(100)  # Dependant Variable
    mod = sm.OLS  # OLS Regression

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
    best_solution, best_fitness = ga.select([GA.random_mutate])

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
    mutated_pop = ga.random_mutate(initial_pop, ga.mutate_prob)

    assert mutated_pop.shape == initial_pop.shape


def test_ga_feature_selection():
    def simulate_dataset(num_samples=100, num_features=100) -> Tuple[ndarray, ndarray]:
        """
        Simulate a dataset where the first half of the variables are relevant and the second half are not.
        returns:
            X: (num_samples, num_features) fp64 matrix
            y: (num_samples,) fp64 vector
        """
        X = np.random.rand(num_samples, num_features)

        # Target depends on the first half of the features
        y = np.sum(X[:, : num_features // 2], axis=1)

        return X, y

    num_samples, num_features = 100, 100
    X, y = simulate_dataset(num_samples, num_features)

    # OLS Regression
    mod = lambda y, X: sm.OLS(y, X)

    ga = GA(X, y, mod, max_iter=50, pop_size=30, mutate_prob=0.01)

    # Run GA to get best solution
    # Store best solution in a list
    best_solution, _ = ga.select([])

    did_ga_favor_first_half = (
        best_solution[: num_features // 2].sum()
        > best_solution[num_features // 2 :].sum()
    )
    assert did_ga_favor_first_half, "GA did not favor the first half of the features"


def test_simple_ga_problem():
    """
    goal: maximize sum of chromosome (reduce zeros in chromosome)
    e.g.
    [0, 0, 0, 0, 1] => fitness_score: 4
    [0, 1, 1, 0, 0] => fitness_score: 3
    [0, 1, 1, 0, 1] => fitness_score: 2
    [1, 1, 1, 1, 1] => fitness_score: 0
    """
    feature_size = 20
    X = np.random.randint(0, 2, size=(120, feature_size))
    y = X.sum(axis=1)
    
    data = pd.DataFrame(X, columns=column_names)
    X = data
    data['y'] = y


    class MyMod:
        def fit(self):
            return self

        def __call__(self, y, X):
            """
            e.g.
            given
            sample dataset X: (dataset_size, sampled_feature_size)
            sample label y: (dataset_size,)

            minimize error
            """
            self.aic = feature_size - X.shape[-1]
            return self

    mod = MyMod()

    ga = GA(X, y, mod, max_iter=300, pop_size=None, mutate_prob=0.01)

    best_solution, best_fitness = ga.select()  # use default setting

    assert (
        best_fitness == 0.0
    ), f"GA did not converge on simple example... may be an algorithmic problem\nfinal_output: {best_solution}"