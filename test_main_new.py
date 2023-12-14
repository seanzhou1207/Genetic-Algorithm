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


@pytest.fixture(autouse=True, scope="module", params=[_ for _ in range(41, 71)])
def seed_everything(request):
    """
    test on different random seeds
    """
    seed = request.param
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# NOTE: think this is pretty meaningless
# def test_initialization():
#     """
#     Test to see if the class GA works with given parameters
#     """
#     X = np.random.rand(100, 10)  # Example dataset with 100 samples, 10 Variables
#     y = np.random.rand(100)  # Dependant Variable
#     mod = sm.OLS  # OLS Regression
#
#     ga = GA(X, y, mod, max_iter=100, pop_size=20, mutate_prob=0.01)
#
#     assert ga.pop_size == 20
#     assert ga.max_iter == 100
#     assert ga.mutate_prob == 0.01
#     assert ga.X.shape == (100, 10)
#     assert ga.y.shape == (100,)


def test_population_initialization():
    """Test `GA::initialize_pop` method"""
    data_sample_size = random.randint(100, 200)
    data_feature_size = random.randint(200, 500)
    X = np.random.rand(data_sample_size, data_feature_size)
    y = np.random.rand(data_sample_size)

    max_iter = random.randint(10, 100)
    pop_size = random.randint(30, 100)

    mod = sm.OLS
    ga = GA(X, y, mod, max_iter=max_iter, pop_size=pop_size)
    initial_pop = ga.initialize_pop()

    assert X.shape[-1] == ga.C, f"{X.shape[-1]} != {ga.C}"
    assert initial_pop.shape == (
        ga.pop_size,
        X.shape[-1],
    ), f"{initial_pop.shape} == {(pop_size, X.shape[-1])}"
    assert not np.any(
        (initial_pop == 0).all(axis=1)
    ), "Individual with all zero chromosome detected. check if `GA::replace_zero_chromosome` is working properly."


# NOTE: think this is pretty meaningless
# def test_selection_process():
#     """
#     Tests select method
#     """
#     X = np.random.rand(100, 10)
#     y = np.random.rand(100)
#     mod = sm.OLS
#
#     ga = GA(X, y, mod, max_iter=10, pop_size=20)
#     best_solution, best_fitness = ga.select([GA.random_mutate])
#
#     assert isinstance(best_solution, np.ndarray)
#     assert isinstance(best_fitness, (float, int))


def test_random_mutate():
    """
    Tests random_mutate method
    """
    data_sample_size = random.randint(100, 200)
    data_feature_size = random.randint(200, 500)
    X = np.random.rand(data_sample_size, data_feature_size)
    y = np.random.rand(data_sample_size)

    max_iter = random.randint(10, 100)
    pop_size = random.randint(30, 100)

    mod = sm.OLS
    ga = GA(X, y, mod, max_iter=max_iter, pop_size=pop_size, mutate_prob=1.0)
    initial_pop = ga.initialize_pop()
    mutated_pop = ga.random_mutate(initial_pop, ga.mutate_prob)

    assert (
        mutated_pop.shape == initial_pop.shape
    ), f"{mutated_pop.shape} != {initial_pop.shape}"
    assert np.allclose(
        initial_pop, 1 - mutated_pop
    ), "If mutate_prob=1.0, `mutated_pop` should be an invert of `initial_pop` tensor"


def test_split_and_glue_population():
    """
    Tests split_and_glue_population method
    """
    feature_size = random.randint(20, 40)
    initial_pop = np.stack([np.ones(feature_size), np.zeros(feature_size)])
    crossed_pop = GA.split_and_glue_population(initial_pop)

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    sorted_tensor = np.cumsum(crossed_pop, axis=1) / np.arange(
        1, crossed_pop.shape[-1] + 1
    )
    if sorted_tensor[0][0] == 0:
        assert is_sorted(
            sorted_tensor[0]
        ), f"There is more than one crossover point\ninitial_pop: {initial_pop}\ncrossed_pop: {crossed_pop}"
        assert is_sorted(
            sorted_tensor[1][::-1]
        ), f"There is more than one crossover point\ninitial_pop: {initial_pop}\ncrossed_pop: {crossed_pop}"
    else:
        assert is_sorted(
            sorted_tensor[1]
        ), f"There is more than one crossover point\ninitial_pop: {initial_pop}\ncrossed_pop: {crossed_pop}"
        assert is_sorted(
            sorted_tensor[0][::-1]
        ), f"There is more than one crossover point\ninitial_pop: {initial_pop}\ncrossed_pop: {crossed_pop}"


def test_random_allel_selection_population():
    """
    Tests random_allel_selection_population method
    """
    feature_size = random.randint(40, 80)
    initial_pop = np.stack([np.ones(feature_size), np.zeros(feature_size)])
    crossed_pop = GA.random_allel_selection_population(initial_pop)

    mean_tensor = crossed_pop.mean(axis=-1)
    assert np.all(
        (0.0 < mean_tensor) & (mean_tensor < 1.0)
    ), f"All zeros or all ones detected. Check the boolean indexing of the crossover function.\ninitial_pop: {initial_pop}\ncrossed_pop: {crossed_pop}"


def test_simple_ga_problem1():
    def _simulate_dataset(num_samples=100, num_features=100) -> Tuple[ndarray, ndarray]:
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

    data_sample_size = random.randint(100, 200)
    data_feature_size = random.randint(20, 50)
    X, y = _simulate_dataset(data_sample_size, data_feature_size)

    # OLS Regression
    mod = sm.OLS
    ga = GA(X, y, mod, max_iter=50, pop_size=30, mutate_prob=0.01)

    # Run GA to get best solution
    # Store best solution in a list
    best_solution, _ = ga.select()

    did_ga_favor_first_half = (
        best_solution[: data_feature_size // 1].sum()
        > best_solution[data_feature_size // 2 :].sum()
    )
    assert did_ga_favor_first_half, "GA did not favor the first half of the features"


def test_simple_ga_problem2():
    """
    goal: maximize sum of chromosome (reduce zeros in chromosome)
    e.g.
    [0, 0, 0, 0, 1] => fitness_score: 4
    [0, 1, 1, 0, 0] => fitness_score: 3
    [0, 1, 1, 0, 1] => fitness_score: 2
    [1, 1, 1, 1, 1] => fitness_score: 0
    """
    # setup datapoints with no correlation
    feature_size = 10
    X = np.random.randn(10, feature_size)
    y = np.random.randn(10) / np.random.randn(10)

    class MyMod:
        def fit(self):
            return self

        def __call__(self, _, X):
            """
            e.g.
            given
            sample dataset X(trimmed_X): (dataset_size, sampled_feature_size)

            minimize error
            """
            self.aic = feature_size - X.shape[-1]  # number of zeros in individual
            return self

    mod = MyMod()
    max_iter = random.randint(250, 300)
    pop_size = random.randint(100, 200)

    # 0 ~ 0.3, if mutate_prob too high => converge slower
    mutate_prob = random.random() * 0.3
    ga = GA(
        X,
        y,
        mod,
        max_iter=max_iter,
        pop_size=pop_size,
        mutate_prob=mutate_prob,
        exploit=True,
    )

    best_solution, best_fitness = ga.select()  # use default setting

    assert (
        best_fitness == 0.0
    ), f"GA did not converge on simple example... may be an algorithmic problem\nfinal_output: {best_solution}"
