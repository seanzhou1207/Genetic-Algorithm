import os
import random
import numpy as np
import pytest
import statsmodels.api as sm
from GA import GA

# Fixture for setting different random seeds
@pytest.fixture(autouse=True, scope="module", params=[_ for _ in range(41, 71)])
def seed_everything(request):
    """
    Test on different random seeds.
    """
    seed = request.param
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# Fixture for setting up the GA instance
@pytest.fixture
def setup_ga():
    X = np.random.rand(100, 10)  # Mock dataset
    y = np.random.rand(100)      # Mock target variable
    mod = sm.OLS                 # OLS Regression
    return GA(X, y, mod, max_iter=100, pop_size=20, mutate_prob=0.01)

# Test for GA class initialization
def test_ga_initialization():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    mod = sm.OLS
    ga = GA(X, y, mod, max_iter=10, pop_size=20, mutate_prob=0.01, random_seed=42)
    assert ga.pop_size == 20
    assert ga.max_iter == 10
    assert ga.mutate_prob == 0.01

# Test for population initialization
def test_initialize_pop(setup_ga):
    ga = setup_ga
    initial_pop = ga.initialize_pop()
    assert initial_pop.shape == (ga.pop_size, ga.C)
    assert not np.any((initial_pop == 0).all(axis=1))

# Test for main selection process
def test_select(setup_ga):
    ga = setup_ga
    final_pop, fit = ga.select()
    assert isinstance(final_pop, np.ndarray)
    assert isinstance(fit, float)

# Test for replacing zero chromosomes
def test_replace_zero_chromosome(setup_ga):
    ga = setup_ga
    population_with_zeros = np.zeros((ga.pop_size, ga.C))
    no_zero_population = ga.replace_zero_chromosome(population_with_zeros)
    assert not np.any((no_zero_population == 0).all(axis=1))

# _CalculateFit tests
def test_calc_fit_sort_population(setup_ga):
    ga = setup_ga
    mock_population = np.random.randint(2, size=(ga.pop_size, ga.C))
    sorted_pop, sorted_scores = ga.calc_fit_sort_population(mock_population)
    assert np.all(sorted_scores[:-1] <= sorted_scores[1:]), "Population not sorted correctly."

def test_calculate_fit_per_organism(setup_ga):
    ga = setup_ga
    organism = np.random.randint(2, size=ga.C)
    X_trimmed = ga.select_features(organism)
    fitness = ga.calculate_fit_per_organism(X_trimmed)
    assert isinstance(fitness, float), "Fitness score should be a float."

def test_select_features(setup_ga):
    ga = setup_ga
    organism = np.random.randint(2, size=ga.C)
    X_trimmed = ga.select_features(organism)
    assert X_trimmed.shape[1] == np.sum(organism), "Incorrect number of features selected."

# _CrossOver tests
def test_split_and_glue_population(setup_ga):
    ga = setup_ga
    mock_population = np.random.randint(2, size=(ga.pop_size, ga.C))
    new_population = ga.split_and_glue_population(mock_population)
    assert new_population.shape == mock_population.shape, "Incorrect population size after crossover."

def test_random_allel_selection_population(setup_ga):
    ga = setup_ga
    mock_population = np.random.randint(2, size=(ga.pop_size, ga.C))
    new_population = ga.random_allel_selection_population(mock_population)
    assert new_population.shape == mock_population.shape, "Incorrect population size after crossover."

# _Mutation test
def test_random_mutate(setup_ga):
    ga = setup_ga
    mock_population = np.random.randint(2, size=(ga.pop_size, ga.C))
    mutated_population = ga.random_mutate(mock_population)
    assert mutated_population.shape == mock_population.shape, "Incorrect population size after mutation."

# _ParentSelection tests
def test_calculate_phi(setup_ga):
    ga = setup_ga
    phi = ga.calculate_phi()
    assert len(phi) == ga.pop_size, "Incorrect number of selection probabilities."

def test_select_from_fitness_rank(setup_ga):
    ga = setup_ga
    mock_population = np.random.randint(2, size=(ga.pop_size, ga.C))
    selected = ga.select_from_fitness_rank(mock_population)
    assert selected.shape == mock_population.shape, "Incorrect number of individuals selected."
