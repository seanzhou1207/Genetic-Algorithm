from GA import *
import statsmodels.api as sm
import pandas as pd 
import numpy as np

# spector_data = sm.datasets.spector.load()

# X = spector_data.exog
# y = spector_data.endog

# print(X)

# temp = GA(X=X, y=y, mod=sm.OLS, max_iter=10, random_seed=1)

# #operators = [GA.random_mutate]

# final_pop, fit = temp.select()
# print(final_pop, fit)

# Baseball Example
data = pd.read_csv("GA/assets/baseball.dat", delimiter = ' ')

X = data.drop("salary", axis = 1)
y = data["salary"]

operator = [GA.random_allel_selection_population, GA.random_mutate]
baseball = GA(X=X, y=y, mod=sm.OLS, max_iter=50, mutate_prob=0.01)

baseball.select()

# Synthetic Dataset
# Number of samples
n_samples = 100

# Generate independent variables
X = np.random.rand(n_samples, 5)  # 5 independent variables

# Define a known relationship
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.5, n_samples)

# Create a DataFrame
data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
data['y'] = y

X = data.drop("y", axis = 1)
y = data["y"]
sa = GA(X=X, y=y, mod=sm.OLS, max_iter=50, mutate_prob=0.01)

sa.select()

# Ideally, GA should identify x1 and x2 as significant predictors
