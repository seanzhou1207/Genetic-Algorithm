from GA import *
import statsmodels.api as sm
import numpy as np

spector_data = sm.datasets.spector.load()

X = spector_data.exog
y = spector_data.endog

# Initialize GA class
ga_1 = GA(X=X.values, y=y.values, mod=sm.OLS, max_iter=10, random_seed=1)

# Run GA under default operators
final_pop, fit = ga_1.select()
print(final_pop, fit)
