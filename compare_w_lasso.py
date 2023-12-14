from GA import *
import statsmodels.api as sm
import pandas as pd 
import numpy as np

n_samples = 1000

# Generate independent variables (20 covariates)
X = np.random.rand(n_samples, 20)

# Define a known relationship
y = 5 * X[:, 0] + 7 * X[:, 1] - 4 * X[:, 2] + np.random.normal(0, 1, n_samples)

# Create a DataFrame
# column_names = [f'x{i+1}' for i in range(20)]
# data = pd.DataFrame(X, columns=column_names)
# X = data
# data['y'] = y

sa = GA(X=X, y=y, mod=sm.OLS, max_iter=10, mutate_prob=0.01, random_seed=12)

final_chrom, final_score = sa.select([sa.random_mutate])
print(final_chrom, final_score)

#print(sm.OLS(data['y'], X[:,"x1":"x3"]).fit().aic)

# Ideally, your GA should identify x1, x2, and x3 as significant predictors

# Compare with lasso
# Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)  # alpha is the regularization parameter
lasso.fit(X, y)

# Lasso Coefficients
print("Lasso Coefficients:")
print(lasso.coef_)
