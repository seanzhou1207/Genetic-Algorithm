from gavs import *
import statsmodels.api as sm

spector_data = sm.datasets.spector.load()

X = spector_data.exog
y = spector_data.endog

temp = GA(pop_size=3.4, X=X, y=y, mod=sm.OLS, max_iter=10.5)
