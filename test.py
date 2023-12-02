{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                  GRADE   R-squared:                       0.416\n",
      "Model:                            OLS   Adj. R-squared:                  0.353\n",
      "Method:                 Least Squares   F-statistic:                     6.646\n",
      "Date:                Mon, 27 Nov 2023   Prob (F-statistic):            0.00157\n",
      "Time:                        21:15:02   Log-Likelihood:                -12.978\n",
      "No. Observations:                  32   AIC:                             33.96\n",
      "Df Residuals:                      28   BIC:                             39.82\n",
      "Df Model:                           3                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "GPA            0.4639      0.162      2.864      0.008       0.132       0.796\n",
      "TUCE           0.0105      0.019      0.539      0.594      -0.029       0.050\n",
      "PSI            0.3786      0.139      2.720      0.011       0.093       0.664\n",
      "const         -1.4980      0.524     -2.859      0.008      -2.571      -0.425\n",
      "==============================================================================\n",
      "Omnibus:                        0.176   Durbin-Watson:                   2.346\n",
      "Prob(Omnibus):                  0.916   Jarque-Bera (JB):                0.167\n",
      "Skew:                           0.141   Prob(JB):                        0.920\n",
      "Kurtosis:                       2.786   Cond. No.                         176.\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n"
     ]
    }
   ],
   "source": [
    "# Load modules and data\n",
    "import numpy as np\n",
    "\n",
    "import statsmodels.api as sm\n",
    "\n",
    "spector_data = sm.datasets.spector.load()\n",
    "\n",
    "spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)\n",
    "\n",
    "# Fit and summarize OLS model\n",
    "mod = sm.OLS(spector_data.endog, spector_data.exog)\n",
    "\n",
    "res = mod.fit()\n",
    "\n",
    "print(res.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.66, 20.  ,  0.  ,  1.  ],\n",
       "       [ 2.89, 22.  ,  0.  ,  1.  ],\n",
       "       [ 3.28, 24.  ,  0.  ,  1.  ],\n",
       "       [ 2.92, 12.  ,  0.  ,  1.  ],\n",
       "       [ 4.  , 21.  ,  0.  ,  1.  ],\n",
       "       [ 2.86, 17.  ,  0.  ,  1.  ],\n",
       "       [ 2.76, 17.  ,  0.  ,  1.  ],\n",
       "       [ 2.87, 21.  ,  0.  ,  1.  ],\n",
       "       [ 3.03, 25.  ,  0.  ,  1.  ],\n",
       "       [ 3.92, 29.  ,  0.  ,  1.  ],\n",
       "       [ 2.63, 20.  ,  0.  ,  1.  ],\n",
       "       [ 3.32, 23.  ,  0.  ,  1.  ],\n",
       "       [ 3.57, 23.  ,  0.  ,  1.  ],\n",
       "       [ 3.26, 25.  ,  0.  ,  1.  ],\n",
       "       [ 3.53, 26.  ,  0.  ,  1.  ],\n",
       "       [ 2.74, 19.  ,  0.  ,  1.  ],\n",
       "       [ 2.75, 25.  ,  0.  ,  1.  ],\n",
       "       [ 2.83, 19.  ,  0.  ,  1.  ],\n",
       "       [ 3.12, 23.  ,  1.  ,  1.  ],\n",
       "       [ 3.16, 25.  ,  1.  ,  1.  ],\n",
       "       [ 2.06, 22.  ,  1.  ,  1.  ],\n",
       "       [ 3.62, 28.  ,  1.  ,  1.  ],\n",
       "       [ 2.89, 14.  ,  1.  ,  1.  ],\n",
       "       [ 3.51, 26.  ,  1.  ,  1.  ],\n",
       "       [ 3.54, 24.  ,  1.  ,  1.  ],\n",
       "       [ 2.83, 27.  ,  1.  ,  1.  ],\n",
       "       [ 3.39, 17.  ,  1.  ,  1.  ],\n",
       "       [ 2.67, 24.  ,  1.  ,  1.  ],\n",
       "       [ 3.65, 21.  ,  1.  ,  1.  ],\n",
       "       [ 4.  , 23.  ,  1.  ,  1.  ],\n",
       "       [ 3.1 , 21.  ,  1.  ,  1.  ],\n",
       "       [ 2.39, 19.  ,  1.  ,  1.  ]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mod.exog"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.12 ('base')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "162b6b064e165ad366d1bd4cdc631c4691b039cb551eb2d47403b16e997dbe09"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
