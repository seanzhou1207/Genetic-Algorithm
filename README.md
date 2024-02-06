# gavs: Genetic Algorithm for GLM Variable Selection
Genetic algorithms (GA) are stochastic methods usually used for optimization or search problems. They utilize principles from biological evolution and natural selection, such as selection, crossover and mutation. The goal of this project is to develop a genetic algorithm for variable selection, including both linear regression and GLMs. The user can input a dataset (with covariates and corresponding response), the desired type of regression, and related operator arguments. The algortihm will perform the variable selection and return the feature combination that is the most fit according to the model's Akaike information criterion (AIC).

## Run examples of linear models
```
bash examples.sh
```

## Run test cases
```
pytest test_main.py -v
```
