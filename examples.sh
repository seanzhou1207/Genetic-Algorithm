#!/bin/bash

python3 example.py --dataset Abalone --fitness_fn OLS > "Abalone_OLS.log"
python3 example.py --dataset Abalone --fitness_fn GLM > "Abalone_GLM.log"
python3 example.py --dataset Abalone --fitness_fn poisson > "Abalone_poisson.log"
python3 example.py --dataset Boston --fitness_fn OLS > "Boston_OLS.log"
python3 example.py --dataset Boston --fitness_fn GLM > "Boston_GLM.log"
python3 example.py --dataset Boston --fitness_fn poisson > "Boston_poisson.log"
