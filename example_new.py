import argparse
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import ndarray

from GA import GA


def load_dataset(dataset: str) -> Tuple[ndarray, ndarray]:
    if dataset == "Boston":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        )
        X = df[
            [
                "crim",
                "zn",
                "indus",
                "chas",
                "nox",
                "rm",
                "age",
                "dis",
                "rad",
                "tax",
                "ptratio",
                "b",
                "lstat",
            ]
        ]
        y = df["medv"]

        return X.values, y.values
    elif dataset == "Abalone":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/SameetAsadullah/K-Means-Clustering-on-Abalone-Dataset/main/src/abalone.data",
            names=[
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole weight",
                "Shucked weight",
                "Viscera weight",
                "Shell weight",
                "Rings",
            ],
            header=None,
        )
        df = df[df.Height != 0]
        df["Sex_m"] = np.where(df["Sex"] == "M", 1, 0)
        df["Sex_f"] = np.where(df["Sex"] == "F", 1, 0)
        X = df[
            [
                "Sex_m",
                "Sex_f",
                "Length",
                "Diameter",
                "Height",
                "Shucked weight",
                "Shell weight",
                "Viscera weight",
                "Rings",
            ]
        ]
        y = df["Whole weight"]

        return X.values, y.values

    else:
        raise ValueError(
            f"Unsupported dataset {dataset}. Please try either `Boston` or `Abalone`"
        )


def main(config):
    X, y = load_dataset(config.dataset)
    if config.fitness_fn == "poisson":
        mod = partial(sm.GLM, family=sm.families.Poisson())
    else:
        mod = getattr(sm, config.fitness_fn)

    # fmt: off
    ga = GA(
        pop_size=config.population_size, 
        X=X, 
        y=y, 
        mod=mod, 
        max_iter=config.max_iter,
        mutate_prob=config.mutate_prob,
        exploit=config.exploit
    )
    # fmt: on
    best_solution, best_score = ga.select(
        [getattr(GA, config.cross_over_fn), GA.random_mutate]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA example", add_help=True)

    # fmt: off
    parser.add_argument("--dataset", default="Abalone", type=str, help="`Abalone` or `Boston`, dataset name to run GA on (default: Abalone)")
    parser.add_argument("--population_size", default=50, type=int, help="population size used for genetic algorithm (default: 50)")
    parser.add_argument("--max_iter", default=100, type=int, help="genetic algorithm iteration steps (default: 100)")
    parser.add_argument("--mutate_prob", default=0.1, type=float, help="genetic algorithm mutation probability (default: 0.1)")
    parser.add_argument("--fitness_fn", default="OLS", type=str, help="objective function type (default: OLS)")
    parser.add_argument("--cross_over_fn", default="split_and_glue_population", type=str, help="`split_and_glue_population` or `random_allel_selection_population` (default: split_and_glue_population)")
    parser.add_argument('--exploit', default=False, action='store_true', help="set exploit=True when initializing GA object")

    opt = parser.parse_args()
    main(opt)
