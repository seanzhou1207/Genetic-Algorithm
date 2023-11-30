import numpy as np

rng = np.random.default_rng(seed=1)


def mutate(G, mu):
    G_new = G.copy()
    mutation_locs = rng.binomial(1, mu, size = G.shape)
    mask = mutation_locs == 1
    G_new[mask] = 1 - G_new[mask] # flip bits using the mask
    return G_new


# questions: how to pass the RNG?
# This works on full Generations!
# needs to be converted to class
