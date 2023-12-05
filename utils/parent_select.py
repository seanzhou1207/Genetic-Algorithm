import numpy as np

class ParentSelection:

    def __init__():
        pass

    def calculate_phi(self, current_population):
        """
        Calculate selection probability as 2r_i/P(P+1)
        """
        P = self.pop_size    # Best ranked is P, then P-1, P-2,...,1
        rs = np.arange(P, 0, -1)
        phi = 2 * rs/(P*(P+1))

        return phi

    def select_from_fitness_rank(self, current_population):
        """
        Choose parents based on fitness ranks
        """
        selection_prob = self.calculate_phi(current_population)
        print(selection_prob)

        if self.pop_size % 2 == 0:    # If pop_size is even
            num_of_parents = self.pop_size / 2
        
        else:    # If pop_size is odd CHECK THIS
            num_of_parents = self.pop_size // 2 + 1
        
        row_idx = np.arange(len(current_population))
        chosen_rows = np.random.choice(row_idx, size=self.pop_size, p=selection_prob,replace=True)
        chosen_individuals = current_population[chosen_rows]

        return chosen_individuals


