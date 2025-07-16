import numpy as np
import copy
import random


def dual_criterion_selection(N, P, P_off):
    """
    Implements the Dual-Criterion Selection Scheme (Algorithm 10).

    Args:
        P (list): The current population. A list of dicts, where each dict represents a learner.
                  e.g., [{'solution': np.array([...]), 'fitness': 0.5}, ...]
        P_off (list): The offspring population, with the same structure as P.
        N (int): The required population size for the next generation.

    Returns:
        list: The next generation population (P_next), a list of N learners.
    """
    # Line 01: Initialize the next generation population
    P_next = []

    # Line 02: Construct the merged population
    P_MG = P + P_off

    # Line 03: Rearrange the solutions of P_MG in ascending order by fitness
    P_MG.sort(key=lambda x: x[1])
    
    D = len(P_MG[0][0])  # Dimension of each solution


    # Lines 04-06: Calculate diversity Dis_n for every solution using Equation (16)
    # The diversity is the Euclidean distance to the single best solution.
    best_solution_vector = P_MG[0][0]
    diversities = [np.linalg.norm(learner[0] - best_solution_vector) for learner in P_MG]


    # Line 07: Randomly generate an integer K
    K = np.random.randint(1, N)

    # Lines 08-11: Select the first K learners (elites) based on fitness
    for k in range(K):
        P_next.append(P_MG[k])

    # --- Calculate Weighted Fitness for the remaining population ---
    # Find min/max values for normalization from the entire merged population
    fitness_values = [learner[1] for learner in P_MG]
    F_min, F_max = min(fitness_values), max(fitness_values)
    Dis_min, Dis_max = min(diversities), max(diversities)

    WF_scores = {}
    # Lines 12-20: Calculate the Weighted Fitness (WF) for each non-elite solution
    # We only need to calculate this for potential tournament candidates (indices from K to 2N-1)
    for n in range(K, 2 * N):
        # Line 13-18: Generate and clamp alpha
        alpha = random.gauss(0.9, 0.05)
        alpha = min(max(alpha, 0.8), 1.0)
        
        # Normalize fitness (handle division by zero)
        norm_fitness = 0.0
        if (F_max - F_min) > 0:
            norm_fitness = (P_MG[n][1] - F_min) / (F_max - F_min)

        # Normalize diversity (handle division by zero)
        # Equation (17) inverts the diversity score to reward more diverse solutions
        norm_diversity = 0.0
        if (Dis_max - Dis_min) > 0:
            norm_diversity = (Dis_max - diversities[n]) / (Dis_max - Dis_min)

        # Line 19: Calculate WF using the correct formula from Equation (17)
        WF_scores[n] = alpha * norm_fitness + (1 - alpha) * norm_diversity

    # Lines 21-25: Select the remaining (N - K) learners using a binary tournament
    candidate_indices = list(WF_scores.keys())
    
    while len(P_next) < N:
        # Line 22: Randomly select two distinct solution indices
        # Ensure there are candidates left to choose from
        if len(candidate_indices) < 2:
            # If not enough unique candidates, fill with the best available ones
            remaining_needed = N - len(P_next)
            sorted_candidates = sorted(WF_scores.items(), key=lambda item: item[1])
            for i in range(remaining_needed):
                winner_index = sorted_candidates[i][0]
                P_next.append(P_MG[winner_index])
            break

        a_idx, b_idx = random.sample(candidate_indices, 2)

        # Line 23: Determine winner (the one with the smaller weighted fitness)
        winner_idx = a_idx if WF_scores[a_idx] <= WF_scores[b_idx] else b_idx
        
        candidate = P_MG[winner_idx]

        # Line 24: Add the winner to the next generation, ensuring no duplicates
        if candidate not in P_next:
             P_next.append(candidate)
        # If a duplicate is selected, the loop will run again to find another winner.
        # To avoid infinite loops if all remaining candidates are already in P_next,
        # we can remove the winner from the candidate pool.
        candidate_indices.remove(winner_idx)


    # Output: The final population for the next generation
    return P_next[:N] # Ensure exact size of N
