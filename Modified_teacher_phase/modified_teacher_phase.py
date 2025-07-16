import random
import numpy as np
from Modified_teacher_phase.unique_mean_pos import compute_unique_mean_positions
from Modified_teacher_phase.unique_social_exemplar import compute_unique_social_exemplars
from Fitness_Evaluation.evaluate_fitness import evaluate_fitness

def modified_teacher_phase(
    N, D, population, X_teacher, F_X_teacher,
    R_train, R_valid, S_batch, ε_train, R_L, C_num,
    NConv_max
):
    """
    Implements Algorithm 6: Modified Teacher Phase of MTLBORKS-CNN

    Args:
        N, D: Population size and vector dimension
        population: List of (Xn, fitness) tuples, sorted by fitness (worst to best)
        X_teacher: Best learner vector (X_T)
        F_X_teacher: Fitness of teacher
        Training data, validation data, etc...

    Returns:
        Updated offspring population P_off and new teacher if found
    """
    # Step 1: Unique mean vectors
    mean_vectors = compute_unique_mean_positions(N, D, population, NConv_max)

    # Step 2: Unique social exemplars
    social_exemplars = compute_unique_social_exemplars(N, D, population)

    # Step 3: Initialize offspring population
    P_off = []

    for n in range(N):
        Xn, _ = population[n]

        # Step 5: Skip if n == N (already the teacher)
        if n == N - 1:
            X_off_n = Xn.copy()
            F_X_off_n = evaluate_fitness(X_off_n, R_train, R_valid, S_batch, ε_train, R_L, C_num)
            del(X_off_n)  # Clean up to save memory
        else:
            r3 = random.uniform(0, 1)
            r4 = random.uniform(0, 1)
            F_T = random.choice([1, 2])  # Not explicitly used in Eq. 10

            # Step 7: Calculate Xn^off using Equation (10)
            Xn_bar = mean_vectors[n]
            Xn_SE = social_exemplars[n]
            X_off_n = []

            for d in range(D):
                teacher_d = X_teacher[d]
                mean_d = abs(Xn_bar[d])
                exemplar_d = Xn_SE[d]
                current_d = Xn[d]

                # Equation (10)
                new_d = (
                    current_d
                    + r3 * (teacher_d - F_T*mean_d)
                    + r4 * (exemplar_d - current_d)
                )

                # Step 9: rounding if not a pooling probability
                is_pool_prob = False
                for l in range(1, NConv_max + 1):
                    pool_prob_index = 2 * NConv_max + 3 * l - 2
                    if d == pool_prob_index:
                        is_pool_prob = True
                        break

                if not is_pool_prob:
                    new_d = round(new_d)
                else:
                    new_d = float(np.clip(new_d, 0.0, 1.0))

                X_off_n.append(new_d)

            # Step 13: Evaluate fitness
            F_X_off_n = evaluate_fitness(X_off_n, R_train, R_valid, S_batch, ε_train, R_L, C_num)

        # Step 15: Update teacher if needed
        if F_X_off_n < F_X_teacher:
            X_teacher = X_off_n.copy()
            F_X_teacher = F_X_off_n

        # Step 18: Add to new offspring population
        P_off.append((X_off_n, F_X_off_n))

    # Step 20: return new population and updated teacher
    return P_off, X_teacher, F_X_teacher
