import numpy as np
import random
from Fitness_Evaluation.evaluate_fitness import evaluate_fitness

def self_learning(
    D, X_off_n, X_teacher, F_X_teacher,
    X_max, X_min,
    R_train, R_valid, S_batch, ε_train, R_L, C_num,
    NConv_max
):
    """
    Implements Algorithm 7: Self-Learning Scheme of MTLBORKS-CNN

    Args:
        D (int): Dimension of each learner vector
        X_off_n (list): Current learner vector
        X_teacher (list): Current teacher vector
        F_X_teacher (float): Fitness of teacher
        X_max, X_min: Upper/lower bounds for each dimension
        R_train, R_valid: Training and validation data
        S_batch, ε_train, R_L: Learning hyperparameters
        C_num: Number of output classes
        NConv_max: Maximum number of convolutional layers

    Returns:
        Updated X_off_n, possibly updated X_teacher and its fitness
    """

    # Step 1: Select random dimension d_rand ∈ [0, D-1]
    d_rand = random.randint(0, D - 1)

    # Step 2: Generate random numbers r5 ∈ [0, 1]
    r5 = random.uniform(0, 1)
   

    # Step 2–10: Update only d_rand using Equation (11)
    for d in range(D):
        if d == d_rand:
            
            X_off_n[d] += r5 * (X_max[d] - X_min[d])

            # Step 6–7: If not a pooling probability, round it
            is_pool_prob = False
            for l in range(1, NConv_max + 1):
                pool_prob_index = 2 * NConv_max + 3 * l - 2
                if d == pool_prob_index:
                    is_pool_prob = True
                    break

            if not is_pool_prob:
                X_off_n[d] = round(X_off_n[d])

            # Boundary check
            X_off_n[d] = max(X_min[d], min(X_off_n[d], X_max[d]))

    # Step 11: Evaluate fitness of new learner
    F_X_off_n = evaluate_fitness(
        X_off_n, R_train, R_valid,
        S_batch, ε_train, R_L, C_num
    )

    # Step 12–14: Teacher update if necessary
    if F_X_off_n < F_X_teacher:
        X_teacher = X_off_n.copy()
        F_X_teacher = F_X_off_n

    return X_off_n, F_X_off_n, X_teacher, F_X_teacher