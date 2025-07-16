import random
from Fitness_Evaluation.evaluate_fitness import evaluate_fitness

def adaptive_peer_learning(
    D, n, poff, X_teacher, F_X_teacher,
    R_train, R_valid, S_batch, ε_train, R_L, C_num,
    NConv_max
):
    """
    Implements Algorithm 8: Adaptive Peer Learning of MTLBORKS-CNN

    Args:
        D (int): Dimension of learners
        n (int): Index of current learner being updated
        poff (list): List of (X_off, fitness) tuples for all learners
        X_teacher: Current teacher solution
        F_X_teacher: Fitness of teacher
        Other inputs: Training data and hyperparameters
        NConv_max: Number of convolutional layers (for rounding logic)

    Returns:
        Updated learner, its fitness, X_teacher (updated if necessary), F_X_teacher
    """

    # Step 1: Compute Rn and p_n^PL using Equations (12) and (13)
    N = len(poff)
    Rn = N - n
    p_n_PL = Rn / N

    Xn_off = poff[n][0].copy()

    for d in range(D):
        r6 = random.uniform(0, 1)
        chi_n = random.uniform(0.5, 1)
        if r6 < p_n_PL:
            # Select p, s, u ≠ n and all distinct
            available_indices = list(set(range(N)) - {n})
            if len(available_indices) < 3:
                raise ValueError("Not enough distinct learners to sample p, s, u")

            p, s, u = random.sample(available_indices, 3)
            Xp = poff[p][0]
            Xs = poff[s][0]
            Xu = poff[u][0]
            # Step 4: Update Xn_off using Equation (14)
            Xn_off[d] = Xp[d] + chi_n * (Xs[d] - Xu[d])
        # Step 5: Rounding if not a pooling prob
        is_pool_prob = False
        for l in range(1, NConv_max + 1):
            if d == 2 * NConv_max + 3 * l - 2:
                is_pool_prob = True
                break

        if not is_pool_prob:
            Xn_off[d] = round(Xn_off[d])

        

    # Step 8: Evaluate fitness
    F_Xn_off = evaluate_fitness(
        Xn_off, R_train, R_valid, S_batch, ε_train, R_L, C_num
    )

    # Step 9: Update teacher if improved
    if F_Xn_off < F_X_teacher:
        X_teacher = Xn_off.copy()
        F_X_teacher = F_Xn_off

    return Xn_off, F_Xn_off, X_teacher, F_X_teacher
