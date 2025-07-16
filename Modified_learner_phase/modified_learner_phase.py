import random
from Modified_learner_phase.adaptive_learning import adaptive_peer_learning
from Modified_learner_phase.self_learner import self_learning

def modified_learner_phase(
    D, poff, X_teacher, F_X_teacher,
    R_train, R_valid, S_batch, ε_train, R_L, C_num,
    NConv_max, X_max, X_min
):
    """
    Implements Algorithm 9: Modified Learner Phase of MTLBORKS-CNN

    Args:
        D (int): Dimensionality of learner vectors
        poff (list): List of (X_off, fitness) for all learners
        X_teacher, F_X_teacher: Current teacher solution and fitness
        Other training/evolution parameters
        NConv_max: For rounding logic
        X_max, X_min: Needed for boundary operations in Algo 7

    Returns:
        Updated poff (list), X_teacher, F_X_teacher
    """

    N = len(poff)

    # Step 01: Sort poff by fitness DESCENDING (worst → best)
    poff_sorted = sorted(poff, key=lambda x: x[1], reverse=True)

    updated_poff = []

    # Step 02: for n = 1 to N do
    for n in range(N):
        Xn_off, _ = poff_sorted[n]
        Rn = N - n  # Equation (12)
        p_n_PL = Rn / N  # Equation (13)

        r6 = random.uniform(0, 1)

        if r6 < p_n_PL:
            # Step 04: Self-learning
            Xn_off_new, F_Xn_off, X_teacher, F_X_teacher = self_learning(
                D, Xn_off, X_teacher, F_X_teacher,
                X_max, X_min,
                R_train, R_valid, S_batch, ε_train, R_L, C_num,
                NConv_max
            )
        else:
            # Step 06: Peer-learning
            Xn_off_new, F_Xn_off, X_teacher, F_X_teacher = adaptive_peer_learning(
                D, n, poff_sorted, X_teacher, F_X_teacher,
                R_train, R_valid, S_batch, ε_train, R_L, C_num,
                NConv_max
            )

        updated_poff.append((Xn_off_new, F_Xn_off))

    return updated_poff, X_teacher, F_X_teacher
