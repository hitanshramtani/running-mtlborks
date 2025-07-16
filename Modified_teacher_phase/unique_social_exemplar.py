import random

def compute_unique_social_exemplars(N=None, D=None, population=None):
    """
    Implements Algorithm 5: Computation of Unique Social Exemplars.
    
    Args:
        N (int): Number of learners
        D (int): Dimension of each learner vector
        population (list): List of tuples [(Xn, fitness), ...], sorted worst to best
    
    Returns:
        List of social exemplar vectors: X_SE[n] for n = 0 to N-1
    """
    if N is None or D is None:
        N = len(population)
        D = len(population[0][0])

    # Step 1–7: Compute social exemplars for learners 0 to N-2
    X_SE_list = []

    for n in range(N):
        if n == N - 1:
            # Best learner — no better peers exist
            X_SE_list.append(None)
            continue

        # Step 2: Initialize X_SE_n as an empty vector
        X_SE_n = []

        # Step 4: Randomly select a better learner from index n+1 to N-1
        k_rand = random.randint(n + 1, N - 1)
        X_k = population[k_rand][0]  # Get learner vector only

        # Step 5: Copy each dimension from X_k
        for d in range(D):
            X_SE_n.append(X_k[d])

        X_SE_list.append(X_SE_n)

    return X_SE_list
