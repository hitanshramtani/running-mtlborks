def compute_unique_mean_positions(N, D, population, NConv_max):
    """
    Inputs:
        population: list of tuples (Xn, fitness)
        NConv_max: maximum number of convolutional layers
    Returns:
        mean_vectors: list of mean vectors (excluding the best learner)
    """
    if N is None or D is None:
        N = len(population)
        D = len(population[0][0])  # dimensionality of each learner
    mean_vectors = []

    # Step 1: Sort learners by fitness (descending: worst to best)
    sorted_population = sorted(population, key=lambda x: x[1], reverse=True)

    # Step 2–10: Compute mean vectors for all except the best (n = 1 to N-1)
    for n in range(N - 1):  # n = 0 to N-2
        mean_vector = []

        # Step 3–4: For each dimension
        for d in range(D):
            # Mean of dimension d across learners X_n to X_N (i.e. m = n to N-1)
            dim_sum = sum(sorted_population[m][0][d] for m in range(n, N))
            Xnd = dim_sum / (N - n)

            # Step 6–7: Round unless it's a pooling probability dimension
            is_pool_prob = False
            for l in range(1, NConv_max + 1):
                pool_prob_index = 2 * NConv_max + 3 * l - 2  # zero-indexed
                if d == pool_prob_index:
                    is_pool_prob = True
                    break

            if not is_pool_prob:
                Xnd = round(Xnd)

            mean_vector.append(Xnd)

        mean_vectors.append(mean_vector)

    return mean_vectors