import random
import math
import numpy as np
from Fitness_Evaluation.evaluate_fitness import evaluate_fitness

def initialize_population(
    N,  # population size
    NConv_min, NConv_max,
    NFil_min, NFil_max,
    SKer_min, SKer_max,
    SPool_min, SPool_max,
    SStr_min, SStr_max,
    NFC_min, NFC_max,
    NNeu_min, NNeu_max
):
    # Step 1: Calculate dimension size D
    D = 5 * NConv_max + NFC_max + 2

    # Step 2: Initialize teacher
    X_Teacher = None
    F_X_Teacher = float('inf')

    # Population list
    population = []

    for n in range(N):
        Xn = []

        # ----- Conv Layer Count -----
        NConv = random.randint(NConv_min, NConv_max)
        Xn.append(NConv)

        # ----- Conv Layer Filters & Kernel -----
        for l in range(NConv_max):
            NFil = random.randint(NFil_min, NFil_max)
            SKer = random.choice([3, 5, 7])  # typical values
            Xn.extend([NFil, SKer])

        # ----- Pooling Layers -----
        for l in range(NConv_max):
            PPool = round(random.uniform(0, 1), 2)  # probability for pooling type
            SPool = random.randint(SPool_min, SPool_max)
            SStr = random.randint(SStr_min, SStr_max)
            Xn.extend([PPool, SPool, SStr])

        # ----- Fully-Connected Layer Count -----
        NFC = random.randint(NFC_min, NFC_max)
        Xn.append(NFC)

        # ----- FC Neurons -----
        for q in range(NFC_max):
            NNeu = random.randint(NNeu_min, NNeu_max)
            Xn.append(NNeu)

        # Step 9: Evaluate fitness
        fitness = evaluate_fitness(Xn)  # you must define this externally

        # Step 10–12: Update teacher
        if fitness < F_X_Teacher:
            X_Teacher = Xn.copy()
            F_X_Teacher = fitness

        population.append((Xn, fitness))

    return population, X_Teacher, F_X_Teacher
