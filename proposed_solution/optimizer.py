import random
import numpy as np
from Population_Initialization.population_initializer import initialize_population
from Modified_teacher_phase.modified_teacher_phase import modified_teacher_phase
from Modified_learner_phase.modified_learner_phase import modified_learner_phase
from Dual_criterion_selection.dual_criterion import dual_criterion_selection
from Fitness_Evaluation.evaluate_fitness import evaluate_fitness

def run_mtlborks_cnn(config):
    """
    Implements the main workflow of MTLBORKS-CNN from Algorithm 11.

    Args:
        config (dict): A dictionary containing all hyperparameters and settings.

    Returns:
        dict: Information about the final, fully-trained optimal CNN.
    """
    # Line 01: Load data (This would be your actual data loading function)
    print(f"Loading data for {config['dataset_name']}...")

    # Line 02: Initialize the population P using Algorithm 2
    # The initializer should return the population and the initial best solution
    P, X_teacher, F_teacher = initialize_population(config)
    print("Population initialized.")

    # Line 03: Initialize the iteration counter
    t = 0
    
    print(f"\nStarting optimization loop for {config['T_max']} iterations...")
    # Line 04-10: Main optimization loop
    while t < config['T_max']:
        print(f"--- Iteration {t + 1}/{config['T_max']} ---")

        # Line 05: Generate P_off using modified teacher phase (Algorithm 6)
        P_off, X_teacher, F_teacher = modified_teacher_phase(P, X_teacher, F_teacher, config)

        # Line 06: Update P_off using modified learner phase (Algorithm 9)
        P_off, X_teacher, F_teacher = modified_learner_phase(P_off, X_teacher, F_teacher, config)
        
        # Line 07: Determine P_next using dual-criterion selection scheme (Algorithm 10)
        P_next = dual_criterion_selection(config['N'], P, P_off)

        # Line 08: Update the main population for the next iteration
        P = P_next
        
        # Line 09: Increment the iteration counter
        t += 1

    print("\nOptimization loop finished.")
    
    # Line 11: Fully train the final CNN architecture from X_teacher with larger ε_FT
    print("Starting full training of the best model found...")
    final_model_info = evaluate_fitness(X_teacher, config)

    # Output: The optimal CNN architecture and its related information
    return final_model_info