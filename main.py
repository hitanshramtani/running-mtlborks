from proposed_solution.optimizer import run_mtlborks_cnn
from Data_loader.data_loader import load_dataset

def main():
    """
    Sets configurations and starts the optimization process.
    """
    # Define all hyperparameters and settings in a configuration dictionary
    # These values are taken from the paper's tables and text
    config = {
        # --- Algorithm Parameters ---
        'N': 20,          # Population size
        'D': 19,          # Dimensional size
        'T_max': 10,      # Maximum iterations
        'dataset_name': 'MNIST', # Or 'MNIST-Fashion', 'Rectangles', etc.
        
        # --- CNN Hyperparameter Search Ranges ---
        'N_conv_min': 1, 'N_conv_max': 3,
        'N_fil_min': 3, 'N_fil_max': 256,
        'S_ker_min': 3, 'S_ker_max': 9, # Assuming square kernels for simplicity
        'S_pool_min': 1, 'S_pool_max': 3,
        'S_str_min': 1, 'S_str_max': 2,
        'N_fc_min': 1, 'N_fc_max': 2,
        'N_neu_min': 1, 'N_neu_max': 300,

        # --- Training Parameters ---
        'S_batch': 128,  # A common batch size
        'R_L': 0.001, # A common learning rate for Adam
        'C_num': 10, # Number of output classes for the dataset
        'ε_train': 1,       # Epochs for fitness evaluation during search
        'ε_FT': 100,  # Epochs for final model training
    }
    # --- DATA LOADING ---
    # Load the actual data and add it to the config dictionary
    # This replaces `R_train, R_valid = None, None`
    R_train, R_valid = load_dataset(config['dataset_name'])
    config['R_train'] = R_train
    config['R_valid'] = R_valid


    # Run the main optimization algorithm
    optimal_cnn_details = run_mtlborks_cnn(config)

    # Print the final results
    print("\n-------------------------------------------")
    print("Optimization Complete!")
    print("Final Optimal CNN Details:")
    print(f"  - Architecture: {optimal_cnn_details.get('architecture_str')}")
    print(f"  - Final Classification Error: {optimal_cnn_details.get('error'):.4f}")
    print(f"  - Number of Trainable Parameters: {optimal_cnn_details.get('param_count')}")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()