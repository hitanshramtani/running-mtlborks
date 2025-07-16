import tensorflow as tf
import numpy as np

def load_dataset(dataset_name='mnist'):
    """
    Loads and preprocesses a specified dataset using TensorFlow/Keras.

    Args:
        dataset_name (str): The name of the dataset to load ('mnist' or 'fashion_mnist').

    Returns:
        tuple: A tuple containing (R_train, R_valid), where each is another
               tuple of (images, labels).
    """
    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name.lower() == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        # For other datasets like Rectangles-I, you would have to write a custom
        # loader to read them from the files downloaded from the umontreal.ca URL.
        raise ValueError(f"Dataset '{dataset_name}' is not supported by this simple loader.")

    # Preprocessing steps mentioned in the paper
    # 1. Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # 2. Add a channel dimension, as CNNs expect it (e.g., 28x28 -> 28x28x1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    R_train = (x_train, y_train)
    R_valid = (x_test, y_test) # The paper uses the test set as the validation set

    print(f"Dataset '{dataset_name}' loaded.")
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_test.shape}")
    
    return R_train, R_valid