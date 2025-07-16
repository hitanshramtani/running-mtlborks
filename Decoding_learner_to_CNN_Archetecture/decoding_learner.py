import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models

def decode_learner_to_cnn(Xn, NConv_max, input_shape=(28, 28, 1), num_classes=10):
    """
    This function decodes a learner's archetecture into a CNN Model.
    
    Args:
        Xn: List of Parameters representing the architecture.
            => Xn = [NConv, NFil1, SKer1, PPool1, SPool1, SStr1, ..., NFC, NNeu1, NNeu2, ..., NNeuNFC]
        NConv_max: Maximum number of convolutional layers.
        NFC_max: Maximum number of fully connected layers.
        input_shape: Shape of the input data (default is (28, 28, 1) for grayscale images).
        num_classes: Number of output classes (default is 10 for digit classification).
    Returns:
        A Keras Sequential model representing the CNN architecture.
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    l = 1  # conv layer index
    q = 1  # fc layer index
    
    NConv = int(round(Xn[0]))
    NFC = int(round(Xn[5 * NConv_max + 2]))  # FC count is at this index
    
    # ---- Decode Convolutional + Pooling Layers ----
    while l <= NConv:
        # Indexing for conv layer
        NFil = int(round(Xn[2 * l]))
        SKer = int(round(Xn[2 * l + 1]))
        
        model.add(layers.Conv2D(NFil, (SKer, SKer), activation='relu', padding='same'))
        
        # Indexing for pooling
        d1 = 2 * NConv_max + 3 * l - 1
        d2 = 2 * NConv_max + 3 * l
        d3 = 2 * NConv_max + 3 * l + 1
        
        PPool = Xn[d1]
        SPool = int(round(Xn[d2]))
        SStr = int(round(Xn[d3]))

        if 0 <= PPool < 1/3:
            pass  # No pooling
        elif 1/3 <= PPool < 2/3:
            model.add(layers.MaxPooling2D(pool_size=(SPool, SPool), strides=(SStr, SStr)))
        else:  # PPool ≥ 2/3
            model.add(layers.AveragePooling2D(pool_size=(SPool, SPool), strides=(SStr, SStr)))
        
        l += 1

    model.add(layers.Flatten())
    
    # ---- Decode Fully Connected Layers ----
    while q <= NFC:
        d = (5 * NConv_max + 2) + q
        NNeu = int(round(Xn[d]))
        model.add(layers.Dense(NNeu, activation='relu'))
        q += 1

    # ---- Output Layer ----
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
