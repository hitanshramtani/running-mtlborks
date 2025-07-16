import tensorflow as tf
import numpy as np
from Decoding_learner_to_CNN_Archetecture.decoding_learner import decode_learner_to_cnn


def evaluate_fitness(
    Xn, R_train, R_valid,
    S_batch=64, ε_train=1, R_L=0.001, C_num=10, # ε_train=1 as defined in paper
    NConv_max=3, NFC_max=2, input_shape=(28, 28, 1)
):
    try:
        x_train, y_train = R_train
        x_val, y_val = R_valid

        # -----------------------------
        # Step 1: Decode and build model
        # -----------------------------
        model = decode_learner_to_cnn(
            Xn, NConv_max,
            input_shape=input_shape,
            num_classes=C_num
        )

        # Add final classification layer
        model.add(tf.keras.layers.Dense(C_num, activation='softmax'))

        # Initialize weights using He Normal, Step 3
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = tf.keras.initializers.HeNormal()

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=R_L)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # -----------------------------
        # Step 2: Train for ε_train epochs, Step 4- 8, Keras handles this Internally
        # -----------------------------
        model.fit(
            x_train, y_train,
            epochs=ε_train,
            batch_size=S_batch,
            verbose=0  # Silent training
        )

        # -----------------------------
        # Step 3: Evaluate fitness on validation set, Step 10-14
        # -----------------------------
        num_batches = int(np.ceil(len(x_val) / S_batch))
        total_error = 0.0

        for j in range(num_batches):
            start = j * S_batch
            end = min((j + 1) * S_batch, len(x_val))
            x_batch = x_val[start:end]
            y_batch = y_val[start:end]

            predictions = model.predict(x_batch, verbose=0)
            pred_labels = np.argmax(predictions, axis=1)
            batch_accuracy = np.mean(pred_labels == y_batch)
            batch_error = 1 - batch_accuracy
            total_error += batch_error

        # Final fitness value = mean error across batches
        F_Xn = total_error / num_batches

        return F_Xn

    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return float('inf')  # Assign worst possible fitness on failure