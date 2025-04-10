import os
import numpy as np
import tensorflow as tf
import json
from os import listdir
from os.path import isfile, join

# Suppress TensorFlow logging (only errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_predictions(folder_path, model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    file_lists = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.txt')]
    raw_dataset = []

    for f in file_lists:
        try:
            curr_example = np.loadtxt(join(folder_path, f))
            raw_dataset.append(curr_example)
        except Exception as e:
            print(f"Error loading file {f}: {e}")

    if raw_dataset:
        raw_dataset = np.array(raw_dataset)
        combined_data = np.concatenate(raw_dataset)

        def make_dimensions_compatible(arr):
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # Add a new axis if it's a single sample
            if arr.shape[1] < 1024:
                padding = np.zeros((arr.shape[0], 1024 - arr.shape[1], 1))
                arr = np.concatenate((arr[:, :, np.newaxis], padding), axis=1)
            elif arr.shape[1] > 1024:
                arr = arr[:, :1024, np.newaxis]
            else:
                arr = arr[:, :, np.newaxis]

            return arr

        X_combined = make_dimensions_compatible(combined_data) / 1000  # Normalize the data
        predictions = loaded_model.predict(X_combined)
        predicted_classes = (predictions[:, 1] > 0.5).astype(int)  # Assuming sigmoid output

        # Check if any prediction indicates a seizure
        if np.any(predicted_classes == 1):
            return "You are diagnosed with epileptic seizure."
        else:
            return "You are perfectly alright."
    else:
        return "No valid files found in the uploads directory."

if __name__ == "__main__":
    folder_path = 'uploads'  # Change to your uploads folder path
    model_path = 'model_directory/cnn_model'  # Adjust path as necessary
    diagnosis = make_predictions(folder_path, model_path)
    print(diagnosis)  # Output the diagnosis