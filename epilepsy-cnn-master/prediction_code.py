import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys

# Load the saved model
model_path = 'model_directory/cnn_model'  # Adjust path as necessary
loaded_model = tf.keras.models.load_model(model_path)

# Define the folder containing the uploaded text files
folder_path = 'uploads/'  # This should match the upload directory in server.js

# Prepare to read files
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

# Read every file from the specified folder and store in raw_dataset
raw_dataset = []

for f in file_lists:
    try:
        curr_example = np.loadtxt(os.path.join(folder_path, f))
        raw_dataset.append(curr_example)
    except Exception as e:
        print(f"Error loading file {f}: {e}", file=sys.stderr)

# Check if any valid data was loaded
if not raw_dataset:
    print("No valid data files found.", file=sys.stderr)
    sys.exit(1)

# Convert to NumPy array
raw_dataset = np.array(raw_dataset)

# Combine the data from all files into a single array
combined_data = np.concatenate(raw_dataset)

# Preprocess the combined data
def make_dimensions_compatible(arr):
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[1] < 1024:
        padding = np.zeros((arr.shape[0], 1024 - arr.shape[1], 1))
        arr = np.concatenate((arr[:, :, np.newaxis], padding), axis=1)
    elif arr.shape[1] > 1024:
        arr = arr[:, :1024, np.newaxis]
    else:
        arr = arr[:, :, np.newaxis]
    return arr

# Normalize and reshape the combined data
X_combined = make_dimensions_compatible(combined_data) / 1000

# Make predictions using the loaded model
predictions = loaded_model.predict(X_combined)

# Convert predictions to binary class labels
predicted_classes = (predictions[:, 1] > 0.5).astype(int)

# Define a mapping from class indices to labels
class_labels = {
    0: "No Seizure",
    1: "Seizure"
}

# Output the final predictions for each file
results = []
for predicted_class in predicted_classes:
    seizure_status = class_labels[predicted_class]
    results.append(seizure_status)

# Print the final result (you can customize this output)
final_result = "Seizure Detected" if any(result == "Seizure" for result in results) else "No Seizure Detected"
print(final_result)