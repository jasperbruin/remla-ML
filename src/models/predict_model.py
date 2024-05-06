"""
This script loads the trained model and the test data, makes predictions on
the test data, and calculates the accuracy score, classification report,
and confusion matrix. The metrics are saved in a JSON file.
"""

import os
import pickle
import json
import keras
import dvc.api
import numpy as np

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

PARAMS = dvc.api.params_show()

if __name__ == "__main__":
    # Load the trained model
    model = keras.models.load_model(os.path.join(PARAMS["trained_folder"],
                                                 "model.keras"))

    # Load test datasets
    with open(os.path.join(PARAMS["tokenized_folder"], "x_test.pickle"),
              "rb") as f:
        x_test = pickle.load(f)
    with open(os.path.join(PARAMS["tokenized_folder"], "y_test.pickle"),
              "rb") as f:
        y_test = pickle.load(f)

    # Make predictions
    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    # Generate and print classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Generate and print confusion matrix
    matrix = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:')
    print(matrix)

    # Calculate and print accuracy score
    accuracy = accuracy_score(y_test, y_pred_binary)
    print('Accuracy Score:')
    print(accuracy)

    # Ensure the output directory exists
    predicted_folder = PARAMS["predicted_folder"]
    if not os.path.exists(predicted_folder):
        os.makedirs(predicted_folder)

    # Save the metrics in a JSON file
    metrics_file = os.path.join(predicted_folder, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        metrics = {
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),  # convert numpy array to list
            "accuracy_score": accuracy
        }
        json.dump(metrics, f, indent=4)  # pretty print json

    print(f"Training metrics saved in {metrics_file}")
