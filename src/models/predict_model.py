import os
import pickle
import json
import keras
import dvc.api
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

PARAMS = dvc.api.params_show()

if __name__ == "__main__":

    model = keras.models.load_model(PARAMS["trained_folder"] + "model.keras")

    with open(PARAMS["tokenized_folder"] + "x_test.pickle", "rb") as f:
        x_test = pickle.load(f)

    with open(PARAMS["tokenized_folder"]+ "y_test.pickle", "rb") as f:
        y_test = pickle.load(f)


    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    matrix = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:')
    print(matrix)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred_binary)
    print('Accuracy Score:')
    print(accuracy)

    if not os.path.exists(PARAMS["predicted_folder"]):
        os.makedirs(PARAMS["predicted_folder"])

    # Save the metrics in a JSON file
    metrics_file = os.path.join(PARAMS["predicted_folder"], "metrics.json")
    with open(metrics_file, "w") as f:
        metrics = {
            "accuracy_score": accuracy
        }
        json.dump(metrics, f)
    print(f"Training metrics saved in {metrics_file}")
