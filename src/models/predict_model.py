import os
import pickle
import json
import keras
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


OUPUT_DIR = "predicted/"

if __name__ == "__main__":

    model = keras.models.load_model("trained/model.keras")

    with open("tokenized/x_test.pickle", "rb") as f:
        x_test = pickle.load(f)

    with open("tokenized/y_test.pickle", "rb") as f:
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

    if not os.path.exists(OUPUT_DIR):
        os.makedirs(OUPUT_DIR)

    # Save the metrics in a JSON file
    metrics_file = os.path.join(OUPUT_DIR, "metrics.json")
    with open(metrics_file, "w") as f:
        metrics = {
            "accuracy_score": accuracy
        }
        json.dump(metrics, f)
    print(f"Training metrics saved in {metrics_file}")
