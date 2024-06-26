# pylint: disable=all
# flake8: noqa

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import random
import re
import pytest
import pandas as pd
import logging
from lib_ml_group3.load_model import load_model
from sklearn.model_selection import train_test_split

INPUT_DIR = "data/external/"

SENSITIVE_PATTERNS = re.compile(r"(@|token|session|user|userid|"
                                r"password|auth|files|pro)", re.IGNORECASE)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger to capture all levels of log messages

model = load_model(
    "data/external/model.keras",
    "data/external/encoder.pickle",
    "data/external/tokenizer.pickle",
)


def read_and_sample_data(file_path, sample_size=100, seed=None):
    """Read data from a file, sample lines randomly,
    and return the sampled lines."""
    random.seed(seed)
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        random.shuffle(lines)
        return lines[:sample_size]


def remove_sensitive_info(text):
    """Remove sensitive information from the text based on
    the SENSITIVE_PATTERNS."""
    return SENSITIVE_PATTERNS.sub("[REDACTED]", text)


@pytest.fixture
def data():
    train = read_and_sample_data(INPUT_DIR + "train.txt", seed=42)
    val = read_and_sample_data(INPUT_DIR + "val.txt", seed=42)
    test = read_and_sample_data(INPUT_DIR + "test.txt", seed=42)

    # Process train data
    raw_x_train = [remove_sensitive_info(line.split("\t")[1]) for line in
                   train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process validation data
    raw_x_val = [remove_sensitive_info(line.split("\t")[1]) for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    # Process test data
    raw_x_test = [remove_sensitive_info(line.split("\t")[1]) for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    return {
        "texts": raw_x_train + raw_x_val + raw_x_test,
        "labels": raw_y_train + raw_y_val + raw_y_test
    }


def evaluate_data_slices(data, model):
    data = pd.DataFrame({'texts': data["texts"], 'labels': data["labels"]})
    short_threshold = 30
    medium_threshold = 50

    # Create slices
    data['text_length'] = data['texts'].apply(len)
    short_texts = data[data['text_length'] <= short_threshold]
    medium_texts = data[(data['text_length'] > short_threshold)
                        & (data['text_length'] <= medium_threshold)]
    long_texts = data[data['text_length'] > medium_threshold]

    results = {}
    for slice_name, texts_slice in \
            zip(["short", "medium", "long"],
                [short_texts, medium_texts, long_texts],
                ):
        y = texts_slice['labels']
        predictions = model.predict(texts_slice['texts'])
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        results[slice_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        logger.info(f"Slice: {slice_name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print(
            f"Slice: {slice_name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return results


def test_data_slices(data):
    logger.info("Test data slices")
    results = evaluate_data_slices(data, model)
    short_metrics = results["short"]
    medium_metrics = results["medium"]
    long_metrics = results["long"]

    assert abs(short_metrics['accuracy'] - medium_metrics['accuracy']) < 0.25
    assert abs(short_metrics['accuracy'] - long_metrics['accuracy']) < 0.25
    assert abs(medium_metrics['accuracy'] - long_metrics['accuracy']) < 0.25


def perturb_text(text):
    """Apply a small perturbation to the text data."""
    return text + "/"


@pytest.fixture(params=[42, 43, 44])
def mutated_data(request):
    seed = request.param
    train = read_and_sample_data(INPUT_DIR + "train.txt", seed=seed)
    val = read_and_sample_data(INPUT_DIR + "val.txt", seed=seed)
    test = read_and_sample_data(INPUT_DIR + "test.txt", seed=seed)

    # Process train data
    raw_x_train = \
        [remove_sensitive_info(line.split("\t")[1]) for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process validation data
    raw_x_val = [remove_sensitive_info(line.split("\t")[1]) for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    # Process test data
    raw_x_test = [remove_sensitive_info(line.split("\t")[1]) for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Apply perturbations
    perturbed_x_train = [perturb_text(text) for text in raw_x_train]
    perturbed_x_val = [perturb_text(text) for text in raw_x_val]
    perturbed_x_test = [perturb_text(text) for text in raw_x_test]

    return {
        "texts": perturbed_x_train + perturbed_x_val + perturbed_x_test,
        "labels": raw_y_train + raw_y_val + raw_y_test
    }


def test_mutamorphic(mutated_data):
    logger.info("Test mutamorphic")
    results = evaluate_data_slices(mutated_data, model)
    short_metrics = results["short"]
    medium_metrics = results["medium"]
    long_metrics = results["long"]

    assert abs(short_metrics['accuracy'] - medium_metrics['accuracy']) < 0.5
    assert abs(short_metrics['accuracy'] - long_metrics['accuracy']) < 0.5
    assert abs(medium_metrics['accuracy'] - long_metrics['accuracy']) < 0.5


def retrain_model(seed, data):
    # Load and preprocess the data
    X = data['texts']
    y = data['labels']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model evaluation: Accuracy: {accuracy}")
    print(f"Model evaluation: Accuracy: {accuracy}")
    return {
        'accuracy': accuracy
    }


@pytest.fixture(params=[42, 43, 44])
def retrained_model(data, request):
    seed = request.param
    model, X_test, y_test = retrain_model(seed, data)
    return model, X_test, y_test


def test_non_determinism_robustness(data, retrained_model):
    logger.info("Test non-determinism")
    metrics = evaluate_model(model, data['texts'], data['labels'])
    retrained_metrics = evaluate_model(retrained_model[0], retrained_model[1],
                                       retrained_model[2])
    logger.info(f"Original model metrics: {metrics}")
    logger.info(f"Retrained model metrics: {retrained_metrics}")
    print(f"Original model metrics: {metrics}")
    print(f"Retrained model metrics: {retrained_metrics}")
    assert abs(retrained_metrics['accuracy'] - metrics['accuracy']) < 0.25
