# pylint: disable=all
# flake8: noqa

import os
import numpy as np
import pytest
import tensorflow as tf
import psutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, \
    Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random

# Setting a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

INPUT_DIR = "data/external/"


def read_and_sample_data(file_path, sample_size=100):
    """Read data from a file, sample lines randomly, and return the sampled lines."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        random.shuffle(lines)
        return lines[:sample_size]


@pytest.fixture
def data():
    """
    Fixture for loading, processing, and saving sampled data.

    Returns:
        dict: Contains processed data, tokenizer, and encoder.
    """
    sample_size = 100  # Set sample size for each file

    train = read_and_sample_data(INPUT_DIR + "train.txt", sample_size)
    val = read_and_sample_data(INPUT_DIR + "val.txt", sample_size)
    test = read_and_sample_data(INPUT_DIR + "test.txt", sample_size)

    # Process train data
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process validation data
    raw_x_val = [line.split("\t")[1] for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    # Process test data
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Tokenizer setup
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    # Label Encoder setup
    encoder = LabelEncoder()
    encoder.fit(raw_y_train + raw_y_val + raw_y_test)

    # Tokenize and pad sequences
    max_sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train),
                            maxlen=max_sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val),
                          maxlen=max_sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test),
                           maxlen=max_sequence_length)

    y_train = encoder.transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "max_sequence_length": max_sequence_length
    }


def create_model(voc_size, max_sequence_length):
    model = Sequential()
    model.add(Embedding(voc_size, 50, input_length=max_sequence_length))
    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


@pytest.fixture
def model_fixture(data):
    voc_size = len(data["tokenizer"].word_index) + 1
    return create_model(voc_size, data["max_sequence_length"])


@pytest.mark.skipif(not tf.config.list_physical_devices('GPU'),
                    reason="GPU is not available")
def test_gpu_available():
    """Test if GPU is available."""
    assert tf.config.list_physical_devices('GPU'), "GPU is not available"


def test_memory_usage(data, model_fixture):
    """Test memory usage before and after model training."""
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 ** 2

    model = model_fixture
    dummy_input = data["x_train"][:1000]
    dummy_output = data["y_train"][:1000]
    model.fit(dummy_input, dummy_output, epochs=1, batch_size=10)

    memory_after = process.memory_info().rss / 1024 ** 2  # Memory usage in MB
    memory_diff = memory_after - memory_before

    assert memory_diff < 100, f"Memory usage increased significantly: {memory_diff} MB"


def test_model_saving_loading(data, model_fixture):
    """Test saving and loading of the model."""
    model = model_fixture
    model.fit(data["x_train"], data["y_train"], epochs=1)
    model.save('model.h5')
    loaded_model = tf.keras.models.load_model('model.h5')

    assert model.get_config() == loaded_model.get_config(), \
        "Loaded model is not the same as the saved model"

    os.remove('model.h5')  # Clean up the saved model


def train_model(data):
    """Train the full model and save it."""
    voc_size = len(data["tokenizer"].word_index) + 1
    model = create_model(voc_size, data["max_sequence_length"])
    model.fit(data["x_train"], data["y_train"], epochs=5,
              validation_data=(data["x_val"], data["y_val"]), batch_size=32)
    model.save('trained_model.keras')


def test_integration_pipeline(data):
    """Test the entire training and saving pipeline."""
    train_model(data)
    model_path = 'trained_model.keras'
    assert os.path.exists(model_path), "Trained model was not saved properly"
    os.remove(model_path)  # Clean up the saved model


def test_non_determinism(data):
    """Test model non-determinism by comparing weights of two models."""
    voc_size = len(data["tokenizer"].word_index) + 1
    model_1 = create_model(voc_size, data["max_sequence_length"])
    model_1.fit(np.random.random((100, data["max_sequence_length"])),
                np.random.random((100, 1)), epochs=1)

    model_2 = create_model(voc_size, data["max_sequence_length"])
    model_2.fit(np.random.random((100, data["max_sequence_length"])),
                np.random.random((100, 1)), epochs=1)

    weights_1 = model_1.get_weights()
    weights_2 = model_2.get_weights()

    for w1, w2 in zip(weights_1, weights_2):
        assert not np.array_equal(w1, w2), \
            "Weights are identical, non-determinism not ensured"


def test_robustness_to_noise(data, model_fixture):
    """Test model robustness to noisy input."""
    model = model_fixture
    model.fit(data["x_train"], data["y_train"], epochs=5)

    original_score = model.evaluate(data["x_val"], data["y_val"])

    noisy_input = data["x_val"] + np.random.normal(0, 0.1, data["x_val"].shape)
    noisy_score = model.evaluate(noisy_input, data["y_val"])

    relative_change = abs(original_score[0] - noisy_score[0]) / original_score[
        0]

    assert relative_change < 0.2, f"Model is not robust to noise: {relative_change:.2f}"
