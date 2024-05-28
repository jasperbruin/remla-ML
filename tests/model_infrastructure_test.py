# pylint: disable=E0401, E0611, E1101, C0114

import os
import numpy as np
import pytest
import tensorflow as tf
import psutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Setting a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


@pytest.mark.skipif(not tf.config.list_physical_devices('GPU'),
                    reason="GPU is not available")
def test_gpu_available():
    """Test if GPU is available."""
    assert tf.config.list_physical_devices('GPU'), "GPU is not available"


def test_memory_usage():
    """Test memory usage before and after model training."""
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 ** 2

    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Artificially increase memory usage
    dummy_input = np.random.random((10000, 2))
    dummy_output = np.random.random((10000, 2))
    model.fit(dummy_input, dummy_output, epochs=1, batch_size=10)

    memory_after = process.memory_info().rss / 1024 ** 2  # Memory usage in MB
    memory_diff = memory_after - memory_before

    assert memory_diff < 100, "Memory usage increased significantly"


def test_model_saving_loading():
    """Test saving and loading of the model."""
    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model_path = 'temp_model.h5'
    model.save(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    assert loaded_model.input_shape == (None, 2)
    os.remove(model_path)  # Clean up the saved model


def train_model():
    """Train a simple model and save it."""
    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(np.random.random((1000, 2)),
              np.random.random((1000, 2)), epochs=5)
    model.save('trained_model.keras')


def test_integration_pipeline():
    """Test the entire training and saving pipeline."""
    train_model()
    model_path = 'trained_model.keras'
    assert os.path.exists(model_path), "Trained model was not saved properly"
    os.remove(model_path)  # Clean up the saved model


def test_non_determinism():
    """Test model non-determinism by comparing weights of two models."""
    model_1 = Sequential()
    model_1.add(Dense(2, input_dim=2))
    model_1.compile(loss='mean_squared_error', optimizer='adam')
    model_1.fit(np.random.random((100, 2)),
                np.random.random((100, 2)), epochs=1)

    model_2 = Sequential()
    model_2.add(Dense(2, input_dim=2))
    model_2.compile(loss='mean_squared_error', optimizer='adam')
    model_2.fit(np.random.random((100, 2)),
                np.random.random((100, 2)), epochs=1)

    weights_1 = model_1.get_weights()
    weights_2 = model_2.get_weights()

    for w1, w2 in zip(weights_1, weights_2):
        assert not np.array_equal(w1, w2), \
            "Weights are identical, non-determinism not ensured"


def test_robustness_to_noise():
    """Test model robustness to noisy input."""
    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    model.fit(np.random.random((1000, 2)),
              np.random.random((1000, 2)), epochs=5)

    original_score = model.evaluate(np.random.random((100, 2)),
                                    np.random.random((100, 2)))

    noisy_input = (np.random.random((100, 2)) +
                   np.random.normal(0, 0.1, (100, 2)))
    noisy_score = model.evaluate(noisy_input, np.random.random((100, 2)))

    assert (abs(original_score - noisy_score) <
            0.05), "Model is not robust to noise"
