# pylint: disable=W0012,E0611,E0401
"""
Tokenization utilities for text data preprocessing.
"""

import os
import pickle
import dvc.api

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

PARAMS = dvc.api.params_show()
INPUT_DIR = PARAMS["data_folder"]
OUTPUT_DIR = PARAMS["tokenized_folder"]


def pickle_save(obj, path):
    """Save the object to the given path using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_data(file_path):
    """Read data from file using 'with' to handle resources."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
    train = read_data(INPUT_DIR + "test.txt")[1:]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = read_data(INPUT_DIR + "test.txt")
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val = read_data(INPUT_DIR + "val.txt")
    raw_x_val = [line.split("\t")[1] for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    SEQUENCE_LENGTH = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train),
                            maxlen=SEQUENCE_LENGTH)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val),
                          maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test),
                           maxlen=SEQUENCE_LENGTH)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pickle_save(char_index, OUTPUT_DIR + "char_index.pickle")
    pickle_save(x_train, OUTPUT_DIR + "x_train.pickle")
    pickle_save(x_val, OUTPUT_DIR + "x_val.pickle")
    pickle_save(x_test, OUTPUT_DIR + "x_test.pickle")

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    pickle_save(y_train, OUTPUT_DIR + "y_train.pickle")
    pickle_save(y_val, OUTPUT_DIR + "y_val.pickle")
    pickle_save(y_test, OUTPUT_DIR + "y_test.pickle")
