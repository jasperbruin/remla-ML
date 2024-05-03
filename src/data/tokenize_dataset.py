import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


INPUT_DIR = "/workspaces/remla-ML-group3/data/external/"
OUPUT_DIR = "/workspaces/remla-ML-group3/tokenized/"

def pickel_save(obj, path):
    """ Save the object to the given path """ 
    with open(path, "wb") as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    train = [line.strip() for line in open(INPUT_DIR + "test.txt", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open(INPUT_DIR + "test.txt", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open(INPUT_DIR + "val.txt", "r").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length=200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    if not os.path.exists(OUPUT_DIR):
        os.makedirs(OUPUT_DIR)

    pickel_save(char_index, OUPUT_DIR + "char_index.pickle")
    pickel_save(x_train, OUPUT_DIR + "x_train.pickle")
    pickel_save(x_val, OUPUT_DIR + "x_val.pickle")
    pickel_save(x_test, OUPUT_DIR + "x_test.pickle")


    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    pickel_save(y_train, OUPUT_DIR + "y_train.pickle")
    pickel_save(y_val, OUPUT_DIR + "y_val.pickle")
    pickel_save(y_test, OUPUT_DIR + "y_test.pickle")
