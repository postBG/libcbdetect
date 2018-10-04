import os
import pickle

TEST_DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), './test_data/')


def export_test_data_to_pickle(obj, filename, basedir=TEST_DATA_PATH):
    filepath = os.path.join(basedir, filename)
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def read_test_data_from_pickle(filename, basedir=TEST_DATA_PATH):
    filepath = os.path.join(basedir, filename)

    with open(filepath, 'rb') as f:
        return pickle.load(f)
