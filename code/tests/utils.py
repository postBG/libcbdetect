import os
import pickle

TEST_DATA_PATH = 'tests/test_data/'


def export_test_data_to_pickle(obj, filename, basedir=TEST_DATA_PATH):
    filepath = os.path.join(basedir, filename)
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
