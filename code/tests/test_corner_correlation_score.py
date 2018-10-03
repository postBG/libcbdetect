import unittest
import pickle

from corner_correlation_score import cornerCorrelationScore


class CornerCorrelationScore(unittest.TestCase):
    TEST_DATA_PATH = 'test_data/cornerCorrelationScore/test_input.pkl'

    def setUp(self):
        with open(self.TEST_DATA_PATH, 'rb') as f:
            test_data = pickle.load(f)
            self.img_sub = test_data['img_sub']
            self.img_weight_sub = test_data['img_weight_sub']
            self.v1 = test_data['v1']
            self.v2 = test_data['v2']

    def test_cornerCorrelationScore(self):
        self.assertAlmostEqual(0.0, cornerCorrelationScore(self.img_sub, self.img_weight_sub, self.v1, self.v2))
