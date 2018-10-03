import unittest

from corner_correlation_score import cornerCorrelationScore
from tests.utils import read_test_data_from_pickle


def read_test_input_from(filename):
    test_input = read_test_data_from_pickle(filename)
    return test_input['img_sub'], test_input['img_weight_sub'], test_input['v1'], test_input['v2']


class CornerCorrelationScore(unittest.TestCase):
    def test_cornerCorrelationScore(self):
        img, img_weight, v1, v2 = read_test_input_from('cornerCorrelationScore/test_input.pkl')
        self.assertAlmostEqual(0.0, cornerCorrelationScore(img, img_weight, v1, v2))

    def test_cornerCorrelationScore2(self):
        img, img_weight, v1, v2 = read_test_input_from('cornerCorrelationScore/test_input2.pkl')
        self.assertAlmostEqual(0.0005118449002040401, cornerCorrelationScore(img, img_weight, v1, v2))
