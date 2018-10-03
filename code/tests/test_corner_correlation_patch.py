import unittest
import numpy as np

from create_correlation_patch import createCorrelationPatch


class TestCreateCorrelationPathch(unittest.TestCase):
    def test_createCorrelationPatch(self):
        template = createCorrelationPatch(0., 0., 1)

        expected_a1 = np.array([[0.10650697891920068, 0.7869860421615987, 0.10650697891920068],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])
        expected_a2 = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.10650697891920068, 0.7869860421615987, 0.10650697891920068]])
        expected_b1 = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
        expected_b2 = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

        np.testing.assert_almost_equal(template.a1, expected_a1)
        np.testing.assert_almost_equal(template.a2, expected_a2)
        np.testing.assert_almost_equal(template.b1, expected_b1)
        np.testing.assert_almost_equal(template.b2, expected_b2)

    def test_createCorrelationPatch2(self):
        template = createCorrelationPatch(0.4, 0.4, 2)
        expected_a1 = np.array([[0.007086828361808099, 0.031760961212477186, 0.05236497232889289, 0.031760961212477186,
                                 0.007086828361808099],
                                [0.031760961212477186, 0.1423427527293902, 0.23468352415495441, 0.1423427527293902,
                                 0.031760961212477186], [0.0, 0.0, 0.0, 0.23468352415495441, 0.05236497232889289],
                                [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        expected_a2 = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
             [0.05236497232889289, 0.23468352415495441, 0.0, 0.0, 0.0],
             [0.031760961212477186, 0.1423427527293902, 0.23468352415495441, 0.1423427527293902, 0.031760961212477186],
             [0.007086828361808099, 0.031760961212477186, 0.05236497232889289, 0.031760961212477186,
              0.007086828361808099]])
        expected_b1 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan]])
        expected_b2 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan]])

        np.testing.assert_almost_equal(template.a1, expected_a1)
        np.testing.assert_almost_equal(template.a2, expected_a2)
        np.testing.assert_almost_equal(template.b1, expected_b1)
        np.testing.assert_almost_equal(template.b2, expected_b2)
