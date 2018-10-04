import unittest
import matplotlib.pyplot as plt
import numpy as np

from refine_corners import refineCorners
from get_image_derivatives import getImgDerivatives

class TestRefineCorner(unittest.TestCase):
    def test_xx(self):
        img = plt.imread('../data/00.png')
        img_du, img_dv, img_angle, img_weight = getImgDerivatives(img)

        np.testing.assert_allclose([1.0, np.pi, np.nan],
                                   [1, np.sqrt(np.pi) ** 2, np.nan])

