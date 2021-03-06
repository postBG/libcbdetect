import numpy as np
from scipy.signal import convolve2d


def get_img_derivatives(img):
    # sobel masks
    du = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

    dv = [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]]

    img_du = convolve2d(img, du, mode='same')
    img_dv = convolve2d(img, dv, mode='same')
    img_angle = np.arctan2(img_dv, img_du)
    img_weight = np.sqrt(img_du ** 2 + img_dv ** 2)

    # correct angle to lie in between [0,pi]
    img_angle[img_angle < 0] = img_angle[img_angle < 0] + np.pi
    img_angle[img_angle > np.pi] = img_angle[img_angle > np.pi] - np.pi

    return img_du, img_dv, img_angle, img_weight
