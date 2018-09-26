import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm

def get_img_derivatives(img):
    # sobel masks
    du = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

    dv = [[-1, -1, -1],
          [ 0,  0,  0],
          [ 1,  1,  1]]

    img_du      = convolve2d(img, du, mode='same')
    img_dv      = convolve2d(img, dv, mode='same')
    img_angle   = np.arctan2(img_dv, img_du)
    img_weight  = np.sqrt(img_du ** 2 + img_dv ** 2)

    # correct angle to lie in between [0,pi]
    img_angle[img_angle <     0] = img_angle[img_angle <     0] + np.pi
    img_angle[img_angle > np.pi] = img_angle[img_angle > np.pi] - np.pi
    '''
    h, w = img_angle.shape
    for i in range(0, h):
        for j in range(0, w):
            if (img_angle[i][j] < 0):
                img_angle[i][j] += np.pi
            elif (img_angle[i][j] > np.pi):
                img_angle[i][j] -= np.pi
    '''

    return img_du, img_dv, img_angle, img_weight