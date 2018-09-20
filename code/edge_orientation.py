import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm

from find_modes_mean_shift import findModesMeanShift

def edgeOrientations(img_angle, img_weight):
    # init v1 and v2
    v1 = [0, 0]
    v2 = [0, 0]

    # number of bins (histogram parameters)
    bin_num = 32

    # convert images to vectors
    vec_angle = img_angle.T.reshape(-1)
    vec_weight = img_weight.T.reshape(-1)

    # convert angles from normals to directions
    vec_angle = vec_angle + np.pi / 2
    vec_angle[vec_angle > np.pi] -= np.pi

    # create histogram
    angle_hist = np.zeros((bin_num))
    for i in range(0, len(vec_angle)):
        bin = int(max(min(np.floor(vec_angle[i] / (np.pi / bin_num)), bin_num - 1), 0))
        angle_hist[bin] = angle_hist[bin] + vec_weight[i];

    modes, angle_hist_smoothed = findModesMeanShift(angle_hist, 1);

    # if only one or no mode => return invalid corner
    if (len(modes) <= 1):
        return v1, v2

    # compute orientation at modes
    new = modes[:, 0] * np.pi / bin_num
    new = np.reshape(new, (-1, 1))
    modes = np.hstack((modes, new))

    # extract 2 strongest modes and sort by angle
    modes = modes[:2]
    modes = modes[np.argsort(modes[:, 2])]

    # compute angle between modes
    delta_angle = min(modes[1, 2] - modes[0, 2], modes[0, 2] + np.pi - modes[1, 2])

    # if angle too small => return invalid corner
    if (delta_angle <= 0.3):
        return v1, v2

    # set statistics: orientations
    v1 = [np.cos(modes[0, 2]), np.sin(modes[0, 2])]
    v2 = [np.cos(modes[1, 2]), np.sin(modes[1, 2])]

    return v1, v2