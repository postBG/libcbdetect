import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm

def findModesMeanShift(hist, sigma):
    # efficient mean-shift approximation by histogram smoothing

    # compute smoothed histogram
    hist_smoothed = np.zeros((len(hist)))
    for i in range(0, len(hist)):
        j = np.arange(-np.round(2 * sigma), np.round(2 * sigma) + 1)
        idx = np.mod(i + j, len(hist))
        hist_smoothed[i] = np.sum(hist[idx] * norm.pdf(j, 0, sigma))

    modes = []
    # check if at least one entry is non-zero
    # (otherwise mode finding may run infinitly)
    if np.abs(hist_smoothed - hist_smoothed[0]).any() < 1e-5:
        return

    modes = np.empty((0, 2), float)
    # mode finding
    for i in range(0, len(hist_smoothed)):
        j = i
        while (1):
            j1 = np.mod(j + 1, len(hist))
            j2 = np.mod(j - 1, len(hist))
            h0 = hist_smoothed[j]
            h1 = hist_smoothed[j1]
            h2 = hist_smoothed[j2]
            if (h1 >= h0 and h1 >= h2):
                j = j1
            elif (h2 > h0 and h2 > h1):
                j = j2
            else:
                break

        if len(modes) == 0:
            modes = np.array([[j, hist_smoothed[j]]])
        elif (j not in modes[:, 0]):
            new = np.array([[j, hist_smoothed[j]]])
            modes = np.concatenate((modes, new))

    # sort in descending order
    modes = modes[np.argsort(-modes[:, 1])]

    return modes, hist_smoothed
