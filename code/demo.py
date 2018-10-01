import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm
import time

from image_preprocessing import imagePreprocessing
from find_corners import findCorners
from non_maxima_suppression import nonMaximumSuppression
from get_image_derivatives import get_img_derivatives
from refine_corners import refineCorners
from score_corners import scoreCorners
from chessboards_from_corners import chessboardsFromCorners
from plot_chessboards import plotChessboards

def main():

    img = plt.imread('../data/00.png')
    # use 3 scales to obtain a modest level of scale invariance and robustness w.r.t blur
    radius = [4, 8, 12]

    start_time = time.time()

    # compute image derivatives
    img_du, img_dv, img_angle, img_weight = get_img_derivatives(img)

    img = imagePreprocessing(img)

    # find initial corners
    corners = findCorners(img, radius)
    findCorners_time = time.time()

    # extract corner candidates via non maximum suppressions
    corners = nonMaximumSuppression(corners, 3, 0.025, 5)
    NMS_time = time.time()

    # subpixel refinement
    corners = refineCorners(img_du, img_dv, img_angle, img_weight, corners, 10)
    refineCorners_time = time.time()

    # score corners
    corners = scoreCorners(img, img_angle, img_weight, corners, radius, 0.01)
    score_time = time.time()

    # make v1(:,1)+v1(:,2) positive
    idx = corners.v1[:, 0] + corners.v1[:, 1] < 0
    corners.v1[idx, :] = -corners.v1[idx, :]
    chessboards = chessboardsFromCorners(corners)

    elapsed_time = time.time() - start_time

    print('findCorners_time = ', findCorners_time - start_time)
    print('NMS_time = ', NMS_time - findCorners_time)
    print('refine_time = ', refineCorners_time - NMS_time)
    print('score_time = ', score_time - refineCorners_time)
    print('total time = ', elapsed_time)

    # matplot
    plotChessboards(img, chessboards, corners)

    print('done')

    return 0

if __name__ == '__main__':
    main()
