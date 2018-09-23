import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm

from find_corners import findCorners
from non_maxima_suppression import nonMaximumSuppression
from get_image_derivatives import get_img_derivatives
from refine_corners import refineCorners
from score_corners import scoreCorners
from plot_corners import plotCorners

def main():
    img = plt.imread('../data/03.png')

    # use 3 scales to obtain a modest level of scale invariance and robustness w.r.t blur
    radius = [4, 8, 12]

    # normalize values between [0, 1]
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # convert to grayscale image
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # compute image derivatives
    img_du, img_dv, img_angle, img_weight = get_img_derivatives(img)

    # scale input image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # find initial corners
    initial_corners = findCorners(img, 0.01, radius)

    # extract corner candidates via non maximum suppressions
    NMS_corners = nonMaximumSuppression(initial_corners, 3, 0.025, 5)

    # subpixel refinement
    refined_corners = refineCorners(img_du, img_dv, img_angle, img_weight, NMS_corners, 10)

    # score corners
    final_corners = scoreCorners(img, img_angle, img_weight, refined_corners, radius, 0.01)

    # to compare with MATLAB
    sorted_p = final_corners.p[np.argsort(final_corners.p[:, 0])]

    # matplot
    plotCorners(img, final_corners)

    print('done')

    return 0

if __name__ == '__main__':
    main()
