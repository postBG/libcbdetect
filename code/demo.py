import time
import argparse

import matplotlib.pyplot as plt

from chessboards_from_corners import chessboardsFromCorners
from find_corners import findCorners
from get_image_derivatives import get_img_derivatives
from image_preprocessing import imagePreprocessing
from non_maxima_suppression import nonMaximumSuppression
from plot_chessboards import plotChessboards
from refine_corners import refineCorners
from score_corners import scoreCorners

from tests.utils import export_test_data_to_pickle


def get_arguments():
    parser = argparse.ArgumentParser(description='Calibration Demo')
    parser.add_argument('--img_path', type=str, default='../data/00.png', help='learning rate (default: 0.0001)')
    return parser.parse_args()


def main(args):
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

    end_time = time.time()

    print('findCorners_time = ', findCorners_time - start_time)
    print('NMS_time = ', NMS_time - findCorners_time)
    print('refine_time = ', refineCorners_time - NMS_time)
    print('score_time = ', score_time - refineCorners_time)
    print('growing time = ', end_time - score_time)
    print('total time = ', end_time - start_time)

    # matplot
    plotChessboards(img, chessboards, corners)

    print('done')

    return 0


if __name__ == '__main__':
    main(get_arguments())
