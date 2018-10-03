import time

import numpy as np

from corner_correlation_score import cornerCorrelationScore


def scoreCorners(img, img_angle, img_weight, corners, radius, tau):
    print('Start Scoring ...')
    print('# of corners  = ', len(corners.p))
    score_st = time.time()

    height, width = img.shape
    idx_to_remove = []
    for_st = time.time()
    # for all corners do
    length = len(corners.p)
    for i in range(0, length):
        loop_st = time.time()
        # corner location
        u, v = np.round(corners.p[i]).astype(int)
        # compute corner statistics @ radius 1
        score_list = [0, 0, 0]
        for j in range(0, len(radius)):
            if (v - radius[j] >= 0 and v + radius[j] + 1 <= height
                    and u - radius[j] >= 0 and u + radius[j] + 1 <= width):
                img_sub = img[v - radius[j]:v + radius[j] + 1, u - radius[j]:u + radius[j] + 1]
                img_weight_sub = img_weight[v - radius[j]:v + radius[j] + 1, u - radius[j]:u + radius[j] + 1]
                corr_st = time.time()
                score_list[j] = cornerCorrelationScore(img_sub, img_weight_sub, corners.v1[i, :], corners.v2[i, :])
                corr_end = time.time()

        max_score = np.max(score_list)
        # take highest score
        corners.score.append(max_score)
        if (max_score < tau or np.isnan(max_score)):
            idx_to_remove.append(i)
        loop_end = time.time()
    for_end = time.time()
    # remove low scoring corners
    corners.p = np.delete(corners.p, idx_to_remove, 0)
    corners.v1 = np.delete(corners.v1, idx_to_remove, 0)
    corners.v2 = np.delete(corners.v2, idx_to_remove, 0)
    corners.score = np.delete(corners.score, idx_to_remove, 0)
    score_end = time.time()

    print('corr time = ', corr_end - corr_st)
    print('score time = ', score_end - score_st)
    print('1 loop time = ', loop_end - loop_st)
    print('for loop time = ', for_end - for_st)
    return corners
