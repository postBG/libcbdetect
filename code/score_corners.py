import numpy as np

from corner_correlation_score import cornerCorrelationScore

def scoreCorners(img, img_angle, img_weight, corners, radius, tau):

    print('Start Scoring ...')
    height, width = img.shape
    idx_to_remove = []
    # for all corners do
    for i in range(0, len(corners.p)):
        # corner location
        u, v = np.round(corners.p[i])
        u = int(u)
        v = int(v)
        # compute corner statistics @ radius 1
        score_list = []
        for j in range(0, len(radius)):
            if(u > radius[j] and u <= width-radius[j] and v > radius[j] and v<= height-radius[j]):
                img_sub         = img[v-radius[j]:v+radius[j], u-radius[j]:u+radius[j]]
                img_sub = img[v - radius[j]:v + radius[j] + 1, u - radius[j]:u + radius[j] + 1]
                img_angle_sub = img_angle[v - radius[j]:v + radius[j] + 1, u - radius[j]: u + radius[j] + 1]
                img_weight_sub = img_weight[v - radius[j]:v + radius[j] + 1, u - radius[j]:u + radius[j] + 1]
                score_list.append(cornerCorrelationScore(img_sub, img_weight_sub, corners.v1[i, :], corners.v2[i, :]))

        max_score = np.max(score_list)
        # take highest score
        corners.score.append(max_score)
        if(max_score < tau or np.isnan(max_score)):
            idx_to_remove.append(i)

    # remove corners without edges
    corners.p = np.delete(corners.p, idx_to_remove, 0)
    corners.v1 = np.delete(corners.v1, idx_to_remove, 0)
    corners.v2 = np.delete(corners.v2, idx_to_remove, 0)
    corners.score = np.delete(corners.score, idx_to_remove, 0)

    return corners