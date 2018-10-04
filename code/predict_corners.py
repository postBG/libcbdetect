import numpy as np


def predictCorners(p1, p2, p3):
    # compute vectors
    v1 = p2 - p1
    v2 = p3 - p2

    # predict angles
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    a3 = 2 * a2 - a1

    # predict scales
    s1 = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2).reshape(-1, 1)
    s2 = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2).reshape(-1, 1)
    s3 = 2 * s2 - s1

    # predict p3 (the factor 0.75 ensures that under extreme
    # distortions (omnicam) the closer prediction is selected)
    temp1 = 0.75 * np.matmul(s3, np.ones((1, 2)))
    temp2 = np.transpose(np.array([np.cos(a3), np.sin(a3)]))
    pred = np.add(p3, temp1 * temp2)

    return pred
