import numpy as np

from predict_corners import predictCorners

def growChessboard(chessboard, corners, border_type):
    # return immediately, if there do not exist any chessboards
    if (len(chessboard) == 0):
        return []
    # extract feature locations
    p = corners.p

    # list of unused feature elements
    unused = np.arange(0, corners.p.shape[0])
    used = chessboard[chessboard != -1]
    unused = np.delete(unused, used)

    # candidates from unused corners
    cand = p[unused, :]

    # switch border type 1 ~ 4
    if (border_type == 0):
        pred = predict




    elif (border_type == 1):

    return chessboard