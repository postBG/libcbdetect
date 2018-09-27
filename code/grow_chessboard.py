import numpy as np

from predict_corners import predictCorners
from assign_closest_corners import assignClosestCorners

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
    if (border_type == 0): # right
        pred = predictCorners(p[chessboard[:,- 3],:], p[chessboard[:, - 2],:], p[chessboard[:, -1],:])
        idx = assignClosestCorners(cand, pred)
        if (idx.any()): #if idx is nonzero matrix
            chessboard = np.hstack((chessboard, unused[idx].reshape(-1, 1)))

    elif (border_type == 1): # down
        pred = predictCorners(p[chessboard[- 3, :],:], p[chessboard[- 2, :],:], p[chessboard[-1, :],:])
        idx = assignClosestCorners(cand, pred)
        if (idx.any()):
            chessboard = np.vstack((chessboard, unused[idx].reshape(1, -1)))

    elif (border_type == 2): # left
        pred = predictCorners(p[chessboard[:, 2],:], p[chessboard[:, 1],:], p[chessboard[:, 0],:])
        idx = assignClosestCorners(cand, pred)
        if (idx.any()):
            chessboard = np.hstack((unused[idx].reshape(-1, 1), chessboard))

    elif (border_type == 3): # up
        pred = predictCorners(p[chessboard[2, :],:], p[chessboard[1, :],:], p[chessboard[0, :],:])
        idx = assignClosestCorners(cand, pred)
        if (idx.any()):
            chessboard = np.vstack((unused[idx].reshape(1, -1), chessboard))

    return chessboard