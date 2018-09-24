import numpy as np
from directional_neighbor import directionalNeighbor
def initChessboard(corners, idx):
    # not enough corners
    if (corners.p.shape[0] < 9):
        return []

    # init chessboard hypothesis
    chessboard = -np.ones((3, 3), dtype=int)

    # extract feature index and orientation (central element)
    v1 = corners.v1[idx, :]
    v2 = corners.v2[idx, :]
    chessboard[1, 1] = idx

    # find left/right/top/bottom neighbors
    dist1 = np.empty((2, 1))
    dist2 = np.empty((6, 1))
    chessboard[1, 2], dist1[0] = directionalNeighbor(idx, +v1, chessboard, corners)
    chessboard[1, 0], dist1[1] = directionalNeighbor(idx, -v1, chessboard, corners)
    chessboard[2, 1], dist2[0] = directionalNeighbor(idx, +v2, chessboard, corners)
    chessboard[0, 1], dist2[1] = directionalNeighbor(idx, -v2, chessboard, corners)

    # find top-left/top-right/bottom-left/bottom-right neighbors
    chessboard[0, 0], dist2[2] = directionalNeighbor(chessboard[1, 0], -v2, chessboard, corners)
    chessboard[2, 0], dist2[3] = directionalNeighbor(chessboard[1, 0], +v2, chessboard, corners)
    chessboard[0, 2], dist2[4] = directionalNeighbor(chessboard[1, 2], -v2, chessboard, corners)
    chessboard[2, 2], dist2[5] = directionalNeighbor(chessboard[1, 2], +v2, chessboard, corners)

    # initialization must be homogenously distributed
    if (any(np.isinf(dist1)) or any(np.isinf(dist2)) or np.std(dist1) / np.mean(dist1) > 0.3 or np.std(dist2) / np.mean(
            dist2) > 0.3):
        return []

    return chessboard