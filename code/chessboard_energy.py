import numpy as np


def chessboardEnergy(chessboard, corners):
    h, w = chessboard.shape
    # energy: number of corners
    E_corners = - h * w

    # energy: structure
    E_structure = 0

    # walk through rows
    for j in range(0, h):
        for k in range(0, w - 2):
            x = corners.p[chessboard[j, k:k + 3]]
            E_structure = max(E_structure,
                              np.linalg.norm(x[0, :] + x[2, :] - 2 * x[1, :]) / np.linalg.norm(x[0, :] - x[2, :]))

    # walk through columns
    for j in range(0, w):
        for k in range(0, h - 2):
            x = corners.p[chessboard[k:k + 3, j]]
            E_structure = max(E_structure,
                              np.linalg.norm(x[0, :] + x[2, :] - 2 * x[1, :]) / np.linalg.norm(x[0, :] - x[2, :]))

    E = E_corners + h * w * E_structure

    return E
