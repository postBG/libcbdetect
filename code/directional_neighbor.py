import numpy as np

def directionalNeighbor(idx, v, chessboard, corners):
    # list of neighboring elements, which are currently not in use
    unused = np.arange(0, corners.p.shape[0])
    used = chessboard[chessboard != -1]
    unused = np.delete(unused, used)

    # direction and distance to unused corners
    dir = corners.p[unused, :] - np.ones((len(unused), 1)) * corners.p[idx, :]
    dist = (dir[:, 0] * v[0] + dir[:, 1] * v[1]).reshape(-1, 1)

    dist_edge = np.subtract(dir, dist * v)
    dist_edge = np.sqrt(dist_edge[:, 0] ** 2 + dist_edge[:, 1] ** 2).reshape(-1, 1)
    dist_point = dist
    dist_point[dist_point < 0] = np.inf

    # find best neighbor
    min_idx = np.argmin(dist_point + 5 * dist_edge)
    min_dist = np.min(dist_point + 5 * dist_edge)
    neighbor_idx = unused[min_idx]

    return neighbor_idx, min_dist