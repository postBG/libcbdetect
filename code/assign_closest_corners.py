import numpy as np

def assignClosestCorners(cand,pred):

    # return error if not enough candidates are available
    if (cand.shape[0] < pred.shape[0]):
        return 0

    # distance matrix
    D = np.zeros((cand.shape[0], pred.shape[0]))
    for i in range(0, pred.shape[0]):
        delta = np.subtract(cand, np.ones((cand.shape[0], 1)) * pred[i, :])
        D[:, i] = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)

    idx = np.zeros((1, D.shape[1]))
    # search greedily for closest ccorners
    for i in range(0, pred.shape[0]):
        nonzero_D = np.ma.masked_equal(D, 0.0, copy=False)
        row, col =  np.unravel_index(np.argmin(nonzero_D, axis=None), nonzero_D.shape)
        idx[col] = row
        D[row, :] = np.inf
        D[:, col] = np.inf

    return idx