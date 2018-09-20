import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
from scipy.stats import norm

from edge_orientation import edgeOrientations

class Corners:
    def __init__(self, NMS_corners):
        self.p = NMS_corners.astype(float)
        # init orientations to invalid (corner is invalid iff orientation=0)
        self.v1 = np.zeros(NMS_corners.shape)
        self.v2 = np.zeros(NMS_corners.shape)
        self.score = []

def refineCorners(img_du, img_dv, img_angle, img_weight, NMS_corners, r):

    print('Start Refining ...')
    corners = Corners(NMS_corners)
    height, width = img_du.shape

    idx_to_remove = []
    # for all corners do
    for i in range(0, len(corners.p)):
        # extract current corner location
        cu, cv = corners.p[i]
        cu = int(cu)
        cv = int(cv)
        # estimate edge orientations
        img_angle_sub  = img_angle[max(cv-r, 0):min(cv+r+1, height), max(cu-r, 0):min(cu+r+1, width)]
        img_weight_sub = img_weight[max(cv-r, 0):min(cv+r+1, height), max(cu-r, 0):min(cu+r+1, width)]
        v1, v2 = edgeOrientations(img_angle_sub, img_weight_sub)

        corners.v1[i] = v1
        corners.v2[i] = v2

        # continue, if invalid edge orientations
        if (v1 == [0, 0] or v2 == [0, 0]):
            continue
        #################################
        # corner orientation refinement #
        #################################

        A1 = np.zeros((2,2))
        A2 = np.zeros((2,2))

        for v in range(max(cv-r, 0), min(cv+r, height) + 1):
            for u in range(max(cu-r, 0), min(cu+r, width) + 1):
                # pixel orientation vector
                o = [img_du[v, u], img_dv[v, u]]
                if (np.linalg.norm(o) < 0.1):
                    continue
                o = o / np.linalg.norm(o)
                # robust refinement of orientation 1
                if(np.abs(np.matmul(o, v1)) < 0.25): # inlier?
                    A1[0] = A1[0] + img_du[v, u] * np.array([img_du[v, u], img_dv[v, u]])
                    A1[1] = A1[1] + img_dv[v, u] * np.array([img_du[v, u], img_dv[v, u]])

                # robust refinement of orientation 2
                if(np.abs(np.matmul(o, v2)) < 0.25): # inlier?
                    A2[0] = A2[0] + img_du[v, u] * np.array([img_du[v, u], img_dv[v, u]])
                    A2[1] = A2[1] + img_dv[v, u] * np.array([img_du[v, u], img_dv[v, u]])


        # set new corner orientation

        foo1, v1 = LA.eig(A1)      # eigenvalue, eigenvector
        min_eigenval_idx = np.argmin(foo1)
        v1 = v1[min_eigenval_idx, :]
####        v1 = v1[:, min_eigenval_idx]
        corners.v1[i] = v1

        foo2, v2 = LA.eig(A2)
        min_eigenval_idx = np.argmin(foo2)
        v2 = v2[min_eigenval_idx, :]
####        v2 = v2[:,min_eigenval_idx]
        corners.v2[i] = v2

        ##############################
        # corner location refinement #
        ##############################
        G = np.zeros((2,2))
        b = np.zeros((2,1))

        for v in range(max(cv - r, 0), min(cv + r, height) + 1):
            for u in range(max(cu-r, 0), min(cu+r, width) + 1):
                # pixel orientation vector
                o = [img_du[v, u], img_dv[v, u]]
                if (np.linalg.norm(o) < 0.1):
                    continue
                o = o / np.linalg.norm(o)

                # robust subpixel corner estimation
                if (cu != u or v != cv): # do not consider center pixel

                    #compute rel. position of pixel and distance to vectors
                    w = np.subtract([u, v], [cu, cv])
                    d1 = np.linalg.norm(w - np.matmul(w, v1) * v1)
                    d2 = np.linalg.norm(w - np.matmul(w, v2) * v2)

                    # if pixel corresponds with either of the vectors / directions
                    if ((d1 < 3 and np.abs(np.matmul(o, v1)) < 0.25) or (d2 < 3 and np.abs(np.matmul(o, v2)) < 0.25)):
                        du = img_du[v, u]
                        dv = img_dv[v, u]
                        H = np.matmul(np.transpose([[du, dv]]), np.array([[du, dv]]))
                        G = G + H
                        b = np.add(b, np.matmul(H, np.array(np.transpose([[u, v]]))))

        # set new corner location if G has full rank
        if (LA.matrix_rank(G) == 2):
            corner_pos_old = corners.p[i]
            corner_pos_new = np.transpose(np.matmul(LA.inv(G), b))

            # set corner to invalid, if position update is very large
            if (np.linalg.norm(corner_pos_new - corner_pos_old) >= 4):
                idx_to_remove.append(i)
                corners.v1[i] = [0, 0]
                corners.v2[i] = [0, 0]

            corners.p[i] = corner_pos_new ######

        # otherwise: set corner to invalid
        else:
            idx_to_remove.append(i)
            corners.v1[i] = [0, 0]
            corners.v2[i] = [0, 0]

    # remove corners without edges
    corners.p = np.delete(corners.p, idx_to_remove, 0)
    corners.v1 = np.delete(corners.v1, idx_to_remove, 0)
    corners.v2 = np.delete(corners.v2, idx_to_remove, 0)

    return corners
