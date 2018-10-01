import numpy as np
from scipy.stats import norm
import time

class Template:
    def __init__(self, height, width):
        self.a1 = np.zeros((height, width))
        self.b1 = np.zeros((height, width))
        self.a2 = np.zeros((height, width))
        self.b2 = np.zeros((height, width))

    def do_normalize(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.a1 = self.a1 / np.sum(self.a1)
        self.a2 = self.a2 / np.sum(self.a2)
        self.b1 = self.b1 / np.sum(self.b1)
        self.b2 = self.b2 / np.sum(self.b2)


def createCorrelationPatch(angle_1, angle_2, radius):

    patch_size = int(radius*2 + 1)

    # initialize template
    template = Template(patch_size, patch_size)

    # compute normals from angles
    n1 = [-np.sin(angle_1), np.cos(angle_1)]
    n2 = [-np.sin(angle_2), np.cos(angle_2)]

    # for all points in template do
    vecs = np.mgrid[0:patch_size, 0:patch_size].T.reshape(-1, 2)
    vecs = vecs - [radius, radius]

    s1s = np.matmul(vecs, np.transpose(n1))
    s2s = np.matmul(vecs, np.transpose(n2))

    dists = np.linalg.norm(vecs, axis=1)
    pdfs = norm.pdf(dists, 0, radius / 2)

    for v in range(0, patch_size):
        for u in range(0, patch_size):
            # check on which side of the normals we are
            s1 = s1s[v * patch_size + u]
            s2 = s2s[v * patch_size + u]

            pdf = pdfs[v * patch_size + u]
            if s1 <= -0.1 and s2 <= -0.1:
                template.a1[v][u] = pdf
            elif s1 >= 0.1 and s2 >= 0.1:
                template.a2[v][u] = pdf
            elif s1 <= -0.1 and s2 >= 0.1:
                template.b1[v][u] = pdf
            elif s1 >= 0.1 and s2 <= -0.1:
                template.b2[v][u] = pdf

    '''
    # for all points in template do
    for v in range(0, patch_size):
        for u in range(0, patch_size):
            # vector
            vec = [u-radius, v-radius]

            # check on which side of the normals we are
            s1 = np.matmul(vec, n1)
            s2 = np.matmul(vec, n2)

            if (s1 <= -0.1 and s2 <= -0.1):
                template.a1[v][u] = norm.pdf(np.linalg.norm(vec), 0, radius/2) # x, mu, sigma
            elif (s1 >= 0.1 and s2 >= 0.1):
                template.a2[v][u] = norm.pdf(np.linalg.norm(vec), 0, radius/2)
            elif (s1 <= -0.1 and s2 >= 0.1):
                template.b1[v][u] = norm.pdf(np.linalg.norm(vec), 0, radius/2)
            elif (s1 >= 0.1 and s2 <= -0.1):
                template.b2[v][u] = norm.pdf(np.linalg.norm(vec), 0, radius/2)
    '''

    # normalize
    template.do_normalize()
    return template