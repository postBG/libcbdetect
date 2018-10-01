import numpy as np
import time
from create_correlation_patch import createCorrelationPatch

def cornerCorrelationScore(img, img_weight, v1, v2):
    # center
    center = (img_weight.shape[0] + 1) / 2
    c = [center - 1, center - 1]
    img_filter = -1 * np.ones(img_weight.shape)

    # compute gradient filter kernel (bandwith = 3 px)
    for y in range(0, img_weight.shape[0]):
        for x in range(0, img_weight.shape[1]):
            p1 = np.subtract([x, y], c)
            p2 = np.matmul(p1, v1) * v1
            p3 = np.matmul(p1, v2) * v2

            if (np.linalg.norm(p1 - p2) <= 1.5 or np.linalg.norm(p1 - p3) <= 1.5):
                img_filter[y, x] = 1

    # convert into vectors
    vec_weight = np.transpose(img_weight).reshape(-1, 1)
    vec_filter = np.transpose(img_filter).reshape(-1, 1)

    # normalize
    vec_weight = (vec_weight - np.mean(vec_weight)) / np.std(vec_weight)
    vec_filter = (vec_filter - np.mean(vec_filter)) / np.std(vec_filter)

    # compute gradient score
    score_gradient = max(np.sum(vec_weight * vec_filter) / (len(vec_weight) - 1), 0)

    # create intensity filter kernel
    st = time.time()
    template = createCorrelationPatch(np.arctan2(v1[1], v1[0]), np.arctan2(v2[1], v2[0]), c[0])
    end = time.time()
    #print('createCorrelationPatch  = ', end-st)

    # checkerboard responses
    a1 = np.sum(template.a1 * img)
    a2 = np.sum(template.a2 * img)
    b1 = np.sum(template.b1 * img)
    b2 = np.sum(template.b2 * img)

    mu = (a1 + a2 + b1 + b2) / 4

    # case 1: a=white, b=black
    score_a = min(a1 - mu, a2 - mu)
    score_b = min(mu - b1, mu - b2)
    score_1 = min(score_a, score_b)

    # case 2: b=white, a=black
    score_a = min(mu - a1, mu - a2)
    score_b = min(b1 - mu, b2 - mu)
    score_2 = min(score_a, score_b)

    # intensity score: max. of the 2 cases
    score_intensity = max(max(score_1, score_2), 0)
    if(np.isnan(score_intensity)):
        score_intensity = 0
    # final score: product of gradient and intensity score
    score = score_gradient * score_intensity

    return score