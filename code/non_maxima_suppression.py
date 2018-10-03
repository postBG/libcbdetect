import numpy as np


def nonMaximumSuppression(img, n, tau, margin):
    height, width = img.shape
    maxima = []

    for j in range(n + margin, height - n - margin + 1, n + 1):
        for i in range(n + margin, width - n - margin + 1, n + 1):
            # initial value
            maxi, maxj = i, j
            # update maxval if found
            maxval = img[j][i]

            for j2 in range(j, j + n + 1):
                for i2 in range(i, i + n + 1):
                    if (img[j2][i2] > maxval):
                        maxi, maxj = i2, j2
                        maxval = img[j2][i2]

            failed = 0

            for j2 in range(maxj - n, min(maxj + n, height - margin) + 1):
                for i2 in range(maxi - n, min(maxi + n, width - margin) + 1):
                    currval = img[j2][i2]
                    if (currval > maxval and (i2 < i or i2 > i + n or j2 < j or j2 > j + n)):
                        failed = 1
                        break
                if (failed):
                    break

            if (maxval >= tau and failed == 0):
                maxima.append([maxi, maxj])

    maxima = np.array(maxima)

    return maxima
