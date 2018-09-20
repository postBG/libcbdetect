import matplotlib.pyplot as plt


def plotCorners(img, corners):

    implot = plt.imshow(img)
    plt.scatter(corners.p[:, 0], corners.p[:, 1])
    plt.show()

    return 0