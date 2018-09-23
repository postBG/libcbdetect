import matplotlib.pyplot as plt


def plotCorners(img, corners):
    plt.figure(figsize = (16, 8))
    implot = plt.imshow(img)
    plt.scatter(corners.p[:, 0], corners.p[:, 1], color = 'red')
    plt.show()

    return 0