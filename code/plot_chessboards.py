import matplotlib.pyplot as plt

def plotChessboards(img, chessboards, corners):
    plt.figure(figsize = (16, 8))
    implot = plt.imshow(img)

    for key in chessboards.keys():
        idx = chessboards[key].flatten()
        pos = corners.p[idx]
        plt.scatter(pos[:, 0], pos[:, 1], color = 'red')
    plt.show()

    return 0