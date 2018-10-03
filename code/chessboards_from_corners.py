import numpy as np

from init_chessboard import initChessboard
from chessboard_energy import chessboardEnergy
from grow_chessboard import growChessboard


def chessboardsFromCorners(corners):
    print('Start Structure Recovery ...')

    # initialize chessboards
    chessboards = {}
    proposal = {}
    p_energy = np.zeros((4, 1))

    # for all seeds
    for i in range(0, corners.p.shape[0]):
        # output
        if i % 100 == 0:
            print(i + 1, "/", corners.p.shape[0])

        # init 3x3 chessboard from seed i
        chessboard = initChessboard(corners, i)

        if len(chessboard) == 0 or chessboardEnergy(chessboard, corners) > 0:
            continue

        # try growing chessboard
        while True:
            # compute current energy
            energy = chessboardEnergy(chessboard, corners)

            # compute proposals and energies
            for j in range(0, 4):
                proposal[j] = growChessboard(chessboard, corners, j)
                p_energy[j] = chessboardEnergy(proposal[j], corners)

            # find best proposal
            min_idx = np.argmin(p_energy)

            # accept best proposal, if energy is reduced
            if p_energy[min_idx] < energy:
                chessboard = proposal[min_idx]
            else:
                break

        # if chessboard has low energy(corresponding to high quality)
        if chessboardEnergy(chessboard, corners) < -10:

            # check if new chessboard proposal overlaps with existing chessboards
            overlap = np.zeros((len(chessboards), 2))
            for j in range(0, len(chessboards)):
                mask = np.isin(chessboards[j], chessboard)
                if np.any(mask):
                    overlap[j, 0] = 1
                    overlap[j, 1] = chessboardEnergy(chessboards[j], corners)

            # print(chessboards)
            if not np.any(overlap[:, 0]):
                idx = len(chessboards)
                chessboards[idx] = chessboard
            else:
                row, col = np.where(overlap == 1)
                if not np.any(overlap[row, 1] <= chessboardEnergy(chessboard, corners)):
                    for item in row:
                        chessboards.pop(item, None)
                    idx = len(chessboards)
                    chessboards[idx] = chessboard

    return chessboards
