from init_chessboard import initChessboard
from chessboard_energy import chessboardEnergy
from grow_chessboard import growChessboard

def chessboardsFromCorners(corners):
    print('Start Structure Recovery ...')

    # initialize chessboards
    chessboards = []

    # for all seeds
    for i in range(0, corners.p.shape[0]):
        # output
        if (i % 100 == 0):
            print(i + 1, "/", corners.p.shape[0])

        # init 3x3 chessboard from seed i
        chessboard = initChessboard(corners, i)

        if (len(chessboard) == 0 or chessboardEnergy(chessboard, corners) > 0):
            continue

            # try growing chessboard
        while (1):
            # compute current energy
            energy = chessboardEnergy(chessboard, corners)

            # compute proposals and energies
            for j in range(0, 4):
                proposal[j] = growChessboard(chessboard, corners, j);
                p_energy[j] = chessboardEnergy(proposal[j], corners);



