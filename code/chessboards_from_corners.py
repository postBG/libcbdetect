from init_chessboard import initChessboard

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
        print('i = ', i)
        print(chessboard)