import numpy as np

def design_matrix(deg, x, y):

    m = len(x)
    polynomial = [np.ones((m)), x, y]

    # polynomials
    for p in range(2, deg + 1):
        polynomial.append(x**p)
        polynomial.append(y**p)
    
    # cross-terms 
    for m in range(1, deg):
        for n in range(1, deg):
            polynomial.append((x**m) * (y**n))

    X = np.stack((polynomial), axis = -1) 
    return X
