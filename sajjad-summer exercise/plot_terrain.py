import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from predict import predict
import numpy as np

def plot_terrain(m, n, beta):
    """
    :param m: integer. number of rows in original figure
    :param n: integer. number of colums in origina figure
    :beta: numpy array, size (21, 1). regressionmodel
    """
    
    M = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            z = predict(np.array([i])/m,np.array([j])/n, beta)
            M[i,j] = z
            
    plt.figure()
    #plt.title('Terrain')
    plt.imshow(M, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
