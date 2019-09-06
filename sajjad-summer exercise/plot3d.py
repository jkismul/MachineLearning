import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from design_matrix import design_matrix

def plot3d(B, p):
    fig = plt.figure()
    ax = fig.gca(projection = '3d') 
    
    n = 20
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    xmesh, ymesh = np.meshgrid(x, y)
    xmesh_array = np.reshape(xmesh, (n * n,))
    ymesh_array = np.reshape(ymesh, (n * n,))
    
    zmesh_array = design_matrix(p, xmesh_array, ymesh_array)@B
    
    zmesh = np.reshape(zmesh_array, (n, n))
    
    surf = ax.plot_surface(xmesh, ymesh, zmesh, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    ax.set_zlim( - 0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink = 0.5, aspect = 5)

    plt.show()
