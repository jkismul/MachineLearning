import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from frankeFunction import FrankeFunction
from scipy import misc

def OSLRegression(x, y, z, degree=5, l=0.1):
    # Split into training and test
    x_train = np.random.rand(100,1)
    y_train = np.random.rand(100,1)
    z = FrankeFunction(x_train,y_train)

    # traning data and finding design matrix X_
    X = np.c_[x_train,y_train]
    poly = PolynomialFeatures(degree)
    X_ = poly.fit_transform(X)
    clf = linear_model.LassoCV()
    clf.fit(X_, z)
    print ("X_: ", np.shape(X_))
    beta = clf.coef_

    # predicting and preparing plot
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta.T)

    # plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_,y_,predict.reshape(20,20),cmap=cm.coolwarm,linewidth=0,antialiased=False)
    plt.show()

    return beta


if __name__ == '__main__':
    #terrain1 = misc.imread('data.tif',flatten=0)
    x = np.arange(0, 1, 0.05).reshape((20,1))
    y = np.arange(0, 1, 0.05).reshape((20,1))
    z = FrankeFunction(x, y)

    beta = OSLRegression(x,y,z,14)
