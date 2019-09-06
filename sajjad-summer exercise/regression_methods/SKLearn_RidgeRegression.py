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
from ConfidenceInterval import mean_confidence_interval

def RidgeRegression(x, y, z, degree=5, alpha=10**(-6), verbose=False):
    # Split into training and test
    x_train = np.random.rand(100,1)
    y_train = np.random.rand(100,1)
    z = FrankeFunction(x_train,y_train)

    # training and finding design matrix X_
    X = np.c_[x_train,y_train]
    poly = PolynomialFeatures(degree)
    X_ = poly.fit_transform(X)
    ridge = linear_model.RidgeCV(alphas=np.array([alpha]))
    ridge.fit(X_, z)
    beta = ridge.coef_
    #intercept = ridge.intercept_

    # predict data and prepare for plotting
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta.T)

    if verbose:
        print ("X_: ", np.shape(X_))
        print ("M: ", np.shape(M))
        print ("M_: ", np.shape(M_))
        print ("predict: ", np.shape(predict))

    # show figure
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

    beta = RidgeRegression(x,y,z,5,10**(-6),True)
    print (beta)

    beta_list = beta.ravel().tolist()
    cofint = mean_confidence_interval(beta_list)
    print (cofint)
