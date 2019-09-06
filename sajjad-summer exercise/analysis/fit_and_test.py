from OrdinaryLeastSquare import ols
from RidgeRegression import RidgeRegression
from Lasso import Lasso
from MSE import MeanSquaredError
from plot_terrain import plot_terrain
from Analysis import R2
from predict import predict
import numpy as np

def fit_and_test(x_train, y_train, z_train, x_test, y_test, z_test):

    num_cols = 100
    num_rows = 100
    
    #fit data with Ordinary Least Squares
    print('ordinary lest squared')
    beta_OLS = ols(x_train, y_train, z_train)
    MSE_OLS = MeanSquaredError(x_test, y_test, z_test, beta_OLS)
    R2_OLS = R2(z_test, predict(x_test, y_test, beta_OLS))
    plot_terrain(num_rows, num_cols, beta_OLS)
    print('Mean squared error {} '.format(MSE_OLS))
    print('R2 score {} '.format(R2_OLS))
    
    #fit data with Ridge Regression, test model with testset and calculate MSE
    print('Ridge Regression')
    beta_Ridge = RidgeRegression(x_train, y_train, z_train)
    MSE_Ridge = MeanSquaredError(x_test, y_test, z_test, beta_Ridge)
    R2_Ridge = R2(z_test, predict(x_test, y_test, beta_Ridge))
    plot_terrain(num_rows, num_cols, beta_Ridge)
    print('Mean squared error {} '.format(MSE_Ridge))
    print('R2 score {} '.format(R2_Ridge))
    
    #fit data with Lasso Regression, test model with testset and calculate MSE
    print('Lasso Regression')
    beta_Lasso = Lasso(np.array(x_train), np.array(y_train), np.array(z_train), 5).reshape((21, 1))
    MSE_Lasso = MeanSquaredError(x_test, y_test, z_test, beta_Lasso)
    R2_Lasso = R2(z_test, predict(x_test, y_test, beta_Lasso))
    plot_terrain(num_rows, num_cols, beta_Lasso)
    print('Mean squared error {} '.format(MSE_Lasso))
    print('R2 score {} '.format(R2_Lasso))
