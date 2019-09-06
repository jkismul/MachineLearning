import imageio
import numpy as np

from sklearn import linear_model

from design_matrix import design_matrix
from k_fold_CV import k_fold_CV
import metrics
from terrain_plot import terrain_plot

# Import data
data = imageio.imread('SRTM_data_Norway_1.tif')
a, b = np.shape(data)
z =  np.reshape(data, [b * a,])
   
x = np.linspace(0, 1, b)
y = np.linspace(0, 1, a)

x, y = np.meshgrid(x, y)
x = np.reshape(x, (a * b,))
y = np.reshape(y, (a * b,))

# Parameters
number_of_models = 4
highest_number_of_betas = 27
k = 5

# Choose model
#model = 'OLS'
model = 'Ridge'
#model = 'Lasso'

####################################################################
# Ordinary least squares

if model == 'OLS':
    
    # Regression statistics matrices
    MSE_train = np.zeros([number_of_models,])
    R2_train = np.zeros([number_of_models,])
    MSE_test = np.zeros([number_of_models,])
    R2_test = np.zeros([number_of_models,])
    Beta_conf_interval = np.zeros([highest_number_of_betas, number_of_models], dtype='O')
    Bias2 = np.zeros([number_of_models,])
    Variance_error = np.zeros([number_of_models,])

    for m in range(number_of_models):

        # Polynomial degree
        p = m + 2  
    
        # Design matrix
        X = design_matrix(p, x, y) 
        Xm, Xn = np.shape(X) 
    
        # Least squares
        normal_equation = X.T@X
        B = np.linalg.solve(normal_equation, X.T@z)
    
        # Regression statistics calcualtions
        zhat = X@B
        MSE_train[m] = metrics.mean_squared_error(z, zhat)
        R2_train[m] = metrics.r2_score(z, zhat)
        Beta_conf_interval[:Xn, m] = metrics.confidance_interval(z, zhat, p, normal_equation, B)
        Bias2[m] = metrics.bias2(z, zhat)
        Variance_error[m] = metrics.variance_error(zhat)
    
        # Cross valdiation 2-fold
        CV_pred = []
    
        for X_train, X_test, z_train, z_test in k_fold_CV(k, X, z):
        
            # Least squares
            B = np.linalg.solve(X_train.T@X_train, X_train.T@z_train)
        
            # Cross validation predictions 
            CV_pred.append(X_test@B)
    
        # Cross valdiation regression statistics calculations
        CV_pred = np.reshape(CV_pred, [a * b,])    
        MSE_test[m] = metrics.mean_squared_error(z, CV_pred)
        R2_test[m] = metrics.r2_score(z, CV_pred)
    
##################################################################
# Ridge regression

if model == 'Ridge':
    
    # Regression statistics matrices
    MSE = np.zeros([number_of_models])
    R2 = np.zeros([number_of_models])
    Beta_conf_interval = np.zeros([highest_number_of_betas, number_of_models], dtype='O')
    Bias2 = np.zeros([number_of_models])
    Variance_error = np.zeros([number_of_models])
    Alphas = np.zeros([number_of_models,])

    # Lambda values
    lambdas = np.arange(0, 1, 0.01)

    for m in range(number_of_models):
    
        # Polynomial degree
        p = m + 2 
    
        # Design matrix
        X = design_matrix(p, x, y) 
        Xm, Xn = np.shape(X) 
        
        # Ridge regression 
        model = linear_model.RidgeCV(alphas = lambdas, fit_intercept = False, cv = k)
        model.fit(X, z)
    
        # Regression statistics calculations
        zhat = model.predict(X)
        MSE[m,] = metrics.mean_squared_error(z, zhat)
        R2[m,] = metrics.r2_score(z, zhat)
        Beta_conf_interval[:Xn, m] = metrics.confidance_interval(z, zhat, p, X.T@X + model.alpha_ * np.eye(Xn), model.coef_)
        Bias2[m] = metrics.bias2(z, zhat)
        Variance_error[m] = metrics.variance_error(zhat)    
        Alphas[m] = model.alpha_
      
######################################################################################
# Lasso regression

if model == 'Lasso':

    # Regression statistics matrices
    MSE = np.zeros([number_of_models,])
    R2 = np.zeros([number_of_models,])
    Beta_conf_interval = np.zeros([highest_number_of_betas, number_of_models], dtype='O')
    Bias2 = np.zeros([number_of_models,])
    Variance_error = np.zeros([number_of_models,])
    Alphas = np.zeros([number_of_models,])

    # Lambda values
    lambdas = np.arange(0.0001, 0.0101, 0.0001)

    for m in range(number_of_models):
    
            # Polynomial degree
            p = m + 2 
            
            # Design matrix
            X = design_matrix(p, x, y) 
            Xm, Xn = np.shape(X) 
    
            # Lasso regression    
            model = linear_model.LassoCV(alphas = lambdas, fit_intercept = False, cv = k)
            model.fit(X, z)

            # Regression statistics calculations        
            zhat = model.predict(X)
            MSE[m] = metrics.mean_squared_error(z, zhat)   
            R2[m] = metrics.r2_score(z, zhat)
            Beta_conf_interval[:Xn, m] = metrics.confidance_interval(z, zhat, p, X.T@X + model.alpha_ * np.eye(Xn), model.coef_)   
            Bias2[m] = metrics.bias2(z, zhat)
            Variance_error[m] = metrics.variance_error(zhat)
            Alphas[m] = model.alpha_
        
###########################################################################
        
# Plot model        
image = np.reshape(zhat, (a, b)).astype(int)
terrain_plot(image)




