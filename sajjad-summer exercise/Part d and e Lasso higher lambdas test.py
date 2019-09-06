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
      
######################################################################################
# Lasso regression

# Regression statistics matrices
MSE = np.zeros([number_of_models,])
R2 = np.zeros([number_of_models,])
Beta_variance = np.zeros([highest_number_of_betas, number_of_models])
Bias = np.zeros([number_of_models,])
Variance_error = np.zeros([number_of_models,])

# Lambda values
lambdas = np.arange(0.01, 1.01, 0.01)

for m in range(number_of_models):
    
        # Polynomial degree
        p = m + 2 
    
        # Design matrix
        X = design_matrix(p, x, y) 
        Xm, Xn = np.shape(X) 
    
        # Lasso regression    
        model = linear_model.LassoCV(alphas = lambdas, fit_intercept = False, cv = k)
        model.fit(X, z)
        print('p =', p, ', lambda = ', model.alpha_)

        # Regression statistics calculations        
        zhat = model.predict(X)
        MSE[m] = metrics.mean_squared_error(z, zhat)   
        R2[m] = metrics.r2_score(z, zhat)
        Variance_model = metrics.variance_model(z, zhat, p)
        Beta_variance[:Xn, m] = np.diag(metrics.covariance_matrix(X, Variance_model))     
        Bias[m] = metrics.bias(z, zhat)
        Variance_error[m] = metrics.variance_error(zhat)
        
###########################################################################
        
# Plot model        
image = np.reshape(zhat, (a, b)).astype(int)
terrain_plot(image)




