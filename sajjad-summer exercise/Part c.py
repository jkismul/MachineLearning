import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

from Frankefunction import FrankeFunction
from design_matrix import design_matrix
from k_fold_CV import k_fold_CV
import metrics

# Data   
n = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

n = n * n
x, y = np.meshgrid(x, y)
x = np.reshape(x, (n,))
y = np.reshape(y, (n,))
z = FrankeFunction(x, y)

# Parameters
number_of_models = 4
highest_number_of_betas = 27
k = 5
lambdas = np.arange(0.0001, 0.0101, 0.0001)
n = 100

# Regression statistics matrices
MSE_train = np.zeros([number_of_models, n])
R2_train = np.zeros([number_of_models, n])
MSE_test = np.zeros([number_of_models, n])
R2_test = np.zeros([number_of_models, n])
Beta_conf_interval = np.zeros([highest_number_of_betas, number_of_models, n], dtype='O')
Bias2 = np.zeros([number_of_models, n])
Variance_error = np.zeros([number_of_models, n])

for m in range(number_of_models):
    for l in range(len(lambdas)):
        print(l)
        # Polynomial degree
        p = m + 2  
    
        # Design matrix
        X = design_matrix(p, x, y) 
        Xm, Xn = np.shape(X) 
     
        # Lasso regression
        model = linear_model.Lasso(alpha = lambdas[l], fit_intercept = False)
        model.fit(X, z)
        
        # Regression statistics matrices
        zhat = model.predict(X)
        MSE_train[m, l] = metrics.mean_squared_error(z, zhat)
        R2_train[m, l] = metrics.r2_score(z, zhat)
        Beta_conf_interval[:Xn, m, l] = metrics.confidance_interval(z, zhat, p, X.T@X + lambdas[l] * np.eye(Xn), model.coef_)
        Bias2[m, l] = metrics.bias2(z, zhat)
        Variance_error[m, l] = metrics.variance_error(zhat)
    
        # Cross validation 2-fold
        CV_pred = []
    
        for X_train, X_test, z_train, z_test in k_fold_CV(k, X, z):
        
            # Lasso regression
            model = linear_model.Lasso(alpha = lambdas[l], fit_intercept = False)
            model.fit(X_train, z_train)
            
            # Cross validation predictions
            CV_pred.append(model.predict(X_test))
        
        # Cross valdiation regression statistics 
        CV_pred = np.reshape(CV_pred, [n * number_of_models,])    
        MSE_test[m, l] = metrics.mean_squared_error(z, CV_pred)
        R2_test[m, l] = metrics.r2_score(z, CV_pred)

# Plot model
fig = plt.figure()
ax = fig.gca(projection = '3d') 
    
n = 20
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
xmesh, ymesh = np.meshgrid(x, y)
xmesh_array = np.reshape(xmesh, (n * n,))
ymesh_array = np.reshape(ymesh, (n * n,))
    
zmesh_array = model.predict(design_matrix(p, xmesh_array, ymesh_array))
    
zmesh = np.reshape(zmesh_array, (n, n))
    
surf = ax.plot_surface(xmesh, ymesh, zmesh, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

ax.set_zlim( - 0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink = 0.5, aspect = 5)

plt.show()
