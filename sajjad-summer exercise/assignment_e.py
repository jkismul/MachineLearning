import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
#from fit_and_test import fit_and_test
#from plot_terrain import plot_terrain
#from k_fold import k_fold_validation
from RidgeRegression import RidgeRegression
from MSE import MeanSquaredError
from Lasso import Lasso
from R2 import R2
from predict import predict
from bootstrap import bootstrap
from frankeFunction import FrankeFunction
#from OrdinaryLeastSquare import ols

#Read data
#terrain_over_Norway = imread('SRTM_data_Norway_1.tif')

#Choose to lo look at a little part of the data-set:
#terrain = terrain_over_Norway[0:100,0:100]
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
n = np.size(x, 0)
terrain = FrankeFunction(x, y)  


"""
#plot of the original
plt.figure()
plt.title('Terrain')
plt.imshow(terrain_over_Norway, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('oiginal_subset.png')
plt.show()
"""
num_rows = len(terrain)
num_cols = len(terrain[0])
num_observations = num_rows*num_cols
X = np.zeros((num_observations,3))

#make a matrix with all the values from the data on the form [x y z]
ind = 0;
for i in range(0, num_rows):
    for j in range(0, num_cols):
        X[ind,0] = i              #x
        X[ind,1] = j              #y
        X[ind,2] = terrain[i,j]   #z
        ind += 1

x = X[:,0, np.newaxis]
y = X[:,1, np.newaxis]
z = X[:,2, np.newaxis] 

#np.random.shuffle(X)                #shuffle rows in x
n = 9

mse_o = np.zeros([n])
mse_r = np.zeros([n])
mse_l = np.zeros([n])
r_o = np.zeros([n])
r_r = np.zeros([n])
r_l = np.zeros([n])
b_o = np.zeros([n])
b_r = np.zeros([n])
b_l = np.zeros([n])
v_o = np.zeros([n])
v_r = np.zeros([n])
v_l = np.zeros([n])

index = 0
degrees = np.linspace(1,n,n)

print(np.shape(degrees))
print(np.shape(mse_o))
for degree in range(1,n+1):
    print(degree)
    result_ols = bootstrap(x, y, z, degree, 'OLS', n_bootstrap=100)
    result_Ridge = bootstrap(x, y, z, degree, 'Ridge' , n_bootstrap=100)
    result_Lasso = bootstrap(x, y, z, degree, 'Lasso', n_bootstrap=10)
    mse_o[degree-1] = result_ols[0]
    r_o[degree-1] = result_ols[1]
    b_o[degree-1] = result_ols[2]
    v_o[degree-1] = result_ols[3]
    mse_r[degree-1] = result_Ridge[0]
    r_r[degree-1] = result_Ridge[1]
    b_r[degree-1] = result_Ridge[2]
    v_r[degree-1] = result_Ridge[3]
    mse_l[degree-1] = result_ols[0]
    r_l[degree-1] = result_Lasso[1]
    b_l[degree-1] = result_Lasso[2]
    v_l[degree-1] = result_Lasso[3]

print(np.shape(degrees))
print(np.shape(mse_o))

plt.figure()
plt.title('MSE and R2-score for different degees of OLS')    
plt.plot(degrees, mse_o, degrees, r_o)
#plt.axis([1,21,-0.1,0.1])
plt.legend(['MSE', 'R2'])
plt.xlabel('degree')
plt.ylabel('error')
plt.savefig('MSE_degree_9.png')
plt.show()

plt.figure()
plt.title('MSE and R2-score for different degees of Ridge')    
plt.plot(degrees, mse_r, degrees, r_r)
#plt.axis([1,21,-0.1,0.1])
plt.legend(['MSE', 'R2'])
plt.xlabel('degree')
plt.ylabel('error')
plt.savefig('Ridge_degree_9.png')
plt.show()

plt.figure()
plt.title('MSE and R2-score for different degees of Lasso')    
plt.plot(degrees, mse_l, degrees, r_l)
#plt.axis([1,21,-0.1,0.1])
plt.legend(['MSE', 'R2'])
plt.xlabel('degree')
plt.ylabel('error')
plt.savefig('Lasso_degree_9.png')
plt.show()

#Try with the whole set for training and testing
#fit_and_test(x, y, z, x, y, z)

"""
#find the optimal value for lamda in ridge regression
test_error1 = np.zeros((10))
test_error2 = np.zeros((10))
x_train = x[0:49,0, np.newaxis]
y_train = y[0:49,0, np.newaxis]
z_train = z[0:49,0, np.newaxis]
x_test = x[50:100,0, np.newaxis]
y_test = y[50:100,0, np.newaxis]
z_test = z[50:100,0, np.newaxis]

x_values = np.linspace(-7,-3,10)
index = 0
for Lambda in x_values:
    beta = RidgeRegression(x_train, y_train, z_train, l=10**Lambda)
    MSE = MeanSquaredError(x_test, y_test, z_test, beta)
    R2_score = R2(z_test, predict(x_test, y_test, beta))
    test_error1[index] = MSE
    test_error2[index] = R2_score
    index += 1

# Make figure
fig, ax1 = plt.subplots()
ax1.plot(x_values, test_error1, 'bo-')
ax1.set_xlabel('lambda')
ax1.set_ylabel('MSE', color='b')
#ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x_values, test_error2, 'r*-')
ax2.set_ylabel('R2 score', color='r')
#ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('MSE and R2 Score for different lambda values')
fig.tight_layout()
plt.show()



index = 0
x_values = np.linspace(-9,-1,10)
for alfa in x_values:
    beta = Lasso(x_train, y_train, z_train, a = 10**alfa)
    MSE = MeanSquaredError(x_test, y_test, z_test, beta.reshape(21,1))
    R2_score = R2(z_test, predict(x_test, y_test, beta.reshape(21,1)))
    test_error1[index] = MSE
    test_error2[index] = R2_score
    index += 1
 


# Make figure
fig, ax1 = plt.subplots()
ax1.plot(x_values, test_error1, 'bo-')
ax1.set_xlabel('alpha')
ax1.set_ylabel('MSE', color='b')
#ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x_values, test_error2, 'r*-')
ax2.set_ylabel('R2 score', color='r')
#ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('MSE and R2 Score for different alpha values')
fig.tight_layout()
plt.show()

#try with bootstrap and find the best beta-value
result_OLS = bootstrap(x, y, z, 5, 'OLS')
result_Ridge = bootstrap(x, y, z, 5, 'Ridge')
result_Lasso = bootstrap(x, y, z, 5, 'Lasso', 10)

#MSE_M, R2_M, bias, variance
print('Ordinary Least squares:')
print('MSE: {} '.format(result_OLS[0]))
print('R2: {} '.format(result_OLS[1]))
print('bias: {} '.format(result_OLS[2]))
print('variance: {} '.format(result_OLS[3]))

print('Ridge:')
print('MSE: {} '.format(result_Ridge[0]))
print('R2: {} '.format(result_Ridge[1]))
print('bias: {} '.format(result_Ridge[2]))
print('variance: {} '.format(result_Ridge[3]))

print('Lasso:')
print('MSE: {} '.format(result_Lasso[0]))
print('R2: {} '.format(result_Lasso[1]))
print('bias: {} '.format(result_Lasso[2]))
print('variance: {} '.format(result_Lasso[3]))
   
print('Best fit OLS')      
plot_terrain(100, 100, best_beta_OLS)
print('Mean squared error {} '.format(min_mse_OLS))
print('R2 score {} '.format(best_R2_OLS))

print('Best fit Ridge')      
plot_terrain(100, 100, best_beta_Ridge)
print('Mean squared error {} '.format(min_mse_Ridge))
print('R2 score {} '.format(best_R2_Ridge))

print('Best fit Lasso')      
plot_terrain(100, 100, best_beta_Lasso)
print('Mean squared error {} '.format(min_mse_Lasso))
print('R2 score {} '.format(best_R2_Lasso))
"""