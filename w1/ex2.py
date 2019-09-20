import numpy as np
import matplotlib.pyplot as plt

# "Dataset"
x = np.random.rand(100)
y = 5*x*x+0.1*np.random.rand(100)

# Design matrix
X = np.zeros((len(x),3))
X[:,0] = 1
X[:,1] = x
X[:,2] = x*x

# find parameters
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# make prediction
ytilde = X @ beta


#sort data
order = np.argsort(x)
xs = np.array(x)[order]
ytildes = np.array(ytilde)[order]


# Make plots to compare
fig,ax = plt.subplots()
ax.plot(xs,ytildes,label='Fit')
ax.scatter(x,y,c='r',label='Data')
ax.legend()
plt.show()