import numpy as np

def mean_squared_error(y, yhat):
    res = y - yhat
    mse = np.divide(res.T@res, len(yhat))
    return mse

def r2_score(y, yhat):
    res = y - yhat
    ymean = np.mean(y)
    ssr = y - ymean * np.ones((len(y),))
    R2 = 1 - np.divide(res.T@res, ssr.T@ssr)
    return R2

def bias2(z,zhat):
    n = len(zhat)
    bias2 = np.sum((z - (np.mean(zhat)))**2) / n
    return bias2

def variance_error(yhat):
    variance = np.mean(yhat**2) - np.mean(yhat)**2
    return variance 

def variance_model(y, yhat, n):
    m = len(y)
    res = y - yhat
    variance = (1 / (m - n - 1)) * res.T@res
    return variance

def covariance_matrix(XTX, variance):
    n, n = np.shape(XTX)
    XTXinv = np.linalg.solve(XTX, np.eye(n))
    covariance_matrix = variance * XTXinv 
    return covariance_matrix

def confidance_interval(z, zhat, p, X, B):
    Variance_model = variance_model(z, zhat, p)
    Beta_std = np.sqrt(np.diag(covariance_matrix(X, Variance_model)))
    conf_upper = B + 2 * Beta_std
    conf_lower = B - 2 * Beta_std
    Beta_conf = list(zip(conf_lower, conf_upper))
    return Beta_conf
