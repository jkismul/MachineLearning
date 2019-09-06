import numpy as np

def k_fold_CV(k, X, y):
    n, m = np.shape(X)
    ind = np.arange(0, n, k)
    for i in ind:
        X_test, y_test = X[i:(i + k),:], y[i:(i + k)]
        if i == 0:
            X_train, y_train = X[i + k:, :], y[i + k:]
        elif i == (n - k):
            X_train, y_train = X[:i, :], y[:i]
        else:
            X_train = np.vstack([X[:i, :], X[i + k:, :]])
            y_train = np.hstack([y[:i], y[i + k:]])
        yield X_train, X_test, y_train, y_test 
        