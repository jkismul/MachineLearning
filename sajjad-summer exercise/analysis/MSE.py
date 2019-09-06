import numpy as np
from predict import predict

def MeanSquaredError(x, y, z, beta):
    """
    Calculates the Mean Squared Error
    :param y: numpy vector with y data, size (n, 1)
    :param x: numpy vector with x data, size (n, 1)
    :param beta: model
    :return: Mean squared error
    """
    # Calculate z_hat, the predicted z-values
    z_hat = predict(x, y, beta)

    # Calculate MSE
    MSE = 0
    for i in range(0, len(z)):
        MSE += np.power(z[i, 0] - z_hat[i, 0], 2)
    MSE = MSE/len(z)

    return MSE
