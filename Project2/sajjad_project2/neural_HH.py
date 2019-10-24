import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import SGDClassifier
import scipy.linalg as scl
import numpy as np
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit
from hiddenLayer import Layer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numba import jit

class MultilayerNeuralNetwork:
    def __init__(self, X_data, Y_data, n_hidden_layers, n_neurons, eta, batch_size=100):
        self.X_data_full = X_data  # N x M matrix
        self.Y_data_full = Y_data  # N x 1 vector
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_neurons = n_neurons
        self.eta = eta
        self.n_categories = Y_data.shape[1]

        # Initiate hidden layer(s)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers = []
        if n_hidden_layers > 0:
            self.create_hidden_layers()

        # Initiate outer layer
        #self.output_weights = np.ones(shape=[self.n_neurons, self.n_categories])/5       # initial weights are all = 1
        self.output_weights = np.random.randn(self.n_neurons, self.n_categories)       # Initiate Normal Distribution
        self.output_bias = np.zeros(shape=[self.n_categories, 1]) + 0.01

        # Output
        self.a_in = None
        self.out = None

        # Training data
        self.X_train = X_data
        self.Y_train = Y_data

        self.b_error = 0
        self.batch_size = batch_size

    def update_train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def create_hidden_layers(self):
        # Create the hidden layers
        self.hidden_layers = [Layer(self.n_features, self.n_neurons, self.eta, activation_function='ELU')]
        for l in range(self.n_hidden_layers-1):
            self.hidden_layers.append(Layer(self.n_neurons, self.n_neurons, self.eta, activation_function='ELU'))

    def train_batch(self, batch_size):
        indices = np.arange(np.shape(self.X_data_full)[0])
        batch = np.random.choice(indices, batch_size)
        self.X_train = self.X_data_full[batch, :]
        self.Y_train = self.Y_data_full[batch, :]

        self.feed_forward()
        self.backpropagation()

    def backpropagation_batch(self):
        # Update weights and bias in outer layer
        self.output_weights -= self.eta * np.matmul(self.a_in.T, self.b_error/self.batch_size)
        self.output_bias -= self.eta * np.sum(self.b_error/self.batch_size)

        # Update the hidden layers, starting from the one closest to the outer layer
        reversed_layers = list(reversed(self.hidden_layers))
        for layer in reversed_layers:
            layer.backwards_propagation_b()

    def activation(self, z):
        return z

    def a_derivative(self, z):
        # Linear regression
        return np.ones(np.shape(z))

    @jit
    def feed_forward(self):
        # Define input
        a_in = self.X_train

        # Go through all hidden layers
        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

    @jit
    def backpropagation(self):
        # Calculate outer error, number of inputs us%%%%%%% own implementation regression %%%%%%%ed to scale the error
        outer_error = (self.out - self.Y_train)/self.n_inputs
        #print('MSE: ', np.mean((self.out-self.Y_train)**2))%%%%%%% own implementation regression %%%%%%%

        # calculate errors in hidden layers
        reversed_layers = list(reversed(self.hidden_layers))
        f_weights = self.output_weights
        f_error = outer_error
        for layer in reversed_layers:
            layer.calculate_error(f_error, f_weights)
            f_weights = layer.weights
            f_error = layer.error

        # Update outer weights and bias
        self.output_weights -= self.eta * np.matmul(self.a_in.T, outer_error)
        self.output_bias -= self.eta * np.sum(outer_error)
        for layer in reversed_layers:
            layer.backwards_propagation()

        return np.mean((self.out - self.Y_train)**2)

    def save_weights(self):
        # Save weights in order to iterate further later
        np.save('outer_weights.npy', self.output_weights)
        np.save('output_bias.npy', self.output_bias)

        counter = 1
        for layer in self.hidden_layers:
            np.save('weights_{}.npy'.format(counter), layer.weights)
            np.save('bias_{}.npy'.format(counter), layer.biases)
            counter += 1

    def accuracy(self, X, Y):
        # Print MSE and R2 with regards to training data, and new input test data X, Y
        a_in = self.X_train

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

        # Accuracy on training data
        #print(np.c_[self.out, self.Y_train])
        #loss = np.sum(self.out * np.log(self.Y_train) + (1-self.out) * np.log(1-self.Y_train))
        Error_T = abs(np.mean(self.out - self.Y_train))
        #R2 = 1 - np.sum((self.Y_train - self.out)**2)/np.sum((self.Y_train-np.mean(self.Y_train))**2)
        print('Error Training Data: ', Error_T)
        #print('R2 Training Data: ', R2)

        # Test data
        a_in = X_test

        for layer in self.hidden_layers:
            a_out = layer.feed_forward(a_in)
            a_in = a_out

        # Outer layer
        self.a_in = a_in
        self.out = np.matmul(self.a_in, self.output_weights) + self.output_bias

        # Accuracy on test data
        #loss = np.sum(self.out * np.log(Y_test) + (1-self.out) * np.log(1-Y_test))
        Error_test = abs(np.mean(self.out - Y_test))
        #R2 = 1 - np.sum((Y_test - self.out)**2)/np.sum((Y_test-np.mean(Y_test))**2)
        print('Error Test Data: ', Error_test)
        #print('R2 Test data: ', R2)
        return Error_test
#def ReadData():

#importing data set(s)
filename = 'default of credit card clients.xls'
nanDict = {} #this does nothing with this data set
#read file
df = pd.read_excel(filename,header=1,skiprows=0,index_col=0,na_values=nanDict) 
#rename last column
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
#Replace nonsensical values in PAY_i columns with 0
for i in [0,2,3,4,5,6]:
    col = 'PAY_{}'.format(i)
    df[col].replace(to_replace=-2, value = 0, inplace=True)
#shuffle dataset by row
df.sample(frac=1)
    
# Define features and targets 
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    
# Categorical variables to one-hots, setting nonsensical values to 0
onehotencoder1 = OneHotEncoder(categories='auto')
onehotencoder2 = OneHotEncoder(categories='auto',drop='first')

    # sets number of elements in onehot vectors automatically from data.
Xt= ColumnTransformer(
        [("one", onehotencoder1, [1]),("two", onehotencoder2, [2,3]),],
        remainder="passthrough"
    ).fit_transform(X)

# Train-test split
trainingShare = 0.8
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(Xt, y, train_size=trainingShare, \
                                                     random_state=seed)
    
#scale data, except one-hotted
sc = StandardScaler()
XTrain_fitting = XTrain[:,11:]
XTest_fitting = XTest[:,11:]
#removes mean, scales by std
XTrain_scaler = sc.fit_transform(XTrain_fitting)
XTest_scaler = sc.transform(XTest_fitting)
#puts together the complete model matrix again
XTrain_scaled=np.c_[XTrain[:,:11],XTrain_scaler]
XTest_scaled = np.c_[XTest[:,:11],XTest_scaler]


    
    
    #return XTrain_scaled,XTest_scaled,yTrain,yTest

if __name__ == '__main__':
    # Generate data
    # Set random seed
    np.random.seed(62)

    # Split into training and test data
    X_train = XTrain_scaled
    X_test  = XTest_scaled
    Y_train = yTrain
    Y_test  = yTest

    # Initialise Neural Network
    MLN = MultilayerNeuralNetwork(X_train, np.atleast_2d(Y_train), n_hidden_layers=2, n_neurons=31, eta=1e-5, batch_size=50)
    epochs = 600
    batch_size = 1000
    iterations = int(np.shape(Xt)[0]/batch_size)
    indices = np.arange(0, np.shape(Xt)[0])

    for e in range(epochs):
        for i in range(iterations):
            MLN.train_batch(batch_size)
        MLN.accuracy(X_test, Y_test)
        print ("epoch:",e)


    # Print MSE and R2 for test data and training data
    MLN.accuracy(X_test, Y_test)
