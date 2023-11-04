import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from typing import Any, Literal

from FFNN import FFNN
from Scheduler import Scheduler


from autograd import grad
from sklearn.datasets import load_breast_cancer

def train_pred_FFNN(
        network: FFNN,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        eta_vals: np.ndarray,
        lmbda_vals: np.ndarray,
        scheduler: Scheduler,
        batches: int,
        epochs: int,
    ):
    '''Trains the given FFNN network for a grid of different learning rates (eta) and regularisation parameters (lmbda).
    Returns the test MSE and R2 score for each combination of eta and lmbda.'''

    mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

    # Do grid search of eta and lambda values
    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            network.reset_weights_and_bias()
            scheduler.set_eta(eta_vals[i])
            network.fit(x_train, y_train, scheduler, batches, epochs, lmbda_vals[j])
            y_pred = network.predict(x_test)
            mse_vals[i][j] = mse(y_pred, y_test)
            r2_vals[i][j] = r2(y_pred, y_test)

    return mse_vals, r2_vals

def train_pred_skl(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        eta_vals: np.ndarray,
        lmbda_vals: np.ndarray,
        hidden_layers: tuple, #(int),
        activation: Literal['relu', 'identity', 'logistic', 'tanh'],
        solver: Literal['lbfgs', 'sgd', 'adam'],
        batches: int,
        epochs: int,
        regression: bool = True,
        seed: int = 10
    ):
    '''Trains either a scikit-learn regression network or a scikit-learn classification network
    for a grid of different learning rates (eta) and regularisation parameters (lmbda).
    Returns the test MSE and R2 score for each combination of eta and lmbda.'''

    mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

    if (regression): # if regression, use scikit-learn's regression neural network
        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network = MLPRegressor(hidden_layers, activation, solver, lmbda_vals[j], eta_vals[i], epochs, batch_size = x_train.shape[0]//batches, random_state = seed)
                network.fit(x_train, y_train.flatten())
                y_pred = network.predict(x_test)
                mse_vals[i][j] = mse(y_pred, y_test)
                r2_vals[i][j] = r2(y_pred, y_test)

    else: # else, use scikit-learn's classifier neural network
        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network = MLPClassifier(hidden_layers, activation, solver, lmbda_vals[j], eta_vals[i], batch_size = x_train.shape[0]//batches, max_iter = epochs)
                network.fit(x_train, y_train.flatten())
                y_pred = network.predict(x_test)
                mse_vals[i][j] = mse(y_pred, y_test)
                r2_vals[i][j] = r2(y_pred, y_test)

    return mse_vals, r2_vals

def plot_heatmap(
        data: np.ndarray,
        save_path: str,
        ylabel: str,
        xlabel: str,
        yticks: Any,
        xticks: Any,
    ):
    '''Creates heatmap of MSE or R2 data from grid search over learning rates and regularisation parameters.'''

    fig, ax = plt.subplots(figsize = (10, 8))
    sns.heatmap(data, annot = True, cmap = "viridis", square = True, yticklabels = yticks, xticklabels = xticks)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(save_path)

def mse(y_tilde, y):
    '''Calculates the mean square error of a prediction y_tilde.'''
    mse = 1/len(y) * np.sum((y-y_tilde)**2)
    return mse

def r2(y_tilde, y):
    '''Calculates the R^2 score of a prediction y_tilde.'''
    a = np.sum((y-y_tilde)**2)
    b = np.sum((y-np.mean(y))**2)
    return 1 - a/b



''' Writing Logistic regression code which will be used to compare with FFNN '''

"""
- Define your cost function before you start writing your code.
- Define your design matrix before you start writing your code.

Can I skip the cost function definition?
"""
def learning_schedule(t, t0, t1):
    return t0/(t+t1)

def logreg_cost(beta, X, z, L):
    cost = -(X.T@beta).dot(z - ( 1.0 / (1 + np.exp(-z) ) ) ) + L*(beta.T).dot(beta)
    
    return cost


def logreg_sgd(X, z, M=32, lamba=5, epochs=50):   #*** Epochs
    '''
    Function which performs Stochastic Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 1000 but can be changed
    rate is a factor you can add to the learning rate eta
    M is the size of the mini-batch used in each iteration
    epochs is number of epochs
    '''
    
    #beta_list = list()
    b_shape = np.shape(X)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)
    print(b_shape)

    change = 0
    momentum = 0.9  # IDEAL MOMENTUM

    n = X.shape[0]  # Number of samples
    print(f'I RAIN {n}')
    m = int(n/M) #number of mini-batches
   
    t0, t1 = 5, 50  #scheduling constants for learning rate
    
   
    for e in range(epochs):    
        for i in range(m):
            indices = np.random.choice(n, size=M, replace=False)  # Randomly select indices for the batch

            X_batch = X[indices]
            z_batch = np.ravel(z)[indices] 
            
            #1/m ????***
            gradient =  grad(logreg_cost)(beta, X_batch, z_batch, lamba) # -X_batch.T @ (z_batch -( 1.0 / (1 + np.exp(-z)))) # CHECK SIGMOID FUNC
            #gradient = -X_batch.T @ (z_batch -( 1.0 / (1 + np.exp(-z_batch)))) # CHECK SIGMOID FUNC

            eta = learning_schedule(epochs*m+i, t0, t1)

            change = eta*gradient + momentum*change
            #beta_list.append(beta)

            beta -= change 
        
        save_e = e

    predict = X.dot(beta)  
    mse = 1/(n*n) * np.sum((np.ravel(z)-predict)**2)
    info = [f'Mse = {mse}, momentum = {momentum}, last learning rate = {eta}, batch size = {M}, epochs = {save_e}']
    
    print(f'MSE for stochastic gradient descent with batches is {mse} \n')
    print(f'{info}\n')
    return predict, beta, mse, info


X,y = load_breast_cancer(return_X_y=True)
print(np.shape(X))

predict, beta, mse, info = logreg_sgd(X, y, M=5, lamba=5, epochs=50)

print(info)