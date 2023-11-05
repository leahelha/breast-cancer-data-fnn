import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from typing import Any, Literal

from FFNN import FFNN
import Scheduler

from sklearn.model_selection import train_test_split, KFold
from autograd import grad
import autograd.numpy as anp
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
        regression: bool = True
    ):
    '''Trains the given FFNN network for a grid of different learning rates (eta) and regularisation parameters (lmbda).
    Returns the test MSE and R2 score for each combination of eta and lmbda.'''


    if (regression):
        mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
        r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network.reset_weights_and_bias()
                scheduler.set_eta(eta_vals[i])
                network.fit(x_train, y_train, scheduler, batches, epochs, lmbda_vals[j])
                y_pred = network.predict(x_test)
                acc_vals[i][j] = network.accuracy(y_pred, y_test)
                mse_vals[i][j] = mse(y_pred, y_test)
                r2_vals[i][j] = r2(y_pred, y_test)

        return mse_vals, r2_vals
    else:
        acc_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network.reset_weights_and_bias()
                scheduler.set_eta(eta_vals[i])
                network.fit(x_train, y_train, scheduler, batches, epochs, lmbda_vals[j])
                y_pred = network.predict(x_test)
                acc_vals[i][j] = network.accuracy(y_pred, y_test)
        
        return acc_vals

def train_pred_skl(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        eta_vals: np.ndarray,
        lmbda_vals: np.ndarray,
        hidden_layers: tuple[int],
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


    if (regression): # if regression, use scikit-learn's regression neural network
        mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
        r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network = MLPRegressor(hidden_layer_sizes = hidden_layers, activation = activation, solver = solver, alpha = lmbda_vals[j], learning_rate_init = eta_vals[i], max_iter = epochs, batch_size = x_train.shape[0]//batches, random_state = seed)
                network.fit(x_train, y_train.flatten())
                y_pred = network.predict(x_test)
                mse_vals[i][j] = mse(y_pred, y_test)
                r2_vals[i][j] = r2(y_pred, y_test)

        return mse_vals, r2_vals

    else: # else, use scikit-learn's classifier neural network
        acc_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

        # Do grid search of eta and lambda values
        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                network = MLPClassifier(hidden_layer_sizes = hidden_layers, activation = activation, solver = solver, alpha = lmbda_vals[j], learning_rate_init = eta_vals[i], batch_size = x_train.shape[0]//batches, max_iter = epochs)
                network.fit(x_train, y_train.flatten())
                acc_vals[i][j] = network.score(x_test, y_test)


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

def save_parameters(parameters_file, file_path):
    '''Saves the parameters for a given run of the FFNN model'''
    filename = file_path / "parameters.txt"

    with open(filename, "w") as outfile:
        outfile.write(parameters_file)

def cost(beta, n, X, z, lamba=0):
    residual = z - X @ beta
    cost = anp.dot(residual.T, residual) / n + lamba * anp.dot(beta.T, beta)

    return cost

def gradients(beta, n, X, z, lamba=0, Auto=False):
    '''
    For OLS regression set lamba=0, for Ridge set lamba=1
    
    '''

    if lamba not in [0,1]:
        raise ValueError('The lambda (lamba) value must be 0 or 1. 0 for OLS and 1 for Ridge')
    
    gradient = (2.0 /n) * X.T @ (X @ beta - z) + 2*lamba*beta
    
    if Auto is True:
        '''Implementing Autograd'''
        gradient = grad(cost)(beta, n, X, z, lamba=0)
    
    return gradient

def gd(x_train, x_test, y_train, y_test, method='adagrad', iterations=1000, rate=1, Auto=False):   #beta ***
    '''
    Function which performs Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 1000 but can be changed
    rate is a factor you can add to the learning rate eta
    '''

    b_shape = np.shape(x_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    n = x_train.shape[0]  # Number of samples
    #beta_list = list()

    # Hessian matrix
    x = np.concatenate(x_train, x_test)
    H = (2.0/n)* x.T @ x
    # Get the eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)
    
    # Attempting to find an optimal learning rate
    eta = rate*(1.0/np.max(EigValues))  # learning rate   #IDEAL rate = 3, or 1

    sched = None
    
    if method == 'adagrad':
        sched = Scheduler.Adagrad(eta)
    elif method == 'rmsprop':
        sched = Scheduler.RMS_prop(eta, beta=0.9)
    else:
        sched = Scheduler.Adam(eta, beta1=0.9, beta2=0.999)

    for i in range(iterations):
        
        gradient = gradients(beta, n, x_train, y_train, lamba=0, Auto=Auto)

        #beta_list.append(beta)
        change = sched.update_change(gradient)
        beta -= change

        save_iter = i
        if np.linalg.norm(change) < 1e-3:
            break
    
    predict_test = x_test @ beta
    predict = x @ beta

    mse_val = mse(predict_test, y_test)
    
    info = f'Method {method} \n iterations = {save_iter}', f'learning rate = {eta}', f'mse= {mse_val}'
    
    print('For Gradient Descent\n')
    print(f'{info}\n')
    return predict, beta, mse_val, info

def gd_momentum(x_train, x_test, y_train, y_test, method='adagrad', iterations=1000, rate=1, Auto=False):   #beta ***
    '''
    Function which performs Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 1000 but can be changed
    rate is a factor you can add to the learning rate eta
    '''

    b_shape = np.shape(x_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    n = x_train.shape[0]  # Number of samples
    #beta_list = list()

    change = 0
    momentum = 0.9    #IDEAL MOMENTUM
    # Hessian matrix
    x = np.concatenate(x_train, x_test)
    H = (2.0/n)* x.T @ x
    # Get the eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)
    
    # Attempting to find an optimal learning rate
    eta = rate*(1.0/np.max(EigValues))  # learning rate   #IDEAL rate = 3, or 1

    sched = None
    
    if method == 'adagrad':
        sched = Scheduler.AdagradMomentum(eta, momentum)
    elif method == 'rmsprop':
        sched = Scheduler.RMS_propMomentum(eta, beta=0.9, momentum = momentum)
    else:
        sched = Scheduler.AdamMomentum(eta, beta1=0.9, beta2=0.999, momentum = momentum)

    for i in range(iterations):
        
        gradient = gradients(beta, n, x_train, y_train, lamba=0, Auto=Auto)

        #beta_list.append(beta)
        change = sched.update_change(gradient)
        beta -= change

        save_iter = i
        if np.linalg.norm(change) < 1e-3:
            break
    
    predict_test = x_test @ beta
    predict = x @ beta

    mse_val = mse(predict_test, y_test)
    
    info = f'Method {method} \n iterations = {save_iter}', f'momentum = {momentum}', f'learning rate = {eta}', f'mse= {mse_val}'
    
    print('For Gradient Descent with momentum\n')
    print(f'{info}\n')
    return predict, beta, mse_val, info

def learning_schedule(t, t0, t1):
    return t0/(t+t1)

def sgd_momentum(x_train, x_test, y_train, y_test, method='adagrad', M=32, epochs=1, Auto=False):   #*** Epochs
    '''
    Function which performs Stochastic Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    
    M is the size of the mini-batch used in each iteration
    epochs is number of epochs
    '''
    # print('X shape and z shape')
    # print(np.shape(X))
    # print(np.shape(z))

    b_shape = np.shape(x_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    # print('shape is')
    # print(np.shape(beta))
    beta_list = list()

    change = 0
    momentum = 0.9  # IDEAL MOMENTUM
    n = x_train.shape[0]  # Number of samples
    m = int(n/M) #number of mini-batches
   
    t0, t1 = 80, 50  #scheduling constants for learning rate
    t = 0 
    mom1 = 0
    mom2 = 0

    sched = None
    
    if method == 'adagrad':
        sched = Scheduler.AdagradMomentum(eta, momentum)
    elif method == 'rmsprop':
        sched = Scheduler.RMS_propMomentum(eta, beta=0.9, momentum = momentum)
    else:
        sched = Scheduler.AdamMomentum(eta, beta1=0.9, beta2=0.999, momentum = momentum)

    for e in range(epochs):    
        Giter = 0

        for i in range(m):
            indices = np.random.choice(n, size=M, replace=False)  # Randomly select indices for the batch

            x_batch = x_train[indices]
            y_batch = y_train[indices] 
            
            gradient = gradients(beta, M, x_batch, y_batch, lamba=0, Auto=Auto)
            

            beta_list.append(beta)

            eta = learning_schedule(epochs*m+i, t0, t1)
            sched.set_eta(eta)

            change = sched.update_change(gradient)
            beta -= change

            # save_iter = i
            # if np.linalg.norm(change) < 1e-3:
            #     break

        save_e = e
    
    predict_test = x_test @ beta
    x = np.concatenate(x_train, x_test)
    predict = x @ beta

    mse_val = mse(predict_test, y_test)
    abs_error_avg= 1/(len(y_test))*np.sum(np.abs(np.ravel(y_test)-predict_test)) 

    info = [f'Method {method} \n mse = {mse}, momentum = {momentum}, last learning rate = {eta}, batch size = {M}, epochs = {save_e}']
    
    print(f'MSE for stochastic gradient descent with batches is {mse_val} \n avg abs error {abs_error_avg}')
    print(f'{info}\n')
    return predict, beta, mse_val, info

''' Writing Logistic regression code which will be used to compare with FFNN '''

"""
- Define your cost function before you start writing your code.
- Define your design matrix before you start writing your code.

Can I skip the cost function definition?
"""

np.random.seed(11)


def learning_schedule(t, t0, t1):
    return t0/(t+t1)

def logreg_cost(n, beta, X, z, L):
    # res = z - ( 1.0 / (1 + np.exp(-z) ) ) 
    # cost = -(1/n)*res.T@res + L*(beta.T).dot(beta) 
    
    cost = -np.sum(z*(beta@X) - np.log(1 + np.exp(beta@X)))
    return cost


''' Change this to gradient descent '''
def logreg_gd(X, z, method='adagrad', lamba=2, iterations=1000, rate=1):   #*** Epochs
    '''
    Function which performs Stochastic Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 1000 but can be changed
    lamba is the regularization parameter lambda used in the cross entropy cost function
        -> We have chosen to use the analytical gradient, thus you will not find a cost function with the lamba parameter

    rate is a factor you can add to the learning rate eta
    M is the size of the mini-batch used in each iteration
    epochs is number of epochs
    '''
    
    z = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) 

    print(np.shape(X_train))
    b_shape = np.shape(X_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    n = X_train.shape[0]  # Number of samples
    #beta_list = list()

    change = 0
    momentum = 0.9 #0.9   
    Giter = 0
    t = 0
    mom1 = 0
    mom2 = 0


    # Hessian matrix
    H = (2.0/n)* X.T @ X 
    # Get the eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)
    
    # Attempting to find an optimal learning rate
    eta = rate*(1.0/np.max(EigValues))  # learning rate   #IDEAL rate = 3, or 1
    
    sched = None
    if method == 'basic':
        sched = Scheduler.Momentum(eta, momentum)
    elif method == 'adagrad':
        sched = Scheduler.AdagradMomentum(eta, momentum)
    elif method == 'rmsprop':
        sched = Scheduler.RMS_propMomentum(eta, momentum)
    elif method == 'adam':
        sched = Scheduler.AdamMomentum(eta, 0.9, 0.999, momentum)
    
    for i in range(iterations):
        #gradient =  grad(logreg_cost)(n, beta, X_train, z_train, lamba) # -X_batch.T @ (z_batch -( 1.0 / (1 + np.exp(-z)))) # CHECK SIGMOID FUNC

        gradient = -X_train.T @ (z_train -( 1.0 / (1 + np.exp(-z_train)))) + 2*lamba*beta # CAN USE beta.T

        save_iter = i

        beta -= sched.update_change(gradient)
       
    predict_test = X_test.dot(beta)
    predict = X.dot(beta)

    mse_val = mse(z_test, predict_test)
    
    info = f'Method {method} \n iterations = {save_iter}', f'momentum = {momentum}', f'learning rate = {eta}', f'mse= {mse_val}'
    
    print(f'MSE for stochastic gradient descent with batches is {mse_val} \n')
    print(f'{info}\n')
    return predict, beta, mse_val, info



if __name__=="__main__":
    X,y = load_breast_cancer(return_X_y=True)
    print(np.shape(X))

    predict, beta, mse_val, info = logreg_gd(X, y, method='adagrad', lamba=30, iterations=100000, rate=60)


"""
Best so far:
('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.20014116045837e-05', 'mse= 43076.029628381344'), rate = 50

('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.6801976246417182e-05', 'mse= 42441.838602851516'), rate = 70

('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.440169392550044e-05', 'mse= 41388.257435600455'), rate = 60   #BEST LEARNING RATE

"""