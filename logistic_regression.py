import autograd.numpy as anp
import numpy as np
import warnings

from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import Scheduler
from helper_functions import accuracy, plot_heatmap

''' Writing Logistic regression code which will be used to compare with FFNN '''

"""
- Define your cost function before you start writing your code.
- Define your design matrix before you start writing your code.

Can I skip the cost function definition?
"""



def learning_schedule(t, t0, t1):
    return t0/(t+t1)

def logreg_cost(n, beta, X, z, L):
    # res = z - ( 1.0 / (1 + np.exp(-z) ) ) 
    # cost = -(1/n)*res.T@res + L*(beta.T).dot(beta) 
    
    cost = -np.sum(z*(beta@X) - np.log(1 + np.exp(beta@X)))
    return cost

def logreg_gd(X_train, X_test, z_train, z_test, eta, method='adagrad', lamba=2, iterations=100):
    '''
    Function which performs Stochastic Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 100 but can be changed, the same as scikit-learn's LogisticRegression
    lamba is the regularization parameter lambda used in the cross entropy cost function
        -> We have chosen to use the analytical gradient, thus you will not find a cost function with the lamba parameter

    rate is a factor you can add to the learning rate eta
    M is the size of the mini-batch used in each iteration
    epochs is number of epochs
    '''
    
    b_shape = np.shape(X_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    change = 0
    momentum = 0.9 #0.9   
    
    sched = None
    if method == 'basic':
        sched = Scheduler.Momentum(eta, momentum)
    elif method == 'adagrad':
        sched = Scheduler.AdagradMomentum(eta, momentum)
    elif method == 'rmsprop':
        sched = Scheduler.RMS_propMomentum(eta, 0.9, momentum) # Same beta as beta1 in AdamMomentum
    elif method == 'adam':
        sched = Scheduler.AdamMomentum(eta, 0.9, 0.999, momentum)
    
    for i in range(iterations):
        #gradient =  grad(logreg_cost)(n, beta, X_train, z_train, lamba) # -X_batch.T @ (z_batch -( 1.0 / (1 + np.exp(-z)))) # CHECK SIGMOID FUNC

        gradient = -X_train.T @ (z_train -( 1.0 / (1 + np.exp(-z_train)))) + 2*lamba*beta # CAN USE beta.T

        save_iter = i

        beta -= sched.update_change(gradient)
       
    predict_test = X_test.dot(beta)
    predict_test = np.where(predict_test > 0.5, 1, 0)
    # predict was not used
    # predict = X.dot(beta)
    # predict = np.where(predict > 0.5, 1, 0)

    acc_score = accuracy(z_test, predict_test)
    
    info = f'Method {method}:\niterations = {save_iter} momentum = {momentum} learning rate = {eta} acc_score= {acc_score}'
    
    # print(f'Accuracy for gradient descent with is {acc_score}\n')
    # print(f'{info}\n')
    return acc_score, info, beta

def train_pred_logistic_regression(
        X_train: np.ndarray,
        X_test: np.ndarray,
        z_train: np.ndarray,
        z_test: np.ndarray,
        eta_vals: np.ndarray,
        lmbda_vals: np.ndarray,
        method = "adagrad",
        max_iter = 100
    ):
    acc_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
 
    # Do grid search of eta and lambda values
    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            acc_score, info, _ = logreg_gd(X_train, X_test, z_train, z_test, eta=eta_vals[i], lamba=lmbda_vals[j], method=method, iterations=max_iter)
            acc_vals[i][j] = acc_score
    return acc_vals

def train_pred_logistic_regression_skl(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        eta_vals: np.ndarray,
        lmbda_vals: np.ndarray,
        loss = "log_loss",
        max_iter = 100
    ):
    '''Trains a scikit-learn logistic regression network for a grid of different learning rates (eta) and regularisation parameters (lmbda).
    Returns the accuarcy score for each combination of eta and lmbda.'''
    acc_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
 
    # Do grid search of eta and lambda values
    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            network = SGDClassifier(loss = loss, learning_rate = "constant", alpha = lmbda_vals[j], eta0 = eta_vals[i], max_iter = max_iter)
 
            # If it does not converge ignore it, the accuarcy score for that will be None or zero 
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        "ignore", category=ConvergenceWarning, module="sklearn"
                        )
                network.fit(x_train, y_train.flatten())
            acc_vals[i][j] = network.score(x_test, y_test)
    return acc_vals


def main():
    # To run scikit learns logistic regression with SGD
    run_sklearn = True


    ### DATA SETUP ###
    X, y = load_breast_cancer(return_X_y = True)
    y_fit = y.ravel()
    X_train, X_test, z_train, z_test = train_test_split(X, y_fit, test_size=0.2) 

    # Normalise data
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)

    X_train_norm = (X_train-X_mean)/X_std
    X_test_norm = (X_test-X_mean)/X_std


    ### LOGISTIC REGRESSION ###
    eta_vals = anp.logspace(-3, -2, 3)
    lmbda_vals = anp.logspace(-5, 0, 6)
    grad_methods = ["basic", "adagrad", "rmsprop", "adam"]

    root_path = Path.cwd()
    problem = "logistic_regression"

    print("--- LOGISTIC REGRESSION ON BREAST CANCER CLASSIFICATION ---")

    if run_sklearn:
        print("\nRunning sklearn logistic regression with stg")
        file_path_sklearn = root_path / "plots" / problem
        file_path_sklearn.mkdir(parents=True, exist_ok=True)

        acc_score = train_pred_logistic_regression_skl(X_train, X_test, z_train, z_test, eta_vals, lmbda_vals)
        plot_heatmap(acc_score, file_path_sklearn / "logistic_reg_skl.pdf", r"$\eta$", r"$\lambda", eta_vals, lmbda_vals)

    for method in grad_methods:
        print(f"\nRunning our logistic regression with gd, method={method}")
        file_path = root_path / "plots" / problem / method
        file_path.mkdir(parents=True, exist_ok=True)

        acc_score = train_pred_logistic_regression(X_train, X_test, z_train, z_test, eta_vals, lmbda_vals, method=method)
        plot_heatmap(acc_score, file_path / "logistic_reg.pdf", r"$\eta$", r"$\lambda", eta_vals, lmbda_vals)

if __name__=="__main__":
    main()

"""
Best so far:
('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.20014116045837e-05', 'mse= 43076.029628381344'), rate = 50

('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.6801976246417182e-05', 'mse= 42441.838602851516'), rate = 70

('Method adagrad \n iterations = 99999', 'momentum = 0.9', 'learning rate = 1.440169392550044e-05', 'mse= 41388.257435600455'), rate = 60   #BEST LEARNING RATE

"""
