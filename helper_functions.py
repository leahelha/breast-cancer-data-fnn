import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from typing import Any, Literal

from FFNN import FFNN
from Scheduler import Scheduler

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
    
    mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

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
        hidden_layers: tuple(int),
        activation: Literal['relu', 'identity', 'logistic', 'tanh'],
        solver: Literal['lbfgs', 'sgd', 'adam'],
        batches: int,
        epochs: int,
        seed: int = 10
    ):
    
    mse_vals = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            network = MLPRegressor(hidden_layers, activation, solver, lmbda_vals[j], eta_vals[i], epochs, batch_size = x_train.shape[0]//batches, random_state = seed)
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