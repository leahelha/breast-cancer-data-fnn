from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from tqdm import trange
from pathlib import Path
from typing import Callable
# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from tabulate import tabulate
from rounders import round_to_figures
from matplotlib.colors import BoundaryNorm
from autograd import grad, elementwise_grad
import autograd.numpy as anp
"""
This task is done in collaboration, by Ida Monsen, Vetle Henrik Hvoslef and Leah Hansen

"""
np.random.seed(11)
seed(11)

class regression_class:
    '''Does OLS, Ridge and Lasso regression with a polynomial model of degree up to n_deg_max.
    
    If you want to access items from the class after doing regression, results are stored in the following way:
    - Dictionaries ols, ridge and lasso with keys "beta", "mse_train", "mse_test", "r2_train", "r2_test".
    - dict[key][i] gives the beta coefficients/MSE/R^2 for polynomial degree i+1.
    - For Ridge and Lasso this gives a list with values for each lambda, so
    dict[key][i][j] gives the beta coefficients/MSE/R^2 for polynomial degree i+1 and lmbda[j].'''

    def __init__(self, x, y, n_deg_max, lmbda):
        self.x = x
        self.y = y
        self.n_deg_max = n_deg_max
        self.lmbda = lmbda
        self.root_path = Path.cwd()
        self.sigfig = 5 # Amount of significant figures in the Latex tables

        # Initialise dictionaries to store results
        keys = ["beta", "mse_train", "mse_test", "r2_train", "r2_test", "mse_kfold"]
        self.ols = dict.fromkeys(keys)
        self.ridge = dict.fromkeys(keys)
        self.lasso = dict.fromkeys(keys)
        for key in self.ols.keys():
            self.ols[key] = [0]*self.n_deg_max
            self.ridge[key] = [0]*self.n_deg_max
            self.lasso[key] = [0]*self.n_deg_max
        
        self.make_design_matrix()
        self.normalise_design_matrix()

    def make_design_matrix(self):
        '''Makes design matrix and splits into training and test data'''
        self.X = PolynomialFeatures(self.n_deg_max, include_bias = False).fit_transform(self.x) # without intercept
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls
        return self.X, self.X_train, self.X_test

    def normalise_design_matrix(self):
        '''Normalise data by subtracting mean and dividing by standard deviation'''
        self.X_mean = np.mean(self.X_train, axis = 0)
        self.y_mean = np.mean(self.y_train)
        self.X_std = np.std(self.X_train, axis = 0)
        self.y_std = np.std(self.y_train)

        self.X_train_scaled = (self.X_train - self.X_mean)/self.X_std
        self.y_train_scaled = (self.y_train - self.y_mean)/self.y_std
        self.X_test_scaled = (self.X_test - self.X_mean)/self.X_std
        self.y_test_scaled = (self.y_test - self.y_mean)/self.y_std

        #*** L
        self.X_scaled = (self.X - self.X_mean)/self.X_std

        return self.X_scaled, self.X_train_scaled, self.X_test_scaled

    def predict_ols(self, pol_degree):
        '''Makes a prediction with OLS model of polynomial degree pol_deg, using all X data'''
        # Pick out relevant part of design matrix for this pol_degree
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X = ((self.X - self.X_mean)/self.X_std)[:, 0:N]
        
        prediction = X @ self.ols["beta"][pol_degree-1]*self.y_std + self.y_mean
        return prediction

    def predict_ridge(self, pol_degree, lmbda_n):
        '''Makes a prediction with Ridge model of polynomial degree pol_deg, using all X data'''
        # Pick out relevant part of design matrix for this pol_degree
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X = ((self.X - self.X_mean)/self.X_std)[:, 0:N]

        prediction = X @ self.ridge["beta"][pol_degree-1][lmbda_n]*self.y_std + self.y_mean
        return prediction

    def predict_lasso(self, pol_degree, lmbda_n):
        '''Makes a prediction with Lasso model of polynomial degree pol_deg, using all X data'''
        # Pick out relevant part of design matrix for this pol_degree
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X = ((self.X - self.X_mean)/self.X_std)[:, 0:N]

        prediction = X @ self.lasso["beta"][pol_degree-1][lmbda_n]*self.y_std + self.y_mean
        return prediction
    
    def beta_ols(self, X, y):
        '''Given the design matrix X and the output y, calculates the coefficients beta using OLS.'''
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        return beta

    def beta_ridge(self, X, y, lmbda):
        '''Given the design matrix X, the output y and the parameter lmbda, calculates the coefficients beta using OLS.'''
        n = np.shape(X)[1]
        beta = np.linalg.pinv(X.T @ X + lmbda*np.eye(n)) @ X.T @ y
        return beta

    def mse_own(self, y_tilde, y):
        '''Calculates the mean square error of a prediction y_tilde.'''
        mse = 1/len(y) * np.sum((y-y_tilde)**2)
        return mse

    def r2_own(self, y_tilde, y):
        '''Calculates the R^2 score of a prediction y_tilde.'''
        a = np.sum((y-y_tilde)**2)
        b = np.sum((y-np.mean(y))**2)
        return 1 - a/b

    def fit_predict_ols(self, pol_degree):
        '''For a given polynomial order, makes and trains an OLS model and calculates MSE for both training and test data.'''
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]
        
        # Fit parametres
        beta = self.beta_ols(X_train_scaled, self.y_train_scaled)

        # Make predictions
        # y_train_pred = X_train_scaled @ beta * self.y_std + self.y_mean
        # y_test_pred = X_test_scaled @ beta * self.y_std + self.y_mean

        y_train_pred = X_train_scaled @ beta
        y_test_pred = X_test_scaled @ beta

        # Calculate MSE and R^2 for both training and test data
        mse_train = self.mse_own(y_train_pred, self.y_train_scaled)
        mse_test = self.mse_own(y_test_pred, self.y_test_scaled)
        r2_train = self.r2_own(y_train_pred, self.y_train_scaled)
        r2_test = self.r2_own(y_test_pred, self.y_test_scaled)

        return beta, mse_train, mse_test, r2_train, r2_test
    
    def fit_predict_ridge(self, pol_degree):
        '''For a given polynomial order, makes and trains a Ridge regression model and calculates MSE for both training and test data.'''
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]

        beta = [0] * len(self.lmbda)
        mse_train = np.zeros_like(self.lmbda)
        mse_test = np.zeros_like(self.lmbda)
        r2_train = np.zeros_like(self.lmbda)
        r2_test = np.zeros_like(self.lmbda)

        for i in range(len(self.lmbda)):
            # Fit parametres
            beta[i] = self.beta_ridge(X_train_scaled, self.y_train_scaled, self.lmbda[i])

            # Make predictions
            y_train_pred = X_train_scaled @ beta[i]
            y_test_pred = X_test_scaled @ beta[i]

            # y_train_pred = X_train_scaled @ beta * self.y_std + self.y_mean
            # y_test_pred = X_test_scaled @ beta * self.y_std + self.y_mean

            # Calculate MSE and R^2 for both training and test data
            mse_train[i] = self.mse_own(y_train_pred, self.y_train_scaled)
            mse_test[i] = self.mse_own(y_test_pred, self.y_test_scaled)
            r2_train[i] = self.r2_own(y_train_pred, self.y_train_scaled)
            r2_test[i] = self.r2_own(y_test_pred, self.y_test_scaled)

        return beta, mse_train, mse_test, r2_train, r2_test
    
    def fit_predict_lasso(self, pol_degree):
        '''For a given polynomial order, makes and trains a Lasso regression model and calculates MSE for both training and test data.'''
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]

        beta = [0] * len(self.lmbda)
        mse_train = np.zeros_like(self.lmbda)
        mse_test = np.zeros_like(self.lmbda)
        r2_train = np.zeros_like(self.lmbda)
        r2_test = np.zeros_like(self.lmbda)

        for i in range(len(self.lmbda)):
            # Fit parametres
            model = Lasso(self.lmbda[i], max_iter = 5000, tol = 1e-2).fit(X_train_scaled, self.y_train_scaled)
            beta[i] = model.coef_

            # Make predictions
            y_train_pred = X_train_scaled @ beta[i]
            y_test_pred = X_test_scaled @ beta[i]

            # y_train_pred = X_train_scaled @ beta * self.y_std + self.y_mean
            # y_test_pred = X_test_scaled @ beta * self.y_std + self.y_mean

            # Calculate MSE and R^2 for both training and test data
            mse_train[i] = self.mse_own(y_train_pred, self.y_train_scaled)
            mse_test[i] = self.mse_own(y_test_pred, self.y_test_scaled)
            r2_train[i] = self.r2_own(y_train_pred, self.y_train_scaled)
            r2_test[i] = self.r2_own(y_test_pred, self.y_test_scaled)

        return beta, mse_train, mse_test, r2_train, r2_test



    def kFold_linreg(self, pol_degree, lin_model, k = 5, lmbda = None):
        '''Calculate the kfold cross validation for a specific polynomial degree, pol_degree, and a specific number of folds, k.'''
        poly = PolynomialFeatures(pol_degree, include_bias = False) # Skal være False hvis sentrerer
        if lmbda is None:
            model = lin_model(fit_intercept = False) # Forventer sentrert data
        else:
            model = lin_model(alpha = lmbda, fit_intercept = False) # Forventer sentrert data
        x = self.x
        y = self.y

        indicies = np.arange(len(x))
        np.random.shuffle(indicies)

        x_shuffled = x[indicies]
        y_shuffled = y[indicies]

        # Initialize a KFold instance:
        kfold = KFold(n_splits = k)
        scores_KFold = np.zeros(k)

        # Perform the cross-validation to estimate MSE:
        for i, (train_inds, test_inds) in enumerate(kfold.split(x_shuffled)):
            x_train = x_shuffled[train_inds]
            y_train = y_shuffled[train_inds]

            x_test = x_shuffled[test_inds]
            y_test = y_shuffled[test_inds]

            # Train: Standardize and design matrix
            X_train = poly.fit_transform(x_train)

            X_train_scalar = np.mean(X_train, axis = 0)
            X_train_std = np.std(X_train, axis = 0)
            y_train_scalar = np.mean(y_train)
            y_train_std = np.std(y_train)

            X_standardize_train = (X_train - X_train_scalar) / X_train_std
            y_standardize_train = (y_train - y_train_scalar) / y_train_std

            # Test: Standardize and design matrix
            X_test = poly.fit_transform(x_test)

            X_standardize_test = (X_test - X_train_scalar) / X_train_std # Trent på trenings skaleringen
            y_standardize_test = (y_test - y_train_scalar) / y_train_std

            # Fitting on train data, and predicting on test data:
            model.fit(X_standardize_train, y_standardize_train)
            y_standardize_pred = model.predict(X_standardize_test)
            
            # Scores: mse
            scores_KFold[i] = np.sum((y_standardize_pred - y_standardize_test)**2)/np.size(y_standardize_pred)      

        scores_KFold_mean = np.mean(scores_KFold)
        return scores_KFold_mean

    def ols_kfold(self):
        '''Calculates kfold with OLS for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            ols_score = self.kFold_linreg(i + 1, LinearRegression)
            self.ols["mse_kfold"][i] = ols_score
    
    def ridge_kfold(self):
        '''Calculates kfold with Ridge regression for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            ridge_score = [0]*len(self.lmbda)
            for j in range(len(self.lmbda)):
                ridge_score[j] = self.kFold_linreg(i + 1, Ridge, lmbda=self.lmbda[j])
            self.ridge["mse_kfold"][i] = ridge_score
    
    def lasso_kfold(self):
        '''Calculates kfold with Lasso regression for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            lasso_score = [0]*len(self.lmbda)
            for j in range(len(self.lmbda)):
                lasso_score[j] = self.kFold_linreg(i + 1, Lasso, lmbda=self.lmbda[j])
            self.lasso["mse_kfold"][i] = lasso_score

    def ols_regression(self):
        '''Calculates OLS for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            ols_results = self.fit_predict_ols(i+1)
            self.ols["beta"][i] = ols_results[0]
            self.ols["mse_train"][i] = ols_results[1]
            self.ols["mse_test"][i] = ols_results[2]
            self.ols["r2_train"][i] = ols_results[3]
            self.ols["r2_test"][i] = ols_results[4]

    def ridge_regression(self):
        '''Calculates Ridge regression for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            ridge_results = self.fit_predict_ridge(i+1)
            self.ridge["beta"][i] = ridge_results[0]
            self.ridge["mse_train"][i] = ridge_results[1]
            self.ridge["mse_test"][i] = ridge_results[2]
            self.ridge["r2_train"][i] = ridge_results[3]
            self.ridge["r2_test"][i] = ridge_results[4]
    
    def lasso_regression(self):
        '''Calculates Lasso regression for polynomials of degree 1 to n_deg_max.'''
        for i in trange(self.n_deg_max):
            lasso_results = self.fit_predict_lasso(i+1)
            self.lasso["beta"][i] = lasso_results[0]
            self.lasso["mse_train"][i] = lasso_results[1]
            self.lasso["mse_test"][i] = lasso_results[2]
            self.lasso["r2_train"][i] = lasso_results[3]
            self.lasso["r2_test"][i] = lasso_results[4]
    
    def find_optimal_lambda(self, type):
        '''For either Ridge or Lasso regression, finds and returns lambda value that gives lowest MSE_test and corresponding MSE_test for each polynomial degree.'''
        if type == "ridge":
            mse_values = self.ridge["mse_test"]
        elif type == "lasso":
            mse_values = self.lasso["mse_test"]
        else:
            raise ValueError("Must specify 'ridge' or 'lasso' when calling find_optimal_lambda.")
            
        # List to store optimal lambda and corresponding MSE for each polynomial degree
        optimaL_values = [0]*self.n_deg_max

        for i in range(self.n_deg_max): # for each polynomial degree
            min_index = 0
            min_el = mse_values[i][0]

            # Find lowest MSE and corresponding lambda
            for j in range(len(self.lmbda)):
                if (mse_values[i][j] < min_el):
                    min_el = mse_values[i][j]
                    min_index = j
            optimaL_values[i] = (self.lmbda[min_index], min_el)
        return optimaL_values
    
    def find_optimal_lambda_kfold(self, type):
        '''For either Ridge or Lasso regression, finds and returns lambda value and MSE_score, that gives lowest MSE_score for each polynomial degree.'''
        if type == "ridge":
            mse_kfold = self.ridge["mse_kfold"]
        elif type == "lasso":
            mse_kfold = self.lasso["mse_kfold"]
        else:
            raise ValueError("Must specify 'ridge' or 'lasso' when calling find_optimal_lambda.")
        
        # List to store optimal lambda and corresponding MSE for each polynomial degree
        optimaL_values = [0]*self.n_deg_max
        
        for i in range(self.n_deg_max): # for each polynomial degree
            min_index = 0
            min_el = mse_kfold[i][0]

            # Find lowest MSE and corresponding lambda
            for j in range(len(self.lmbda)):
                if (mse_kfold[i][j] < min_el):
                    min_el = mse_kfold[i][j]
                    min_index = j
            optimaL_values[i] = (self.lmbda[min_index], min_el)
        return optimaL_values

    def savefig(self, fname, folder = None, fig = None):
        if folder is None:
            folder = "plots"

        root_path = self.root_path
        abs_path = root_path / folder
        abs_path.mkdir(exist_ok=True)

        filename = abs_path / fname
        if fig is None:
            plt.savefig(filename)
        else:
            fig.savefig(filename)



    def plot_ols(self, train_results, test_results, ylabel, name):
        '''Plots either MSE or R2 score for train and test data from OLS and saves to file.'''
        plt.figure(figsize = (6,4))
        plt.plot(range(1, len(train_results)+1), train_results, label = "Training data")
        plt.plot(range(1, len(train_results)+1), test_results, label = "Test data")
        plt.xlabel("Polynomial degree")
        plt.ylabel(ylabel)
        plt.legend()
        self.savefig(f"{name}.pdf")

    def plot_ridge_or_lasso(self, train_results, test_results, ylabel, name):
        '''Plots either MSE or R2 score for train and test data from Ridge or Lasso regression and saves to file.'''    
        for i in range(self.n_deg_max):
            plt.figure()
            plt.semilogx(self.lmbda, train_results[i], label = "Training data")
            plt.semilogx(self.lmbda, test_results[i], label = "Test data")

            plt.text(1, 1, f"Polynomial of order {i+1}")
            plt.xlabel("$\lambda$")
            plt.ylabel(ylabel)
            plt.legend()
            self.savefig(f"{name}_{i+1}.pdf")
            plt.close()

    def plot_beta_ols(self, beta, name, degrees = None):
        '''Plots beta values with standard deviation from OLS regression.'''
        if degrees is None:
            n_deg_max = self.n_deg_max
            degrees = range(1, n_deg_max + 1, 2)

        plt.figure(figsize = (15,5))
        plt.title(f"{name}")
        for degree in degrees:
            indicies = range(len(beta[degree - 1]))
            degree_std = np.std(beta[degree - 1])
            plt.bar(indicies, beta[degree - 1], label = f"degree = {degree}, std = {degree_std}")
        plt.legend()
        self.savefig(f"{name}.pdf")
        # plt.show()
              
    def plot_beta_ridge_or_lasso(self, beta, lmbda, name, degrees = None):
        '''Plots beta values with standard deviation from Ridge or Lasso regression.'''
        if degrees is None:
            n_deg_max = self.n_deg_max
            degrees = range(1, n_deg_max + 1, 2)

        for j in trange(len(lmbda)):
            plt.figure(figsize = (15,5))
            plt.title(f"{name}, with lambda = {lmbda[j]}")
            for degree in degrees:
                indicies = range(len(beta[degree - 1][j]))
                degree_std = np.std(beta[degree - 1][j])
                plt.bar(indicies, beta[degree - 1][j], label = f"degree = {degree}, std = {degree_std}")
            plt.legend()
            self.savefig(f"{name}_{lmbda[j]}.pdf")
        # plt.show()

    def plot_ols_results(self, degrees = None):
        '''Plots MSE, R2 score and beta values for OLS regression and saves to file.'''
        self.plot_ols(self.ols["mse_train"], self.ols["mse_test"], "Mean Squared Error", "mse_ols")
        self.plot_ols(self.ols["r2_train"], self.ols["r2_test"], f"$R^2$", "r2_ols")
        self.plot_beta_ols(self.ols["beta"], "beta_ols", degrees = None)
    
    def plot_ridge_results(self, degrees = None):
        '''Plots MSE, R2 score and beta values for Ridge regression and saves to file.'''
        self.plot_ridge_or_lasso(self.ridge["mse_train"], self.ridge["mse_test"], "Mean Squared Error", "mse_ridge")
        self.plot_ridge_or_lasso(self.ridge["r2_train"], self.ridge["r2_test"], f"$R^2$", "r2_ridge")
        self.plot_beta_ridge_or_lasso(self.ridge["beta"], self.lmbda, "beta_ridge", degrees = None)

    def plot_lasso_results(self, degrees = None):
        '''Plots MSE, R2 score and beta values for Lasso regression and saves to file.'''
        self.plot_ridge_or_lasso(self.lasso["mse_train"], self.lasso["mse_test"], "Mean Squared Error", "mse_lasso")
        self.plot_ridge_or_lasso(self.lasso["r2_train"], self.lasso["r2_test"], f"$R^2$", "r2_lasso")
        self.plot_beta_ridge_or_lasso(self.lasso["beta"], self.lmbda, "beta_lasso", degrees = None)

    def create_latex_table(self, optimal_values, name, type, kfold = None):
        '''Creates a latex table from the optimal values from the regclass code.'''
        if kfold is True:
            headers = ["degree", "lambda", "mse score"]
        elif kfold is False:
            headers = ["degree", "lambda", "mse test"]
        else:
            raise ValueError("Must specify a value for the bool kfold, not {kfold}")

        output = []
        for i in range(len(optimal_values)):
            pol_degree = i + 1
            lmbda = optimal_values[i][0]
            mse_value = round_to_figures(optimal_values[i][1], self.sigfig)

            output.append([pol_degree, lmbda, mse_value])
        
        table = tabulate(output, headers=headers, tablefmt = "latex")

        root_path = self.root_path
        abs_path = root_path / "tables"
        abs_path.mkdir(exist_ok=True)

        fname = f"{name}_{type}.tex"
        filename = abs_path / fname
        with open(filename, "w") as outfile:
            outfile.write(table)


    def create_latex_table_ols(self, name, kfold = None):
        if kfold is True:
            headers = ["degree", "mse score"]
            mse_key = "mse_kfold"
        elif kfold is False:
            headers = ["degree", "mse test"]
            mse_key = "mse_test"
        else:
            raise ValueError("Must specify a value for the bool kfold, not {kfold}")

        output = []
        mse_values = self.ols[mse_key]
        for i in range(len(mse_values)):
            pol_degree = i + 1
            mse_value = round_to_figures(mse_values[i], self.sigfig)
            output.append([pol_degree, mse_value])
        
        table = tabulate(output, headers=headers, tablefmt = "latex")

        root_path = self.root_path
        abs_path = root_path / "tables"
        abs_path.mkdir(exist_ok=True)

        fname = f"{name}.tex"
        filename = abs_path / fname
        with open(filename, "w") as outfile:
            outfile.write(table)

    
    def create_tables(self):
        optimal_values_rigde = self.find_optimal_lambda(type = "ridge")
        optimal_values_lasso = self.find_optimal_lambda(type = "lasso")

        optimal_values_kfold_rigde = self.find_optimal_lambda_kfold(type = "ridge")
        optimal_values_kfold_lasso = self.find_optimal_lambda_kfold(type = "lasso")

        self.create_latex_table(optimal_values_rigde, "optimal_lambda_values", type = "rigde", kfold = False)
        self.create_latex_table(optimal_values_lasso, "optimal_lambda_values", type = "lasso", kfold = False)
        
        self.create_latex_table(optimal_values_kfold_rigde, "optimal_lambda_values_kfold", type = "rigde", kfold = True)
        self.create_latex_table(optimal_values_kfold_lasso, "optimal_lambda_values_kfold", type = "lasso", kfold = True)


        self.create_latex_table_ols("mse_test_values_ols", kfold = False)
        self.create_latex_table_ols("mse_score_values_ols_kfold", kfold = True)

"""
Testing out the Gradient descent method on a simple function
"""
#Creating a simple function to test the Gradient Descent method on

def function(x):
    """
    This function returns the value of f(x)
    """
    f = 3 + 4*x + 5*x**2
    return f

def grads(x):
    """
    This function returns the value of the derivative df(x)/dx, here called df
    """
    df = 4 + 10*x
    return df

def GradientDescent(gradient, iterations, learning_rate, guess_x=0):
    """
    This function performs gradient descent. Use this on an analytical function of 
    which you know the derivative. Iterations and learning rate need to be specified.  
    The default value for the initial guess, guess_x, is 0.
    """
    x = guess_x
    # Perform gradient descent
    for i in range(iterations):
        grad = grads(x)
        x = x - learning_rate * grad
    
    #returns at which x the function is at its minimum
    return x

# Print the result
min_x = GradientDescent(grad, 100, 0.1, guess_x=0)
min_f = function(min_x)

# print(f"Minimum value of f is {min_f} at x = {min_x}")

x_array = np.linspace(-10, 10, 100)
plot_f = function(x_array)
grad_f = grads(x_array)

# plt.plot(x_array, plot_f, label="function")
# plt.plot(x_array, grad_f, label="gradient")
# plt.plot(min_x, min_f, 'r*')
# plt.xlabel("x")
# plt.ylabel("function")
# plt.legend()
# plt.show()


"""
Now attempting gradient descent for the OLS method
"""
# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

def FrankeFunction(x,y):
    '''Calculates the two-dimensional Franke's function.'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# the number of datapoints
n = 100
x = np.random.rand(n,1)
y = np.random.rand(n,1)

# Grid
x_, y_ = np.meshgrid(x, y)
xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix

# True model 
z = FrankeFunction(x_, y_)

# Making our design matrix using Project 1 class
n_deg_max = 12 # max polynomial degree
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression
model = regression_class(xy, z.flatten(), n_deg_max, lmbda)

X = model.make_design_matrix()[0]  #Making design matrix, is not normalized



b_shape = np.shape(X)[1]
# Initiating a random beta
beta = np.random.randn(b_shape,) #*** did this to match reg_class beta



def AdaGrad(change, gradient, eta, Giter, delta=1e-8, momentum=0):
    ''' 
    Using AdaGrad to update the gradient, this function is to be 
    called inside the Gradient Descent functions 
    '''
    update_gradient = (gradient*eta/(delta+np.sqrt(Giter))) + momentum*change

    return update_gradient, Giter

def RMSprop(change, gradient, eta, Giter, beta=0.9, delta=1e-8, momentum=0):
    ''' 
    Using RMSprop to update the gradient, this function is to be 
    called inside the Gradient Descent functions 
    '''
    Giter = beta * Giter + (1 - beta) * (gradient ** 2)
    update_gradient = ((eta* gradient) / (np.sqrt(Giter) + delta)) + momentum * change
    return update_gradient, Giter

def ADAM(change, gradient, eta, mom1, mom2, t, beta1=0.9, beta2=0.999, delta=1e-8, momentum=0):
    ''' 
    Using ADAM to update the gradient, this function is to be 
    called inside the Gradient Descent functions 
    '''
    mom1 = beta1 * mom1 + (1 - beta1) * gradient  # Update 1st moment vector
    mom2 = beta2 * mom2 + (1 - beta2) * (gradient ** 2)  # Update 2nd moment vector
    mom1_corrected = mom1 / (1 - beta1 ** t)  # Bias-corrected 1st moment estimate
    mom2_corrected = mom2 / (1 - beta2 ** t)  # Bias-corrected 2nd moment estimate

    update_gradient = (eta / (np.sqrt(mom2_corrected) + delta)) * mom1_corrected + momentum * change
    
    return update_gradient, mom1, mom2

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


def gd_momentum(beta, X, z, method='adagrad', iterations=1000, rate=1, Auto=False):   #beta ***
    '''
    Function which performs Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    iterations has a default of 1000 but can be changed
    rate is a factor you can add to the learning rate eta
    '''
    # b_shape = np.shape(X)[1]
    # # Initiating a random beta
    # beta = np.random.randn(b_shape,)
    z = np.ravel(z)

    n = X.shape[0]  # Number of samples
    #beta_list = list()

    change = 0
    momentum = 0.9    #IDEAL MOMENTUM
    # Hessian matrix
    H = (2.0/n)* X.T @ X
    # Get the eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)
    
    # Attempting to find an optimal learning rate
    eta = rate*(1.0/np.max(EigValues))  # learning rate   #IDEAL rate = 3, or 1

    mom1 = 0 
    mom2 = 0
    t = 0
    
    Giter = 0
    


    for i in range(iterations):
        
        gradient = gradients(beta, n, X, z, lamba=0, Auto=Auto)
        

        """
        gradient given by: 
            (20,)
        """

        #beta_list.append(beta)

        if method == 'adagrad':
            Giter += gradient * gradient
            change, Giter = AdaGrad(change, gradient, eta, Giter, delta=1e-8, momentum=momentum)
            
            beta -= change

        elif method == 'rmsprop':
            change, Giter = RMSprop(change, gradient, eta, Giter, beta=0.9, delta=1e-8, momentum=0)
            beta -= change

        else:  # method == 'adam'
            t += 1
            change, mom1, mom2 = ADAM(change, gradient, eta, mom1, mom2, t, beta1=0.9, beta2=0.999, delta=1e-8, momentum=0)
            beta -= change

        save_iter = i
        if np.linalg.norm(change) < 1e-3:
            break
        
    
    
    predict = X.dot(beta)
    
    mse = 1/(n*n) * np.sum((z-predict)**2)
    info = f'Method {method} \n iterations = {save_iter}', f'momentum = {momentum}', f'learning rate = {eta}', f'mse= {mse}'
    
    print('For Gradient Descent\n')
    print(f'{info}\n')
    return predict, beta, mse, info



''' Now implementing Stochastic Gradient Descent '''
def learning_schedule(t, t0, t1):
    return t0/(t+t1)


def sgd_momentum(X, z, method='adagrad', M=32, epochs=1, Auto=False):   #*** Epochs
    '''
    Function which performs Stochastic Gradient Descent with momentum
    beta is the beta parameter
    X is the design matrix
    z is the target data
    
    M is the size of the mini-batch used in each iteration
    epochs is number of epochs
    '''
    print('X shape and z shape')
    print(np.shape(X))
    print(np.shape(z))
    z = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) 

    b_shape = np.shape(X_train)[1]
    # Initiating a random beta
    beta = np.random.randn(b_shape,)

    print('shape is')
    print(np.shape(beta))
    beta_list = list()

    change = 0
    momentum = 0.8  # IDEAL MOMENTUM
    n = X_train.shape[0]  # Number of samples
    m = int(n/M) #number of mini-batches
   
    t0, t1 = 80, 50  #scheduling constants for learning rate
    t = 0 
    mom1 = 0
    mom2 = 0
    for e in range(epochs):    
        Giter = 0

        for i in range(m):
            indices = np.random.choice(n, size=M, replace=False)  # Randomly select indices for the batch

            X_batch = X_train[indices]
            z_batch = z_train[indices] 
            
            gradient = gradients(beta, M, X_batch, z_batch, lamba=0, Auto=Auto)
            

            beta_list.append(beta)

            eta = learning_schedule(epochs*m+i, t0, t1)

            if method == 'adagrad':
                Giter += gradient * gradient
                change, Giter = AdaGrad(change, gradient, eta, Giter, delta=1e-8, momentum=momentum)

                #change = eta*gradient + momentum*change
                beta -= change
                
            elif method == 'rmsprop':
                change, Giter = RMSprop(change, gradient, eta, Giter, beta=0.9, delta=1e-8, momentum=0)
                beta -= change
                    
            elif method == 'adam':
                t += 1
                change, mom1, mom2 = ADAM(change, gradient, eta, mom1, mom2, t, beta1=0.9, beta2=0.999, delta=1e-8, momentum=0)
             
                beta -= change
        
        save_e = e

    
    predict_test = X_test.dot(beta)
    predict = X.dot(beta)

    mse = np.mean( (np.ravel(z_test)-predict_test)**2)#1/(n*n) * np.sum(  (np.ravel(z_test)-predict_test)**2)
    abs_error_avg= 1/(n*n)*np.sum(np.abs(np.ravel(z_test)-predict_test)) 

    info = [f'Method {method} \n mse = {mse}, momentum = {momentum}, last learning rate = {eta}, batch size = {M}, epochs = {save_e}']
    
    print(f'MSE for stochastic gradient descent with batches is {mse} \n abs error is {abs_error_avg}')
    print(f'{info}\n')
    return predict, beta, mse, info



'''Function calls'''



run = gd_momentum(beta, X, z, method='adagrad', iterations=100000, rate=3, Auto=False)
info = run[3]
predict = run[0] #Our model for GD using OLS, this version of predict has a shape (n*n,). It is 1D



sgd = sgd_momentum(X, z, method= 'adagrad', M=32, epochs=50, Auto=False)

sgd_predict = sgd[0]
sgd_info = sgd[3]

print(np.shape(sgd_predict))



'''Contour plot Stochastic Gradient Descent w mini batches'''

sgd_predicted_grid = np.reshape(sgd_predict,(n, n)) #We need to reshape the predict back so that we can plot it on a grid

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Contour plot for z (Actual Franke model)
sgd_contour1 = ax[0].contourf(x_, y_, z, cmap='viridis')

fig.colorbar(sgd_contour1, ax=ax[0])

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[0].set_title(f'Actual Franke Model')

# Contour plot for predicted model using Gradient Descent
sgd_contour2 = ax[1].contourf(x_, y_, sgd_predicted_grid, cmap='viridis')

fig.colorbar(sgd_contour2, ax=ax[1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

ax[1].set_title(f'Stochastic Gradient Descent Prediction with mini batches')

#fig.savefig(f'SGD_comparison_franke_contour {sgd_info}.pdf')
plt.show()


'''Contour plot Gradient Descent'''

predicted_grid = np.reshape(predict, (n, n)) #We need to reshape the predict back so that we can plot it on a grid

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Contour plot for predicted model using Gradient Descent
contour1 = ax[0].contourf(x_, y_, predicted_grid, cmap='viridis')

fig.colorbar(contour1, ax=ax[0])
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title(f'Gradient Descent Prediction')

# Contour plot for z (Actual Franke model)
contour2 = ax[1].contourf(x_, y_, z, cmap='viridis')

fig.colorbar(contour2, ax=ax[1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title(f'Actual Franke Model')

#fig.savefig(f'GD_comparison_franke_contour {info}.pdf')
plt.show()


'''Meshgrid color plot Gradient Descent'''

fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 
cmesh1 = ax[0].pcolormesh(x_, y_, predicted_grid)
fig.colorbar(cmesh1, ax=ax[0])
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Gradient descent prediction')


cmesh2 = ax[1].pcolormesh(x_, y_, z)
fig.colorbar(cmesh2, ax=ax[1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Actual Franke model')
#fig.savefig(f'GD_comparison_franke_colormesh {info}.pdf')
plt.show()


"""
Question for TA:
Should momentum ever be added to the ADAM and/or RMSprop methods?


- ADAM and RMSprop edit the gradient and implicitly then change the learning rate, but not explicitly
        -> Not explicitly changing eta. Is that oki?


        

- Lag bash script for å automatisere:
        
-Grid search:
  -> lambda - regularization parameter
  -> eta - learning rate
- epochs(iterations) -> øker antall noder til MSE ikke blir bedre
                     -> legg til lag

- Bytt aktiveringsfunksjon


Leah:
- Gjør oppg e)



Lørdag:
- Integrere a i NN stuff
- fullføre oppg e

- Produser resultater 

- Start skriving: 
    -> Strukturer ferdig teori
    -> Legg inn og beskriv Resultater
    -> Kan gi notater til diskusjon
    -> Fullfør abstrakt
    

Søndag:
- Fullfør skriving:
    -> Fullfør teori

    -> Start og fullfør metode
    -> Skriv en draft intro og få ChGPT til å fikse den

    -> Skriv en draft diskusjon og konklusjon


Sjekkliste:
- Har vi nevnt alle oppgavene?
- Har alle figurer akse labels og figurtekts?
- Nevner vi all relevant teori?


"""