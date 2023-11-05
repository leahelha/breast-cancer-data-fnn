import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from FFNN import FFNN
from activation_functions import sigmoid, RELU, LRELU
from cost_functions import CostOLS
import Scheduler
from helper_functions import train_pred_FFNN, train_pred_skl, plot_heatmap, save_parameters
### DATA SETUP ###

def FrankeFunction(x,y):
    '''Calculates the two-dimensional Franke's function.'''
    term1 = 0.75*anp.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*anp.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*anp.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*anp.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

n = 101 # number of points along one axis, total number of points will be n^2
rng = anp.random.default_rng(seed = 25) # seed to ensure same numbers over multiple runs
x = anp.sort(rng.random((n, 1)), axis = 0)
y = anp.sort(rng.random((n, 1)), axis = 0)
x_, y_ = anp.meshgrid(x, y)
xy = anp.stack((anp.ravel(x_),anp.ravel(y_)), axis = -1) # formatting needed to set up the design matrix
z = FrankeFunction(x_, y_)
z_fit = z.reshape(-1,1)

# Split into training and test
xy_train, xy_test, z_train, z_test = train_test_split(xy, z_fit, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls

# Normalise data
xy_mean = np.mean(xy_train)
z_mean = np.mean(z_train)
xy_std = np.std(xy_train)
z_std = np.std(z_train)

xy_train_norm = (xy_train-xy_mean)/xy_std
z_train_norm = (z_train-z_mean)/z_std
xy_test_norm = (xy_test-xy_mean)/xy_std
z_test_norm = (z_test-z_mean)/z_std



### REGRESSION WITH NEURAL NETWORK ###

# Create neural network and choose parameters
model_shape = [(50)]
scheduler = Scheduler.Adam(0, 0.9, 0.999)
eta_vals = anp.logspace(-4,-1,4)
lmbda_vals = anp.logspace(-5,0,6)
batches_vals = [1]
epochs_vals = [100]
activation_functions = [sigmoid, RELU, LRELU]

# Train the combinations of parameters
root_path = Path.cwd()
problem = "regression"
for hidden_layer in model_shape:
    if isinstance(hidden_layer, int):
        layer_name = f"hidden_layers_{hidden_layer}"
        network_shape = (xy.shape[1], hidden_layer, z_fit.shape[1])
    elif isinstance(hidden_layer, tuple):
        layer_name = "hidden_layers"
        for layer in hidden_layer:
            layer_name = layer_name + f"_{layer}"
        network_shape = (xy.shape[1], *hidden_layer, z_fit.shape[1])

    for batches in batches_vals:
        batch_name = f"batches_{batches}"
        for epochs in epochs_vals:
            epoch_name = f"epochs_{epochs}"
            for act_func in activation_functions:
                act_name = f"act_func_{act_func.__name__}"
                file_path = root_path / "plots" / problem / layer_name / act_name
                file_path.mkdir(parents=True, exist_ok=True)
                parameters_file = f"""Parameters for the FFNN:
batches = {batches} 
epochs = {epochs}
network shape = {network_shape}
"""
                save_parameters(parameters_file, file_path)
                # import ipdb;ipdb.set_trace()

                network = FFNN(network_shape, act_func, lambda x: x, CostOLS, 10)
                mse_FFNN, r2_FFNN = train_pred_FFNN(network, xy_train_norm, xy_test_norm, z_train_norm, z_test_norm, eta_vals, lmbda_vals, scheduler, batches, epochs)
                plot_heatmap(mse_FFNN, file_path / "mse_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
                plot_heatmap(r2_FFNN, file_path / "r2_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)

                # Using Scikit-Learn for the sigmoid activation functions:
                if act_func.__name__ == "sigmoid":
                    mse_skl, r2_skl = train_pred_skl(xy_train_norm, xy_test_norm, z_train_norm, z_test_norm, eta_vals, lmbda_vals, network_shape[1:-1], 'logistic', 'adam', batches, epochs)
                    plot_heatmap(mse_skl, file_path / "mse_skl.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
                    plot_heatmap(r2_skl, file_path / "r2_skl.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)


### CLASSIFICATION WITH NEURAL NETWORK ###

# Create neural network and choose parameters
model_shape = [(50)] # TODO: Gjør at man kan øke med sett lengde en eller 2 idk
# TODO: Legg til tqdm progress bar
scheduler = Scheduler.Adam(0, 0.9, 0.999)
eta_vals = anp.logspace(-4,-1,4)
lmbda_vals = anp.logspace(-5,0,6)
batches_vals = [1]
epochs_vals = [100]
activation_functions = [sigmoid, RELU, LRELU]

# Train the combinations of parameters
root_path = Path.cwd()
problem = "classification"
for hidden_layer in model_shape:
    if isinstance(hidden_layer, int):
        layer_name = f"hidden_layers_{hidden_layer}"
        network_shape = (xy.shape[1], hidden_layer, z.shape[1])
    elif isinstance(hidden_layer, tuple):
        layer_name = "hidden_layers"
        for layer in hidden_layer:
            layer_name = layer_name + f"_{layer}"
        network_shape = (xy.shape[1], *hidden_layer, z.shape[1])

    for batches in batches_vals:
        batch_name = f"batches_{batches}"
        for epochs in epochs_vals:
            epoch_name = f"epochs_{epochs}"
            for act_func in activation_functions:
                act_name = f"act_func_{act_func}"
                file_path = root_path / "plots" / problem / layer_name / act_name.__name__
                parameters_file = f"""Parameters for the FFNN:
batches = {batches} 
epochs = {epochs}
"""
                save_parameters(parameters_file)

                network = FFNN(network_shape, act_func, lambda x: x, CostOLS, 10)
                mse_FFNN, r2_FFNN = train_pred_FFNN(network, xy_train_norm, xy_test_norm, z_train_norm, z_test_norm, eta_vals, lmbda_vals, scheduler, batches, epochs)
                plot_heatmap(mse_FFNN, file_path / "mse_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
                plot_heatmap(r2_FFNN, file_path / "r2_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
