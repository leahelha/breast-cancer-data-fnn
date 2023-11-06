import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from pathlib import Path

from FFNN import FFNN
from activation_functions import sigmoid, RELU, LRELU
from cost_functions import CostOLS, CostCrossEntropy
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
test_size = 0.2
random_state = 3
xy_train, xy_test, z_train, z_test = train_test_split(xy, z_fit, test_size = test_size, random_state = random_state) # random_state gives same partition across multiple function calls



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
model_shape = [(50), (100), (50,50)]
scheduler = Scheduler.Adam(0, 0.9, 0.999)
eta_vals = anp.logspace(-3,-2,2)
lmbda_vals = anp.logspace(-5,0,6)
batches_vals = [1]
epochs_vals = [100, 200]
activation_functions = [sigmoid, RELU, LRELU]

# Refined search based on search with parameters above, comment these two lines out to do the broader search
model_shape = [(50), (100)]
eta_vals = [0.001]
epochs_vals = [500]

# Train the combinations of parameters
root_path = Path.cwd()
problem = "regression"
print("\n--- TRAINING NEURAL NETWORK ON FRANKE FUNCTION REGRESSION ---")
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
                file_path = root_path / "plots" / problem / layer_name / batch_name / epoch_name / act_name
                file_path.mkdir(parents=True, exist_ok=True)
                parameters_file = f"""Parameters for the FFNN:
problem = {problem}
batches = {batches} 
epochs = {epochs}
network shape = {network_shape}
- First layer: {network_shape[1]}
- Second layer: {network_shape[2]}
Activation function for hidden layers = {act_func.__name__}
"""
                save_parameters(parameters_file, file_path)

                print(f"\nNetwork shape: {network_shape}\nEpochs: {epochs}\nBatches: {batches}\nActivation function: {act_name}\n")
                network = FFNN(network_shape, act_func, lambda x: x, CostOLS, 10)
                mse_FFNN, r2_FFNN = train_pred_FFNN(network, xy_train_norm, xy_test_norm, z_train_norm, z_test_norm, eta_vals, lmbda_vals, scheduler, batches, epochs)
                plot_heatmap(mse_FFNN, file_path / "mse_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
                plot_heatmap(r2_FFNN, file_path / "r2_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)

                # # Using Scikit-Learn for the sigmoid activation functions:
                # if act_func.__name__ == "sigmoid": # TODO: Ordne med Sklearn, ikke konvergerer
                #     mse_skl, r2_skl = train_pred_skl(xy_train_norm, xy_test_norm, z_train_norm, z_test_norm, eta_vals, lmbda_vals, network_shape[1:-1], 'logistic', 'adam', batches, epochs)
                #     plot_heatmap(mse_skl, file_path / "mse_skl.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
                #     plot_heatmap(r2_skl, file_path / "r2_skl.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)

### DATA SETUP ###
X, y = load_breast_cancer(return_X_y = True)
y_fit = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size = test_size, random_state = random_state)

# Normalise data
X_mean = np.mean(X_train)
X_std = np.std(X_train)

X_train_norm = (X_train-X_mean)/X_std
X_test_norm = (X_test-X_mean)/X_std

### CLASSIFICATION WITH NEURAL NETWORK ###

# Create neural network and choose parameters
model_shape = [(50), (100), (50,50)]
scheduler = Scheduler.Adam(0, 0.9, 0.999)
eta_vals = anp.logspace(-4,-2,3)
lmbda_vals = anp.logspace(-5,0,6)
batches_vals = [1]
epochs_vals = [100, 200]
activation_functions = [sigmoid, RELU, LRELU]

# Refined search based on search with parameters above, comment these two lines out to do the broader search
model_shape = [(50)]
eta_vals = [0.001]
epochs_vals = [50, 100]

# Train the combinations of parameters
root_path = Path.cwd()
problem = "classification"

print("\n--- TRAINING NEURAL NETWORK ON BREAST CANCER CLASSIFICATION ---")

for hidden_layer in model_shape:
    if isinstance(hidden_layer, int):
        layer_name = f"hidden_layers_{hidden_layer}"
        network_shape = (X.shape[1], hidden_layer, 1)
    elif isinstance(hidden_layer, tuple):
        layer_name = "hidden_layers"
        for layer in hidden_layer:
            layer_name = layer_name + f"_{layer}"
        network_shape = (X.shape[1], *hidden_layer, 1)

    for batches in batches_vals:
        batch_name = f"batches_{batches}"
        for epochs in epochs_vals:
            epoch_name = f"epochs_{epochs}"
            for act_func in activation_functions:
                if ((act_func == RELU or act_func == LRELU) and hidden_layer == (50, 25)):
                    continue

                act_name = f"act_func_{act_func.__name__}"
                file_path = root_path / "plots" / problem / layer_name / batch_name / epoch_name / act_name
                file_path.mkdir(parents=True, exist_ok=True)

                print(f"\nNetwork shape: {network_shape}\nEpochs: {epochs}\nBatches: {batches}\nActivation function: {act_name}\n")
                network = FFNN(network_shape, act_func, sigmoid, CostCrossEntropy, 10)
                accuracy_FFNN = train_pred_FFNN(network, X_train_norm, X_test_norm, y_train, y_test, eta_vals, lmbda_vals, scheduler, batches, epochs, regression = False)
                parameters_file = f"""Parameters for the FFNN:
problem = {problem}
batches = {batches} 
epochs = {epochs}
network shape = {network_shape}
- First layer: {network_shape[1]}
- Second layer: {network_shape[2]}
Activation function for hidden layers = {act_func.__name__}
"""
                save_parameters(parameters_file, file_path)
                plot_heatmap(accuracy_FFNN, file_path / "accuracy_FFNN.pdf", r"$\eta$", r"$\lambda$", eta_vals, lmbda_vals)
