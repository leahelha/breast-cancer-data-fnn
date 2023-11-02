import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from FFNN import FFNN
from activation_functions import sigmoid
from cost_functions import CostOLS
from Scheduler import Momentum, Constant, Adagrad, Adam
from sklearn.neural_network import MLPRegressor

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
# z = add_noise(FrankeFunction(x_, y_), 0.1)
z = FrankeFunction(x_, y_)
z_fit = z.reshape(-1,1)

xy_train, xy_test, z_train, z_test = train_test_split(xy, z_fit, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls

xy_mean = np.mean(xy_train)
z_mean = np.mean(z_train)
xy_std = np.std(xy_train)
z_std = np.std(z_train)

xy_train_norm = (xy_train-xy_mean)/xy_std
z_train_norm = (z_train-z_mean)/z_std
xy_test_norm = (xy_test-xy_mean)/xy_std
z_test_norm = (z_test-z_mean)/z_std

# Create neural network
network = FFNN((xy.shape[1], 50, 1), sigmoid, lambda x: x, CostOLS, 10)
eta_vals = anp.logspace(-4,-1,4)
lmbda_vals = anp.logspace(-5,0,6)

mse = np.zeros((len(eta_vals), len(lmbda_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbda_vals)):
        network.reset_weights_and_bias()
        sched_xor = Adam(eta_vals[i], 0.9, 0.999)
        network.fit(xy_train_norm, z_train_norm, sched_xor, 1, 100, lmbda_vals[j])
        z_pred = network.predict(xy_test_norm)
        mse[i][j] = np.mean((z_pred - z_test_norm)**2)


        # skl_network = MLPRegressor(hidden_layer_sizes = (50), activation = 'logistic', solver = 'adam', alpha = lmbda, learning_rate_init = eta, max_iter = 100, random_state = 10)
        # skl_network.fit(xy, z.flatten())
        # print(f"Sklearn: {skl_network.score(xy, z.flatten())}")

fig, ax = plt.subplots(figsize = (10, 8))
sns.heatmap(mse, annot = True, cmap = "viridis", square = True, yticklabels = eta_vals, xticklabels = lmbda_vals)
ax.set_title("Test MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig("plots/test.pdf")