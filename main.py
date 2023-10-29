import autograd.numpy as anp
from FFNN import FFNN
from activation_functions import sigmoid
from cost_functions import CostCrossEntropy, CostLogReg
from Scheduler import Momentum, Constant, Adagrad, Adam
from sklearn.neural_network import MLPClassifier

# Design matrix
X = anp.asarray([[0,0], [0,1], [1,0], [1,1]])

# Targets for different gates
y_xor = anp.asarray([0,1,1,0]).reshape(-1,1)
y_and = anp.asarray([0,0,0,1]).reshape(-1,1)
y_or = anp.asarray([0,1,1,1]).reshape(-1,1)

# Create neural network
network = FFNN((2,2,1), sigmoid, sigmoid, CostLogReg, 10)
eta_vals = anp.logspace(-4,-1,4)
lmbda_vals = anp.logspace(-5,-1,5)

# One feed forward (predict without training first)
first_pred = network.predict(X)

# XOR gate
print("--- XOR GATE ---")
for eta in eta_vals:
    for lmbda in lmbda_vals:
        network.reset_weights()
        sched_xor = Adam(eta, 0.9, 0.999)
        network.fit(X, y_xor, sched_xor, 1, 1000, lmbda)
        pred_xor = network.predict(X)
        # print(f"\n{pred_xor}")

        skl_network = MLPClassifier(hidden_layer_sizes = (2), activation = 'logistic', solver = 'adam', alpha = lmbda, learning_rate_init = eta, max_iter = 1000, random_state = 10)
        skl_network.fit(X, y_xor.ravel())
        print(f"Sklearn: {skl_network.score(X, y_xor)}")

# OR gate
print("--- OR GATE ---")
for eta in eta_vals:
    for lmbda in lmbda_vals:
        network.reset_weights()
        sched_or = Adam(eta, 0.9, 0.999)
        network.fit(X, y_or, sched_or, 1, 1000, lmbda)
        pred_or = network.predict(X)
        # print(f"\n{pred_or}")

        skl_network = MLPClassifier(hidden_layer_sizes = (2), activation = 'logistic', solver = 'adam', alpha = lmbda, learning_rate_init = eta, max_iter = 1000, random_state = 10)
        skl_network.fit(X, y_or.ravel())
        print(f"Sklearn: {skl_network.score(X, y_or)}")

# AND gate
print("--- AND GATE ---")
for eta in eta_vals:  
    for lmbda in lmbda_vals:
        network.reset_weights()
        sched_and = Adam(eta, 0.9, 0.999)
        network.fit(X, y_and, sched_and, 1, 1000, lmbda)
        pred_or = network.predict(X)
        # print(f"\n{pred_and}")

        skl_network = MLPClassifier(hidden_layer_sizes = (2), activation = 'logistic', solver = 'adam', alpha = lmbda, learning_rate_init = eta, max_iter = 1000, random_state = 10)
        skl_network.fit(X, y_and.ravel())
        print(f"Sklearn: {skl_network.score(X, y_and)}")