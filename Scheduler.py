import autograd.numpy as np

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass

    def set_eta(self, eta):
        self.eta = eta

class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass

class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass

class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros_like(gradient)

        self.G_t += gradient * gradient

        return self.eta * gradient * 1 / (delta + np.sqrt(self.G_t))

    def reset(self):
        self.G_t = None

class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros_like(gradient)
        
        self.G_t += gradient * gradient

        self.change = self.change * self.momentum + self.eta * gradient * 1 / (delta + np.sqrt(self.G_t))
        return self.change

    def reset(self):
        self.G_t = None

class RMS_prop(Scheduler):
    def __init__(self, eta, beta):
        super().__init__(eta)
        self.beta = beta
        self.Giter = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.Giter = self.beta * self.Giter + (1 - self.beta) * (gradient ** 2)
        update_gradient = ((self.eta* gradient) / (np.sqrt(self.Giter) + delta))
        return update_gradient

    def reset(self):
        self.Giter = 0.0

class RMS_propMomentum(Scheduler):
    def __init__(self, eta, beta, momentum):
        super().__init__(eta)
        self.beta = beta
        self.Giter = 0.0
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.Giter = self.beta * self.Giter + (1 - self.beta) * (gradient ** 2)
        update_gradient = self.change * self.momentum + ((self.eta* gradient) / (np.sqrt(self.Giter) + delta))
        return update_gradient

    def reset(self):
        self.Giter = 0.0

class Adam(Scheduler):
    def __init__(self, eta, beta1, beta2):
        super().__init__(eta)
        self.beta1 = beta1
        self.beta2 = beta2
        self.mom1 = 0
        self.mom2 = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.mom1 = self.beta1 * self.mom1 + (1 - self.beta1) * gradient  # Update 1st moment vector
        self.mom2 = self.beta2 * self.mom2 + (1 - self.beta2) * (gradient ** 2)  # Update 2nd moment vector
        mom1_corrected = self.mom1 / (1 - self.beta1 ** self.n_epochs)  # Bias-corrected 1st moment estimate
        mom2_corrected = self.mom2 / (1 - self.beta2 ** self.n_epochs)  # Bias-corrected 2nd moment estimate

        return (self.eta / (np.sqrt(mom2_corrected) + delta)) * mom1_corrected

    def reset(self):
        self.n_epochs += 1
        self.mom1 = 0
        self.mom2 = 0

class AdamMomentum(Scheduler):
    def __init__(self, eta, beta1, beta2, momentum):
        super().__init__(eta)
        self.beta1 = beta1
        self.beta2 = beta2
        self.mom1 = 0
        self.mom2 = 0
        self.n_epochs = 1
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.mom1 = self.beta1 * self.mom1 + (1 - self.beta1) * gradient  # Update 1st moment vector
        self.mom2 = self.beta2 * self.mom2 + (1 - self.beta2) * (gradient ** 2)  # Update 2nd moment vector
        mom1_corrected = self.mom1 / (1 - self.beta1 ** self.n_epochs)  # Bias-corrected 1st moment estimate
        mom2_corrected = self.mom2 / (1 - self.beta2 ** self.n_epochs)  # Bias-corrected 2nd moment estimate

        self.change = self.change * self.momentum + (self.eta / (np.sqrt(mom2_corrected) + delta)) * mom1_corrected
        return self.change

    def reset(self):
        self.n_epochs += 1
        self.mom1 = 0
        self.mom2 = 0