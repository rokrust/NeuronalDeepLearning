import linearNeuron
import tanhNeuron
import sigmoidNeuron
import reluNeuron

from cost_functions import *
from gradientDescent import *
import numpy as np

class NeuralNetwork:
    def __init__(self, w_i, w_h, n_h, w_o,
                 hidden_neuron_type=reluNeuron,
                 output_neuron_type=linearNeuron):

        # Architecture Parameters:
        self.input_width  = w_i
        self.hidden_width = w_h
        self.hidden_depth = n_h
        self.output_width = w_o
        self.n_layer      = self.hidden_depth + 1

        # Non-Linearity of the individual network layer:
        # -) Sigmoid Neuron
        # -) Tanh Neuron
        # -) ReLu Neuron
        # -) Linear Neuron
        self.g_h        = hidden_neuron_type.g# Hidden unit neuron type
        self.g_o        = output_neuron_type.g# Output unit neuron type
        self.grad_g_h   = hidden_neuron_type.grad_g# Gradient of hidden neurons
        self.grad_g_o   = output_neuron_type.grad_g# Gradient of output neurons

        # Weight Initialization:
        # The weights should be initialize to a small non-zero value and the initial weights need to be asymmetric.
        # Typically the weights are i.i.d. and drawn from a normal or uniform distribution, where the variance is
        # scaled w.r.t. to in- (n) and output (m) connections.
        # -) Uniform distribution   i.e. ~ U(-sqrt(6/(n+m)), +sqrt(6/(n+m)))
        # -) Normal distribution 1  i.e. ~ sqrt(2/n) N(0, 1)
        # -) Normal distribution 2  i.e. ~ sqrt(2/(n+m)) N(0, 1)
        self.w_uniform  = lambda m, n: np.random.uniform(-sqrt(6.0/(n+m)), sqrt(6.0/(n+m)))# Fill in Lambda Function
        self.w_normal_1 = lambda m, n: math.sqrt(2.0/n)*np.random.normal(0, 1)# Fill in Lambda Function
        self.w_normal_2 = lambda m, n: math.sqrt(2.0/(n+m))*np.random.normal(0, 1)# Fill in Lambda Function
        self.weight_pdf = self.w_normal_1

        # The biases can be initialized to 0 or to a small positive number (0.001) to make the ReLu Units active for
        # the input distribution
        self.b_0        = 0.001# Fill in Value

        # Regularization
        # -) no normalization        i.e. J_reg(w) = 0
        # -) l1 norm regularization  i.e. J_reg(w) = alpha * ||w||_1
        # -) l2 norm regularization  i.e. J_reg(w) = alpha * ||w||_2
        self.reg =          # Optional Function Handle
        self.grad_reg =     # Optional Function Handle
        self.reg_alpha =    # Optional Constant

        # Initialize the cost function:
        # Parametrization of the cost function and an additive regularization term. Currently, the MSE cost function
        # and l1-, l2-norm can be used for regularization.
        self.j_entropy      = j_mse# Entropy cost function
        self.grad_j_entropy = grad_j_mse# Gradient of the Mean Squared Error
        self.j_reg          = l2_norm# Additive regularization cost function
        self.grad_j_reg     = grad_l2_norm# Gradient of the additive regularization cost function

        # Training of the Neural Network:
        # The neural network is trained via stochastic gradient descent with varying batch sizes and early stopping to
        # prevent overfitting.
        self.sgd = {'j min': 1e-6,
                    'max epoch': 1000,
                    'batch size': 10,
                    'alpha': 1e-4,
                    'p train': 0.8,
                    'p test': 0.1,
                    'p val': 0.1}

        # -) Momentum:
        # To increase convergence speed, momentum accumulates an exponential mean of the previous gradients and adds
        # this accumulated gradient to the current gradient.
        self.momentum = {'select': False,
                         'rho': 0.9}

        # -) Adaptive Learning Rates:
        # To prevent overshooting, the learning rate is decreased by the accumulated squared gradient.
        self.rmsprop = {'select': False,
                        'rho': 0.9}
        # -) ADAM:
        # Combines the concept of momentum with the adaptive learning rate to increase convergence speed and prevent
        # overshooting.
        self.adam = {'select': True,
                     'alpha': 1e-3,
                     'beta 1': 0.95,
                     'beta 2': 0.9}

        # State Variables:
        self.x_train    = None
        self.y_train    = None
        self.w          = []   # Weights of the respective layer
        self.b          = []   # Bias of the respective layer
        self._grad_w    = []   # Weight change of the respective layer
        self._grad_b    = []   # Bias change of the respective layer
        self._h_i       = []   # Activation of the respective layer
        self._g_i       = []   # Non-Linearity of the respective layer
        self._grad_g_i  = []   # Gradient of the non-linearity of the respective layer
        self._optimizer = None
        self._trained   = False

        # Debugging Variables:
        self._fig = None
        self._fig_j = None

        # Initialize the weight matrix:
        self.weight_initialization()


    def eval(self, x):
        # Evaluate the network output
        self.out = [ [0 for i in range(self.hidden_width)] for j in range(self.n_layer) ]
        
        #first layer
        for i in range(self.hidden_width):
            for j in range(self.input_width):
                out[0][i] += x[j] * self.w[0][i][j]
            
            out[0][i] = self.g_h(node_sum[i])
            
        #hidden layers
        for i in range(1, self.n_layer):
            for j in range(self.hidden_width):
                for k in range(self.hidden_width):
                    out[i][j] += out[i-1][k] * self.w[i][j][k]

                out[i][j] = self.g_h(node_sum[i])
                
        return out[self.n_layer - 1][0:self.output_width]
    

    def weight_initialization(self):
        # Initialize all parameters
        #first layer
        self.w = [ [self.weight_pdf(self.input_width, self.hidden_width) for x in range(self.input_width)] for y in range(self.hidden_depth) ]
        
        #Hidden layers
        for i in range(self.hidden_depth-1):
            self.w.append([ [self.weight_pdf(self.hidden_width, self.hidden_width) for x in range(self.input_width)] for y in range(self.hidden_depth) ])
            
        #Output layer
        self.w.append([ [self.weight_pdf(self.hidden_width, self.output_width) for x in range(self.input_width)] for y in range(self.hidden_depth) ])


    def j(self, x, y):
        # Network Cost Function
        y_hat = self.eval(x)
        return self.j_entropy(y, y_hat)
        
        
    def grad_j(self, x, y):
        # Gradient w.r.t. to all parameters
        #
        #
        # Fill in
        #
        #
        #
        #
        #
        return 0

    def train(self,
              x=None,
              y=None,
              reset = False,
              plot = False,
              verbose = False):

        assert x is not None or self.x_train is not None
        assert y is not None or self.y_train is not None
        assert (x is None) == (y is None)
        assert isinstance(reset, bool)
        assert isinstance(plot, bool)
        assert isinstance(verbose, bool)

        # Gradient w.r.t. to all parameters
        #
        #
        # Fill in
        #
        #
        #
        #
        #
        return 0

    def _backprop(self, y, y_hat):
        pass


if __name__ == '__main__':

    # For debugging purpose with the same seed:
    # np.random.seed(44)
    plt.ion()

    n_samples   = 500
    d_input     = 1
    d_output    = 1

    def func(x):
        shift = 0.1
        if x > shift:
            return 0.7 * (x-shift) + 0.7
        else:
            return -1.5 * (x-shift) + 0.7

    x = np.linspace(-3., 3., n_samples)[np.newaxis].transpose()
    y = np.zeros(x.shape)

    for i in xrange(0,len(x)):
        y[i] = func(x[i])

    network = NeuralNetwork(1, 5, 1, 1,
                            hidden_neuron_type=reluNeuron,
                            output_neuron_type=linearNeuron)

    network.train(x, y, plot=True, verbose=True)
    plt.show()