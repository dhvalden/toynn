import pickle
import numpy as np
from numpy import dot, random

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    """Derivative of sigmoid function"""
    return y * (1.0 - y)

def tanh(x):
    """Hiperbolic tangent function"""
    return np.tanh(x)

def dtanh(y):
    """Derivative of tanh"""
    return 1 - y * y

def softmax(z):
    if len(z.shape) == 2:
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis = 1, keepdims = True)
    elif len(z.shape) == 1:
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis = 0)

class NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network
    based on the version from https://github.com/FlorianMuellerklein/
    Machine-Learning/blob/master/Old/BackPropagationNN.py
    with more concise versions of the same methods using numpy.
    This module was written to learn and for fun.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.5, 
            momentum = 0.5, rate_dacay = 0.01, activation_fun = "sigmoid"):
        self.inp_n = input_nodes + 1 # add 1 for bias node
        self.hid_n = hidden_nodes
        self.out_n = output_nodes
        self.lr = learning_rate
        self.momentum = momentum
        self.rd = rate_dacay
        self.activation_fun = activation_fun

        # Activation arrays
        self.ai = [1.0] * self.inp_n
        self.ah = [1.0] * self.hid_n
        self.ao = [1.0] * self.out_n

        # Weights arrays
        self.w_ih = random.normal(loc = 0, scale = 1, size = (self.inp_n, self.hid_n))
        self.w_ho = random.normal(loc = 0, scale = 1, size = (self.hid_n, self.out_n))

        # Change arrays (these temporally store the changes)
        self.ci = np.zeros((self.inp_n, self.hid_n))
        self.co = np.zeros((self.hid_n, self.out_n))

    def feedforward(self, inputs):
        """Feed forward algorithm short implementation"""
        if len(inputs) != self.inp_n - 1:
            raise ValueError("Input nodes don't match with number of inputs")

        self.ai[0:self.inp_n - 1] = inputs # -1 to avoid the bias

        # Activation of the hidden layer
        self.ah = tanh(dot(self.ai, self.w_ih))

        # Activation of the output layer
        if self.activation_fun == "sigmoid":
            self.ao = sigmoid(dot(self.ah, self.w_ho))
        elif self.activation_fun == "softmax":
            self.ao = softmax(dot(self.ah, self.w_ho))
        else:
            raise ValueError("Invalid activation function")
        return self.ao[:]

    def backpropagate(self, targets):
        if len(targets) != self.out_n:
            raise ValueError("Output nodes don't match with number of ouputs")

        error = np.array(targets) - self.ao
        if self.activation_fun == "sigmoid":
            output_deltas = dsigmoid(self.ao) * -error
        elif self.activation_fun == "softmax":
            output_deltas = -error
        else:
            raise ValueError("Invalid activation function")

        hidden_errors = dot(self.w_ho, output_deltas)
        hidden_deltas = dtanh(self.ah) * hidden_errors

        change_o = np.matrix(self.ah).T * output_deltas
        self.w_ho -= self.lr * change_o + self.co * self.momentum
        self.co = change_o

        change_h = np.matrix(self.ai).T * hidden_deltas
        self.w_ih -= self.lr * change_h + self.ci * self.momentum
        self.ci = change_h

        error = 0.0
        if self.activation_fun == "sigmoid":
            for i in range(len(targets)):
                error += 0.5 * (targets[i] - self.ao[i]) ** 2
        elif self.activation_fun == "softmax":
            for i in range(len(targets)):
                error += -(targets[i] * np.log(self.ao[i]))

        return error

    def train(self, data, iterations, verbose = False):
        num_ex = np.shape(data)[0]

        for i in range(iterations):
            error = 0.0
            random.shuffle(data)
            for d in data:
                inputs = d[0]
                targets = d[1]
                self.feedforward(inputs)
                error += self.backpropagate(targets)
            self.lr = self.lr * (self.lr / (self.lr + (self.lr * self.rd)))
            if i % 1 == 0 and verbose:
                error = error/num_ex
                print("Error -> %.5f - Iteration %.0f" % (error, i))

    def predict(self, data):
        outputs = []
        for d in data:
            outputs.append(self.feedforward(d))
        return outputs

    def save(self, file_name):
        file_manager = open(str("%s.p" % file_name), "wb")
        pickle.dump(self, file_manager)
        file_manager.close()

    @staticmethod
    def load(file_name):
        file_manager = open("%s.p" % file_name, "rb")
        new_object = pickle.load(file_manager)
        file_manager.close()
        return new_object
