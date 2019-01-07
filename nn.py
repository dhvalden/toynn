import math
import numpy as np
from numpy import dot, random
np.seterr(all = 'ignore')

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    """Derivative of sigmoid function"""
    return y * (1.0 - y)

def tanh(x):
    """Hiperbolic tangent function"""
    return math.tanh(x)

def dtanh(y):
    """Derivative of tanh"""
    return 1 - y * y

class NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network
    practically copied from https://github.com/FlorianMuellerklein/
    Machine-Learning/blob/master/Old/BackPropagationNN.py
    with small changes.
    This module was written to learn and for fun.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.5, momentum = 0.5, rate_dacay = 0.01):
        self.inp_n = input_nodes + 1 # add 1 for bias node
        self.hid_n = hidden_nodes
        self.out_n = output_nodes
        self.lr = learning_rate
        self.momentum = momentum
        self.rd = rate_dacay

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
        """Feedforward algorithm long implementation"""
        if len(inputs) != self.inp_n - 1:
            raise ValueError("Input nodes don't match with number of inputs")
        # -1 to avoid the bias
        for i in range(self.inp_n - 1):
            self.ai[i] = inputs[i]

        # Activation of the hidden layer using matrix multiplication
        for j in range(self.hid_n):
            sum = 0.0
            for i in range(self.inp_n):
                sum += self.ai[i] * self.w_ih[i][j]
            self.ah[j] = tanh(sum) # Hyperbolic tangent as activation func

        # Activation of the output layer using matrix multiplication
        for k in range(self.out_n):
            sum = 0.0
            for j in range(self.hid_n):
                sum += self.ah[j] * self.w_ho[j][k]
            self.ao[k] = sigmoid(sum) # Sigmoid as activation func

        return self.ao[:]

    def backpropagate(self, targets):
        if len(targets) != self.out_n:
            raise ValueError("Output nodes don't match with number of ouputs")

        output_deltas = [0.0] * self.out_n
        for k in range(self.out_n):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.hid_n
        for j in range(self.hid_n):
            error = 0.0
            for k in range(self.out_n):
                error += output_deltas[k] * self.w_ho[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        for j in range(self.hid_n):
            for k in range(self.out_n):
                change = output_deltas[k] * self.ah[j]
                self.w_ho[j][k] -= self.lr * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        for i in range(self.inp_n):
            for j in range(self.hid_n):
                change = hidden_deltas[j] * self.ai[i]
                self.w_ih[i][j] -= self.lr * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        error = 0.0
        for i in range(len(targets)):
            error += 0.5 * (targets[i] - self.ao[i]) ** 2
        return error

    def train(self, data, iterations):
        for i in range(iterations):
            error = 0.0
            random.shuffle(data)
            for d in data:
                inputs = d[0]
                targets = d[1]
                self.feedforward(inputs)
                error += self.backpropagate(targets)
            self.lr = self.lr * (self.lr / (self.lr + (self.lr * self.rd)))
            #if i % 100 == 0:
                #print(error)
