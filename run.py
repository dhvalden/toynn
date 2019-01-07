import np_nn
import numpy as np

data = np.loadtxt("sklearn_digits.csv", delimiter = ",")
y = data[:,0:10]

data = data[:,10:]
data -= data.min()
data /= data.max()
x = []
for i in range(data.shape[0]):
    z = list((data[i,:].tolist(), y[i].tolist()))
    x.append(z)


np.random.seed(2)

my_nn = np_nn.NeuralNetwork(64, 300, 10, learning_rate=0.001, momentum = 0.05, rate_dacay = 0.001, activation_fun = "softmax")

#data = [[[1, 0, 1], [1, 0, 0]],
#       [[0, 1, 0], [1, 0, 0]],
#       [[1, 1, 1], [0, 0, 1]],
#       [[0, 0, 0], [0, 0, 1]]]

#inputs = [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]]

#print(my_nn.predict(x[0][0]))

my_nn.train(x, iterations = 10, verbose = True)

print(my_nn.predict([x[0][0]]))

print(x[0][1])