import numpy as np
import nnfs
np.random.seed(0)

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

class layerDense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros([1,n_neurons])

    def forwardPass(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1 = layerDense(4,5)
layer2 = layerDense(5,2)
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()


layer1.forwardPass(X)
# print(layer1.output)
activation1.forward(layer1.output)
# print(activation1.output)
layer2.forwardPass(activation1.output)
print(layer2.output)
activation2.forward(layer2.output)
print(activation2.output)
