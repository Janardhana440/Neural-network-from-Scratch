import numpy as np
import nnfs
from nnfs.datasets import spiral_data
np.random.seed(0)

nnfs.init()

class layerDense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros([1,n_neurons])

    def forwardPass(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias

class Activation_Softmax:
    def forward(self,inputs):
        exp_vals = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) # prevent overflow of values. Keeps values between 0 and 1 as e^0 = 1(max num afater subtraction is always 0)
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

X, y = spiral_data(samples=100, classes=3)


layer1 = layerDense(2,3)
layer2 = layerDense(3,3)
activation1 = Activation_Softmax()
activation2 = Activation_Softmax()


layer1.forwardPass(X)
# print(layer1.output)
activation1.forward(layer1.output)
# print(activation1.output)
layer2.forwardPass(activation1.output)
# print(layer2.output)
activation2.forward(layer2.output)
print(activation2.output[:5])
