import numpy as np
import nnfs
from nnfs.datasets import spiral_data
np.random.seed(0)

nnfs.init()
learning_rate = 0.01
target_output = 0

class layerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forwardPass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_loss_categoricalcrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_categorical_cross_entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimiser_SGD:
    def __init__(self,learning_rate = 1.0,momentum = 0,decay = 0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        

    def pre_update_prams(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self,layer:layerDense):
        
        if self.momentum:
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)

            weight_update = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            layer.weight_momentums = weight_update

            bias_update = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases
            layer.bias_momentums = bias_update

        else:
            weight_update = -self.current_learning_rate * layer.dweights
            bias_update = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_update
        layer.bias += bias_update

    def post_update_params(self):
        self.iterations += 1
    


X, y = spiral_data(samples=100, classes=3)

layer1 = layerDense(2,64)
layer2 = layerDense(64,3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_activation = Activation_Softmax_loss_categoricalcrossentropy()
optimser = Optimiser_SGD(momentum= 0.9,decay=1e-3)


for epoch in range(10001):
    layer1.forwardPass(X)
    # print(layer1.output)
    activation1.forward(layer1.output)
    # print(activation1.output)
    layer2.forwardPass(activation1.output)
    loss = loss_activation.forward(layer2.output,y)
    

    predictions = np.argmax(loss_activation.output,axis=1)
    if len(predictions.shape) == 2:
        y = np.argmax(y,axis=1)
    
    accuracy = np.mean(predictions==y)

    loss_activation.backward(loss_activation.output,y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    optimser.pre_update_prams()
    optimser.update_params(layer1)
    optimser.post_update_params()
    optimser.update_params(layer2)

    if not epoch % 100: 
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimser.current_learning_rate:.3f}')
