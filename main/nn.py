import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)

        self.output = output
        return output

    def backward(self, output, target, lr = 0.00001):
        gradients = []
        errors = []
        error = (output - target)

        for layer in reversed(self.layers):
            gradients.insert(0, layer.back_propogate(error, lr))
            errors.insert(0, error)
            error = error.dot(layer.weights.T)

        for i in range(len(gradients)):
            self.layers[i].adjust_gradients(gradients[i], lr)
            self.layers[i].adjust_biases(gradients[i], lr)
                

class ActivationOutput:
    def forward(self, input):
        return input

    def backward(self, input, error):
        return input.T.dot(error)


class ActivationRELU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, input, error):
        grad = error.copy()
        grad[self.input < 0] = 0  # the derivate of ReLU
        return input.T.dot(grad)

class DenseLayer:
    
    def __init__(self, num_inputs, num_neurons, activation):
        self.activation = activation
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons) * np.sqrt(2/num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation.forward(np.dot(inputs, self.weights)) + self.biases
        return self.output

    def back_propogate(self, error, lr):
        grad = self.activation.backward(self.inputs, error)
        return grad

    def adjust_gradients(self, gradient, lr):
        self.weights -= lr * gradient
    
    def adjust_biases(self, error, lr):
        self.biases -= lr * np.sum(error, axis=0, keepdims=True)


