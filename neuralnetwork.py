import numpy as np

# Activation Function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Loss Function
def cost(yhat, y):
    return np.square(yhat - y)
def cost_dervative(yhat, y):
    return 2.0 * (yhat - y)

class NeuralNetwork():
    def __init__(self, shape):
        self.shape = shape
        self.layers = len(shape)
        self.weights = [np.random.randn(m,n) for m,n in zip(shape[:-1],shape[1:])]
        self.biases  = [np.random.randn(1,n) for n in shape[1:]]

    # Run the nework
    def feedforward(self,x):
        for w,b in zip(self.weights,self.biases):
            x = sigmoid( np.dot(x,w) + b)
        return x

    # Runs the network returning intermediate computations
    def forwardprop(self, x):
        a = x
        outputs = [a]
        inputs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(a,w) + b
            a = sigmoid(z)
            inputs.append(z)
            outputs.append(a)
        return inputs, outputs

    def backprop(self, x, y):
        b_grads = [np.zeros(b.shape) for b in self.biases]
        w_grads = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        inputs, outputs = self.forwardprop(x)
        yh = outputs[-1]

        # Compute gradients of output layer
        L = cost(yh,y.T)
        dL_dyh = cost_dervative(yh, y.T)
        dyh_dz = sigmoid_derivative( inputs[-1] )
        dL_dz = dL_dyh * dyh_dz
        b_grads[-1] = np.sum(dL_dz, axis=1, keepdims=True)
        w_grads[-1] = np.dot(outputs[-2].T, dL_dz) # [dz_dW2][dL_dz]

        # Compute gradients for hidden layers
        for n in range(2,self.layers):
            da_dz = sigmoid_derivative(inputs[-n])
            dL_da = dL_dz * da_dz
            dL_da = np.dot(dL_dz, self.weights[-n+1].T) # [dL_dz][dz_da]
            dL_dz = dL_da * da_dz
            b_grads[-n] = np.sum(dL_dz, axis=1, keepdims=True)
            w_grads[-n] = np.dot(outputs[-n-1].T, dL_dz) # [dz_dW1][dL_dz]

        return (b_grads, w_grads)

    def gradientdescent(self, x, y, iters, lr):
        train_errors = []
        for i in range(iters):
            b_grads, w_grads = self.backprop(x,y)
            self.weights = [ w-lr*dw for w,dw in zip(self.weights, w_grads)]
            self.biases  = [ b-lr*db for b,db in zip(self.biases, b_grads)]
            train_errors.append(np.average(cost(self.feedforward(x), y.T)))
        return train_errors

# XOR Problem:

x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[ 0,    1,    1,    0 ]])    

net = NeuralNetwork([2, 2, 1])

trainingerror = net.gradientdescent(x, y, 1000, 0.1)

print("Output:", net.feedforward(x).T)
print("Errors:", net.feedforward(x).T - y)

import matplotlib.pyplot as plt
plt.title("XOR Training")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.plot(trainingerror)
plt.savefig('XOR_error.png')