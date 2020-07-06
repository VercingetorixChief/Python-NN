import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def mse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


x = np.arange(50)
delta = np.random.uniform(-1,1, size=(50,))
y = np.sin( x)


x = np.array(x)
y = np.array(y)

def gradient_descent(X, y, lr=0.0001, epoch=10):
    
    # Building the model
    m = 0
    c = 0

    epochs = 1000  # The number of iterations to perform gradient descent

    n = float(len(X)) # Number of elements in X
    Y_pred = m*X + c  # The current predicted value of Y
    # Performing Gradient Descent 
    epoch = 0;
    while(mse(Y_pred, y) > 1.0 and epoch < 2000): 
        epoch+=1
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(y - Y_pred)  # Derivative wrt c
        m -= lr * D_m  # Update m
        c -= lr * D_c  # Update c
        print("Epoch: %.5f , Error: %.5f" % (epoch, mse(Y_pred, y)))
    
    return (m, c)

m, b = gradient_descent(x, y)
y_ = m * x + b

plt.plot(x, y, y_)
plt.show()




# N is batch size(sample size); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4, 2, 30, 1

# Create random input and output data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 0.002
loss_col = []
for t in range(200):
    # Forward pass: compute predicted y
    layer1 = x.dot(w1)
    layer1_relu = np.maximum(layer1, 0)  # using ReLU as activate function
    y_pred = layer1_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum() # loss function
    loss_col.append(loss)
    print(t, loss, y_pred)

    # Backprop to compute gradients of w1 and w2 with respect to loss

    # the last layer's error
    grad_y_pred = 2.0 * (y_pred - y) 



    grad_w2 = input.T.dot(error)



    # the second laye's error 
    grad_layer1_relu = grad_y_pred.dot(w2.T)


    grad_layer1 = grad_layer1_relu.copy()
    grad_layer1[layer1 < 0] = 0  # the derivate of ReLU
    grad_w1 = x.T.dot(grad_layer1)





    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(loss_col)
plt.show()