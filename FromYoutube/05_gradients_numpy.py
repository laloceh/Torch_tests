"""
https://www.youtube.com/watch?v=E-I2DNVzQLg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5
"""

import numpy as np

# f = w * x
# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forard(x):
    return w * x

# loss
def loss(y, y_predicted):
    return ((y-y_predicted)**2).mean()

# gradient
# MSE = 1/N * (w*y - y)**2
# dJ/w = 1/N 2x (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print("Prediction before training: f(5) = {:.3f}".format(forard(5)) )

#training
learning_rate = 0.01
n_iters = 15

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forard(X)

    # loss
    l = loss(Y, y_pred)

    # gradient
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print("epoch {}: w = {:.3f}, loss={:.8f}".format(epoch+1, w, l))

print("Prediction after training: f(5) = {:.3f}".format(forard(5)) )

