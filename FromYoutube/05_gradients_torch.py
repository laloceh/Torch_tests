"""
https://www.youtube.com/watch?v=E-I2DNVzQLg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5
"""

import torch
import sys

# f = w * x
# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32).view(-1,1)
Y = torch.tensor([2,4,6,8], dtype=torch.float32).view(-1,1)
print(X.shape)
print(Y.shape)

w = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss
def loss(y, y_predicted):
    return ((y-y_predicted)**2).mean()

# gradient is done now with Pytorch

x_test = torch.tensor(5, dtype=torch.float32)
print("Prediction before training: f(5) = {:.3f}".format(forward(x_test).item()) )

#training
learning_rate = 0.01
n_iters = 75

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient -> dL/dw
    l.backward()

    # update weights
    # As this is another operation involving w, we don't wat the gradients
    # of this operation being accumulated
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero the gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print("epoch {}: w = {:.3f}, loss={:.8f}".format(epoch+1, w.item(), l))

print("Prediction after training: f(5) = {:.3f}".format(forward(x_test).item()) )

