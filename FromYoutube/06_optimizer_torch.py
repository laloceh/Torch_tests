"""
https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6

1. Design model (input_size, output_size, forward pass)
2. Construct loss and optimizer
3. Training loop:
    - forward pass: compute prediction
    - brackward pass: gradients
    - update weights

"""

import torch
import torch.nn as nn
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

# loss is done now with pytorch

# gradient is done now with Pytorch

x_test = torch.tensor(5, dtype=torch.float32)
print("Prediction before training: f(5) = {:.3f}".format(forward(x_test).item()) )

#training
learning_rate = 0.01
n_iters = 75
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient -> dL/dw
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print("epoch {}: w = {:.3f}, loss={:.8f}".format(epoch+1, w.item(), l))

print("Prediction after training: f(5) = {:.3f}".format(forward(x_test).item()) )

