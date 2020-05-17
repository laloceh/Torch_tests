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
input_size = X.shape[1]
output_size = Y.shape[1]

# model prediction
#model = nn.Linear(input_size, output_size)
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# loss is done now with pytorch

# gradient is done now with Pytorch

x_test = torch.tensor([5], dtype=torch.float32)
print("Prediction before training: f(5) = {:.3f}".format(model(x_test).item()) )

#training
learning_rate = 0.02
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient -> dL/dw
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print("epoch {}: w = {:.3f}, loss={:.8f}".format(epoch+1, w[0][0].item(), l))

print("Prediction after training: f(5) = {:.3f}".format(model(x_test).item()) )

