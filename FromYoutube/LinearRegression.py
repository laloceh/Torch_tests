

import torch
import torch.nn as nn
import torch.optim as optim

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

m = Model()

criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(m.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = m(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Testing
hour_var = torch.Tensor([[4.0]])
print("predict for {} = {}".format(hour_var.item(), round(m.forward(hour_var).item(),2)))