"""
https://www.youtube.com/watch?v=YAJ5XBwlN4o&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Data preprocessing
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(-1, 1)

n_samples, n_features = X.shape

# 1. Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 3. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print("epoch: {}, loss={}".format(epoch+1, loss.item()))

# plot
predicted = model(X).detach()   # to prevent calculating the gradient
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

w, b = model.parameters()
print("W:",w.item())
print("b:",b.item())
