"""
https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math


file = "glass.data"


class GlassDaset(Dataset):

    def __init__(self, file):
        xy = pd.read_csv(file, header=None)
        self.X = torch.from_numpy(xy.iloc[:, :-1].values)
        self.y = torch.from_numpy(xy.iloc[:, -1].values)
        self.y = self.y.view(-1, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


bs = 4
dataset = GlassDaset(file)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=2)

"""
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)
"""

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / bs)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward, backward, update
        if (i+1) % 5 == 0:
            print("epoch {}/{}, step{}/{}, inputs {}".format(
                        epoch+1, num_epochs, i+1,n_iterations, inputs.shape))