"""
https://www.youtube.com/watch?v=X_QOZEko5uE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=10

"""


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sys

file = "glass.data"


class GlassDaset(Dataset):

    def __init__(self, file, transform=None):
        xy = pd.read_csv(file, header=None)
        self.X = xy.iloc[:, :-1].values
        self.y = xy.iloc[:, -1].values
        self.y = self.y.reshape(-1,1)
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor():
    def __call__(self, sample):
        inputs, labels = sample
        X = torch.from_numpy(inputs)
        y = torch.from_numpy(labels)
        y = y.view(-1, 1)

        return (X,y)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = GlassDaset(file, transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(features.shape)
print(labels)
print(labels.shape)
print(type(features), type(labels))

dataset = GlassDaset(file, transform=None)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(3)])
dataset = GlassDaset(file, transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

bs=3
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(3)])
dataset = GlassDaset(file, transform=composed)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=2)
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)
print(labels.shape)







