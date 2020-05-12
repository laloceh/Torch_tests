"""
This is for a CUSTOM dataloder

also used:
https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import torch.nn.functional as F

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
#######################################

dataset = DiabetesDataset()
train_loader = DataLoader( dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

#########################################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.linear2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        out = self.linear3(x)
        return out

model = Model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def binary_accuracy(y_pred, y):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y).sum().float()
    acc = correct_results_sum/y.shape[0]
    acc = torch.round(acc * 100)
    return acc

for epoch in range(10):
    epoch_loss = 0
    epoch_acc = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        #inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        #print(epoch, loss.item())
        acc = binary_accuracy(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(epoch, epoch_loss/len(train_loader), epoch_acc/len(train_loader))










