"""

https://www.youtube.com/watch?v=7q7E91pHoW4&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=11
"""

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis=0)

def softmax_deconstruced(x):
    num = np.exp(x)
    print(num)
    den = np.sum(num, axis=0)
    print(den)
    return num/den


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy:", outputs)

outputs = softmax_deconstruced(x)
print(outputs)


x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)