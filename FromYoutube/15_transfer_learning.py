"""

https://www.youtube.com/watch?v=K0lWSB2QoIQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=15

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Option 1: Create a new layer and train everything again
# Este modelo solo tiene un FC al final
model = models.resnet18(pretrained=True)
print(model)

num_ftrs = model.fc.in_features

# modificamos la ultima FC layer
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Podemos usar un scheduler para el lr
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gramma = 0.1)

# Option 2: Freeze previous layers, only train last layer
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False # Freeze layers

num_ftrs = model.fc.in_features

# modificamos la ultima FC layer
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Podemos usar un scheduler para el lr
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gramma = 0.1)
