

import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)

# train your model

################# LAZY METHOD

FILE = "model.pth"
torch.save(model, FILE)

# Load it again
model2 = torch.load(FILE)
model2.eval()
for param in model2.parameters():
    print(param)


############# BEST METHOD
FILE_ = "model_.pth"
torch.save(model.state_dict(), FILE_)

# load it
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE_))
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)

print(model.state_dict())


