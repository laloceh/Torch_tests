"""

https://www.youtube.com/watch?v=9L9jEOwRrCg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17

"""

import torch
import torch.nn as nn


# Metodos para ser usados despues de entrenar los modelos
# por eso el eval

######## METHOD 1

# Complete model
torch.save(model, PATH)

# model class must be defined somewhere
model = torch.load(PATH)
model.eval()


####### METHOD 2

# State dict
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()