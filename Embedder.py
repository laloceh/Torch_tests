

"""
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

"""

import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


embedder = Embedder(10, 5)
x = torch.tensor([0,1,0], dtype=torch.long)
print(x.shape)
embedding = embedder(x)
print(embedding)
print(embedding.shape)
