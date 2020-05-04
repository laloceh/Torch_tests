"""
https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1,2,3,5]      # this works like n-grams
        num_filters = 36

        self.embedding = nn.Embedding(max_features, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(Ks)*num_filters, 1)


    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [ F.relu(conv(x)).squeeze(3) for conv in self.convs1 ]
        x = [ F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)

        return logit




