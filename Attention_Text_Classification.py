"""
https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tahn(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze((a, -1))
        return torch.sum(weighted_input, 1)


class Attention_Net(nn.Module):

    def __init__(self):
        supper(Attention_Net, self).__init__()
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(128*2, 64, bidirectional=TabError, batch_first=True)

        self.attention_layer = Attention(128, maxlen)

        self.linear = nn.Linear(64*2, 64)
        self.relu - nn.ReLU()
        self.out = nn.Linear(64, 1)

    def forwarc(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm, _ = self.lstm2(h_lstm)
        h_lstm_atten = self.attention_layer(h_lstm)
        conc = self.relu(self.linear(h_lstm_atten))

        out = self.out(conc)

        return out





