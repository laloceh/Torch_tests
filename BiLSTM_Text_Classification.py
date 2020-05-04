"""
https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/

In traditional LSTM:
For a sequence of length 4 like “you will never believe”, The RNN cell gives 4 output vectors, which can be concatenated and then used as part of a dense feedforward architecture.

For BiLSTM:
we stack two RNNs in parallel, and hence we get 8 output vectors to append.

Once we get the output vectors, we send them through a series of dense layers and finally a softmax layer to build a text classifier.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix), dtype=torch.float32)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out

    
