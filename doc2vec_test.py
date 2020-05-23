"""

https://github.com/inejc/paragraph-vectors/blob/master/paragraphvec/models.py


"""
import torch
import torch.nn as nn


vec_dim = 5
num_docs = 2
num_words = 10

# Paragraph matrix
_D = nn.Parameter(torch.randn(num_docs, vec_dim), requires_grad=True)

# Word matrix
_W = nn.Parameter(torch.rand(num_words, vec_dim), requires_grad=True)

# Output layer parameters
_O = nn.Parameter(torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

print(_D)
print(_W)
print(_O)


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
doc_ids = [0]
context_ids = [0,1]#torch.LongTensor([0])
print(context_ids)
print("!!!!!!!!!!!!!!")
print(_W[context_ids,:])

print("~~~~~~~~~~~~~~~")
_a = torch.sum(_W[context_ids,:], dim=0)
print("_a")
print(_a)
print("_b")
_b = _D[doc_ids,:]
print(_b)
x = torch.add(_b, _a)
print(x)

print("$$$$$$$$$$$$$$$$$$$$")
target_noise_ids = [2]
print(_O.shape)
print(x.shape)
z = torch.bmm( x.unsqueeze(0), _O[:, target_noise_ids].unsqueeze(0))
print(z)


print("#####################")
index = [1]
print(f"Getting the document {index} embedding")

print(_D[index,:].data.tolist())