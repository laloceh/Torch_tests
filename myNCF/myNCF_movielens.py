#%%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

file = "../ml-100k/u.data"

df = pd.read_csv(file)


#%%




#%%
dataset = np.array([ [1, 1, 5.0],
                     [1, 2, 3.5],
                     [1, 3, 4.0],
                     [1, 6, 3.0],
                     [1, 8, 4.5],
                     [1, 10, 5.0],
                     [2, 2, 4.5],
                     [2, 3, 4.5],
                     [2, 4, 2.5],
                     [2, 5, 4.5],
                     [2, 7, 3.5],
                     [2, 9, 3.5],
                     [2, 10, 3.5]])

print(dataset)
print(dataset.shape)
#%%
user_item = dataset[:, :2]
#print(user_item)
users = user_item[:, 0]
print(users)
items = user_item[:, 1]
print(items)
ratings = dataset[:,2]
print(ratings)

#%%

n_users = len(np.unique(users)) # los user_ids comienzan desde 1, asi que hay que agregar 1 row mas, 0 no cuenta
n_items = len(np.unique(items))
print(n_users, n_items)
#%%
embed_dim= 5
bias_dim = 1
ratings_range = (0, 5.5)
#%%
users = torch.from_numpy(users).view(-1, 1).type(torch.LongTensor)
items = torch.from_numpy(items).view(-1, 1).type(torch.LongTensor)
ratings = torch.from_numpy(ratings).view(-1, 1).type(torch.FloatTensor)

print(users.shape)
print(items.shape)
print(ratings.shape)

print(ratings.type())

#%%
print(users)
print(items)
print(ratings)


#%%
class NCF(nn.Module):
    def __init__(self, users_size, items_size, embed_dim ,bias_dim, ratings_range):
        super(NCF, self).__init__()
        self.embed_dim = embed_dim
        self.ratings_range = ratings_range
        self.embedding_user = nn.Embedding(users_size+1, embed_dim)
        self.embedding_item = nn.Embedding(items_size+1, embed_dim)
        self.bias_user = nn.Embedding(users_size+1, bias_dim)
        self.bias_item = nn.Embedding(items_size+1, bias_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users, items):
        u_emb = self.embedding_user(users)
        i_emb = self.embedding_item(items)
        u_bias = self.bias_user(users)
        i_bias = self.bias_item(items)

        dot = torch.bmm(u_emb.view(-1, 1, self.embed_dim), i_emb.view(-1, self.embed_dim, 1))
        dot = dot.squeeze()

        res = dot + u_bias.squeeze() + i_bias.squeeze()
        pred = self.sigmoid(res) * (self.ratings_range[1]-self.ratings_range[0] + self.ratings_range[0])

        return pred

#%%
model = NCF(n_users, n_items, embed_dim, bias_dim, ratings_range)

#%%
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#%%
num_epochs = 100
for epoch in range(num_epochs):
    rating_preds = model(users, items)
    #print(rating_preds.type())
    #print(ratings.type())
    #print()
    #print()
    optimizer.zero_grad()
    loss = criterion(rating_preds, ratings)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print("epoch {}, loss {:.4}".format(epoch+1, loss.item()))

#%%

#%%
userid = 2
item_missig = 9
userid = torch.LongTensor([userid])
itemid = torch.LongTensor([item_missig])

with torch.no_grad():
    rating_pred = model(userid, itemid)
    print("Predicted rating: {}".format(rating_pred))

#%%



