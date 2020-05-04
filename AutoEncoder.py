
"""
Implementa un Sistema de Recomendacion usando Stacked Autoencoder
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importar el dataset
training_set = pd.read_csv("ml-100k/u1.base", sep='\t', engine='python', encoding='latin-1', header=None)
training_set = np.array(training_set, dtype='int')
print("training_set shape", training_set.shape)

testing_set = pd.read_csv("ml-100k/u1.test", sep='\t', engine='python', encoding='latin-1', header=None)
test_set = np.array(testing_set, dtype='int')
print("testing_set shape", test_set.shape)


# Obtener el numero de usuarios y peliculas (desde training and testing)

nb_users = int(max( max(training_set[:,0]), max(test_set[:,0]) ))
print("ID max de usuario", nb_users)

nb_movies = int(max( max(training_set[:,1]), max(test_set[:,1]) ))
print("ID max de items", nb_movies)

# Convertir los datos en un array X[u,i] con usuarios u en fila y peliculas i en columna

def convert(data):
    new_data = []
    for id_user in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))

    return  new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos en tensores de Torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Crear la arquitectura de la red neuronal

class SAE(nn.Module):

    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(in_features=nb_movies, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=20)
        self.fc4 = nn.Linear(in_features=20, out_features=nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))    # 20 dimensiones
        x = self.activation(self.fc2(x))    # 10 dimensiones
        x = self.activation(self.fc3(x))    # 20 dimensiones
        out = self.fc4(x)                   # nb_movies dimensiones, sin activacion para no estar entre 0 y 1

        return out


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay = 0.5)

# Entrenar el SAE

nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.              # Contador de cuantos usuarios han calificado al menos 1 pelicula
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)    # Para cambiar el vector de forma a un lote de 1 solo usuario
        target = input.clone()                                  # guarda una refencia de los objetivos
        if torch.sum(target.data > 0) > 0:                          # checar si ha calificado al menos pelicula
            output = sae(input)
            target.require_grad = False                         # Para no calcular nada de este vector
            output[target == 0] = 0                             # solo para predecir las que no tenian rating, ni para contar en el loss, ni gradiente
            loss = criterion(output, target)
            # la media no es sobre todas las peliculas, sino sobre las que realmente ha valorado
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 10e-10) # repartir el error entre todas las peliculas
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()

    print("Epoch: {}, Train Loss: {}".format(epoch, train_loss/s))


# Evaluar conjunto de test en el Autoencoder

test_loss = 0
s = 0.                                          # Contador de cuantos usuarios han calificado al menos 1 pelicula
#sae.eval()
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)    # Para cambiar el vector de forma a un lote de 1 solo usuario.
    # Se uso el training_set para que se activen las neuronas con lo que ya el usuario ha visto.
    # despues se compararan con lo que viene en test para aquellos que no ha visto.
    target = Variable(test_set[id_user]).unsqueeze(0)                                # guarda una refencia de los objetivos
    if torch.sum(target.data > 0) > 0:                          # checar si ha calificado al menos pelicula
        output = sae(input)
        target.require_grad = False                         # Para no calcular nada de este vector
        output[target == 0] = 0                             # solo para predecir las que no tenian rating, ni para contar en el loss, ni gradiente
        loss = criterion(output, target)
        # la media no es sobre todas las peliculas, sino sobre las que realmente ha valorado
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 10e-10) # repartir el error entre todas las peliculas
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.

print("Test Loss: {}".format(test_loss/s))

