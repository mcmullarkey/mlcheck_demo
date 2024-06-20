from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
import os


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('../BBDD/ysnp.csv')
dataset.drop('Year/Month/Day', inplace=True, axis=1)
#dataset = dataset.replace(np.nan, 0)
dataset = dataset.fillna(dataset.mean())

data = dataset.values
x = data[:, :20]
y = data[:, 0] #Escollim com a y l'atribut Recreation Visits

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


columns = []
for col in dataset.columns:
    columns.append(col)


for i in range(x.shape[1]):
    plt.xlabel(columns[i])
    plt.ylabel("Recreation Visits")
    ax = plt.scatter(x[:,i],y)
    plt.savefig("../Imagenes/dispersion/" + "Attribute_" + str(i) + ".png")
    plt.clf()

    sns.histplot(data = x[:,i], kde = True, line_kws={'linewidth': 2}, color = 'b', alpha = 0.35)
    plt.savefig("../Imagenes/histogramas/" + "Attribute_" + str(i) + ".png")
    plt.clf()

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.savefig("../Imagenes/correlacion/" + "correlacion_dataset" + ".png")
plt.clf()