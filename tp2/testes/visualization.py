import os
import pandas as pd

from sklearn import preprocessing
from matplotlib import pyplot as plt
from math import sqrt


def mkdir(dir_name):
    '''
    Cria pasta cujo path é recebido como parâmetro.

    :param dir_name: Path para a pasta a ser criada
    '''
    try:
        os.rmdir(dir_name)
    except OSError:
        pass
    try:
        os.mkdir(dir_name)
    except OSError:
        pass


# Ler dados do ficheiro .csv

data = pd.read_csv("datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Imprimir informação sobre dataset para ficheiro

pd.options.display.max_columns = 2000
print(data.describe(), file=open("dataset_description.txt", 'w'))

# Colocar dados na mesma escala para podermos comparar gráficos das distribuições

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
values_standardized = scaler.fit_transform(data.values)
data = pd.DataFrame(values_standardized, columns=data.columns)

# Gerar gráficos de distribuição das features

mkdir("dist")

nRows = data.shape[0]
nBins = int(round(sqrt(nRows)))

for key in data.keys():
    data.hist(column=key, bins=nBins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("dist/" + fig_name)
