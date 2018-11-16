import pandas as pd
import os
from sklearn import preprocessing
from matplotlib import pyplot as plt
from math import sqrt


def mkdir(name):
    try:
        os.rmdir(name)
    except OSError:
        pass
    try:
        os.mkdir(name)
    except OSError:
        pass


data = pd.read_csv("../datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Substituir NaN por valor da mediana

data = data.fillna(data.median())

# Visualizar gráficos de distribuição das features

mkdir("figures")

nrows = data.shape[0]
nbins = int(round(sqrt(nrows)))

for key in data.keys():
    data.hist(column=key, bins=nbins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("figures/" + fig_name)

# Normalização

for key in data.keys():
    # if key in neg_cols:
    #     input_data = data[[key]].values.astype(float)
    #     data_normalized = preprocessing.normalize(input_data)
    #     data[key] = pd.DataFrame(data_normalized)
    # else:
    input_data = data[[key]].values.astype(float)
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data_scaled = data_scaler.fit_transform(input_data)
    data[key] = pd.DataFrame(data_scaled)

