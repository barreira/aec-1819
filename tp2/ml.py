import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from matplotlib import pyplot as plt
from math import sqrt

from sklearn.svm import SVC


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

# Gerar gráficos de distribuição das features

mkdir("figures")

nrows = data.shape[0]
nbins = int(round(sqrt(nrows)))

for key in data.keys():
    data.hist(column=key, bins=nbins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("figures/" + fig_name)
    # plt.show()
    # print(key)

# Normalização

y = data['PSS_Stress']

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

data['PSS_Stress'] = y

# Modelo e K-Fold Cross Validation

svm = SVC(kernel='linear', C=1)
svm.fit(data, data['PSS_Stress'])

label_encoding = preprocessing.LabelEncoder()
training_scores_enc = label_encoding.fit_transform(data['PSS_Stress'])

cv = KFold(n_splits=10)
scores = cross_val_score(svm, data, training_scores_enc, cv=cv)
print(scores)

'''
- Normalização vs standardização
- Normalizar coluna output?
- Utilizar coluna de output no treino?
- Resultados?
'''
