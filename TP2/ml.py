import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("../datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Substituir NaN por valor da média

data = data.fillna(data.median())

# Normalização

# neg_cols = []
#
# for key in data.keys():
#     for item in data[key]:
#         if (item <= 0):
#             neg_cols.append(key)
#             break

# print(neg_cols)

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

# print(data)
