from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import os
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np


__author__ = "André Pereira, Carlos Lemos, João Barreira, Rafael Braga"
__email__ = "pg38923@alunos.uminho.pt, pg38410@alunos.uminho.pt, a73831@alunos.uminho.pt, a61799@alunos.uminho.pt"


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


def normalizacao_e_resultados(data):
    print("### Normalization ###")

    # Normalizar dados

    normalizer = preprocessing.Normalizer(norm='l1') # 'l2' (mesmos resultados)
    values_normalized = normalizer.transform(data.values)
    data = pd.DataFrame(values_normalized, columns=data.columns)

    # Criar modelos

    clf_models = [
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB()
    ]

    clf_names = ["Nearest Neighbors", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net",
                 "AdaBoost", "Naive Bayes", "QDA"]

    # K-Fold Cross-validation e Calcular resultados

    for name, clf in zip(clf_names, clf_models):
        # cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
        scores = cross_val_score(clf, data, target, cv=5)
        print(name, "Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


def standardizacao_e_resultados(data):
    print("\n### Standardization ###")

    # Standardizar dados

    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # values_standardized = scaler.fit_transform(data.values)
    # data = pd.DataFrame(values_standardized, columns=data.columns)

    # Quando temos outliers, é melhor usar RobustScaler (in doc sklearn)
    robust_scaler = preprocessing.RobustScaler()
    values_standardized = robust_scaler.fit_transform(data.values)
    data = pd.DataFrame(values_standardized, columns=data.columns)

    # Criar modelos

    clf_models = [
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB()
    ]

    clf_names = ["Nearest Neighbors", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net",
                 "AdaBoost", "Naive Bayes", "QDA"]

    # K-Fold Cross-validation e Calcular resultados

    for name, clf in zip(clf_names, clf_models):
        # cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
        scores = cross_val_score(clf, data, target, cv=5)
        print(name, "Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Ler dados do ficheiro .csv

data = pd.read_csv("datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Gerar gráficos de distribuição das features (gráficos não estão à mesma escala: utilizar visualization.py para tal)

mkdir("dist")

nRows = data.shape[0]
nBins = int(round(sqrt(nRows)))  # binning

for key in data.keys():
    data.hist(column=key, bins=nBins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("dist/" + fig_name)

# Imprimir informação sobre dataset para ficheiro

pd.options.display.max_columns = 2000
print(data.describe(), file=open("dataset_description.txt", 'w'))

# Retirar coluna de output 'PSS_Stress' do conjunto de dados (para variável auxiliar)

target = data['PSS_Stress']
data = data.drop('PSS_Stress', 1)

# Missing Data Filtering

# print(data.isnull().any(axis=1).sum())  # número de registos que possuem pelo menos um valor 'NaN'

data = data.fillna(data.median())  # substituir NaN por valor da mediana
# data = data.fillna(data.mean())  # substituir NaN por valor da média
# data = data.dropna()  # descartar registos que possuem NaN

# Feature selection

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

# Comparar resultados:

## Normalizar
normalizacao_e_resultados(data)

## Standardizar
standardizacao_e_resultados(data)
