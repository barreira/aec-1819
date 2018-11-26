import pandas as pd
import os
from sklearn import preprocessing
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from matplotlib import pyplot as plt
from math import sqrt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def mkdir(dir_name):
    try:
        os.rmdir(dir_name)
    except OSError:
        pass
    try:
        os.mkdir(dir_name)
    except OSError:
        pass


# Ler dados do ficheiro .csv

data = pd.read_csv("../datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Retirar coluna de output 'PSS_Stress' do conjunto de dados (para variável auxiliar)

target = data['PSS_Stress']
data = data.drop('PSS_Stress', 1)

# Substituir NaN por valor da mediana

data = data.fillna(data.median())

# Gerar gráficos de distribuição das features

mkdir("figures")

nRows = data.shape[0]
nBins = int(round(sqrt(nRows)))

for key in data.keys():
    data.hist(column=key, bins=nBins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("figures/" + fig_name)


print("### Normalization ###\n")

# Normalizar dados

normalizer = preprocessing.Normalizer(norm='l1')  # ou 'l2'
values_normalized = normalizer.transform(data.values)
data = pd.DataFrame(values_normalized, columns=data.columns)

# Feature Selection

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)

cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

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
    # QuadraticDiscriminantAnalysis()  # é o único que não está a dar
]

clf_names = ["Nearest Neighbors", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

# K-Fold Cross-validation e Calcular resultados

for name, clf in zip(clf_names, clf_models):
    cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
    scores = cross_val_score(clf, data, target, cv=cv)
    print(name, "Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


print("\n\n### Standardization ###")

# Standardizar dados

# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# values_standardized = scaler.fit_transform(data.values)
# data = pd.DataFrame(values_standardized, columns=data.columns)

# Quando temos outliers, é melhor usar outro scaler (?)
robust_scaler = preprocessing.RobustScaler()
values_standardized = robust_scaler.fit_transform(data.values)
data = pd.DataFrame(values_standardized, columns=data.columns)

# Feature selection

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

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
    # QuadraticDiscriminantAnalysis()  # é o único que não está a dar
]

clf_names = ["Nearest Neighbors", "SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

# K-Fold Cross-validation e Calcular resultados

for name, clf in zip(clf_names, clf_models):
    cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
    scores = cross_val_score(clf, data, target, cv=cv)
    print(name, "Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# daqui conclui-se que a adaBoost é a melhor tanto com normalization como standardization
