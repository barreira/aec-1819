import pandas as pd
import os
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from math import sqrt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def mkdir(name):
    try:
        os.rmdir(name)
    except OSError:
        pass
    try:
        os.mkdir(name)
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

nrows = data.shape[0]
nbins = int(round(sqrt(nrows)))

for key in data.keys():
    data.hist(column=key, bins=nbins)
    fig_name = "dist-" + key + ".png"
    plt.savefig("figures/" + fig_name)
    # plt.show()

# Normalizar dados

# values = data.values
# normalizer = preprocessing.Normalizer(norm='l1')  # ou 'l2'
# values_normalized = normalizer.transform(values)
# data = pd.DataFrame(values_normalized, columns=data.columns)

# Standardizar dados

values = data.values
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
values_standardized = scaler.fit_transform(values)
data = pd.DataFrame(values_standardized, columns=data.columns)

# # Quando temos outliers, é melhor usar outro scaler (?)
# values = data[['averageExcessOfDistanceBetweenClicks']].values
# robust_scaler = preprocessing.RobustScaler()
# values_standardized = robust_scaler.fit_transform(values)
# # data = pd.DataFrame(values_standardized, columns=data.columns)
# print(values_standardized)

# Feature selection

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

# Modelos, Treino (K-Fold Cross Validation) e Resultados

clf = KNeighborsClassifier()
# clf = SVC()
# clf = GaussianProcessClassifier()
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()
# clf = MLPClassifier()
# clf = AdaBoostClassifier()
# clf = GaussianNB()
# clf = QuadraticDiscriminantAnalysis()  # é o único que não está a dar

scores = cross_val_score(clf, data, target, cv=5)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

# clf_models = [
#     KNeighborsClassifier(),
#     SVC(),
#     GaussianProcessClassifier(),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     MLPClassifier(),
#     AdaBoostClassifier(),
#     GaussianNB()
#     # QuadraticDiscriminantAnalysis()  # é o único que não está a dar
# ]
#
# clf_names = ["Nearest Neighbors", "SVM", "Gaussian Process", "Decision Tree", "Random Forest",
#              "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
#
# for name, clf in zip(clf_names, clf_models):
#     scores = cross_val_score(clf, data, target, cv=5)
#     print(name, "Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

'''
- resultados muito baixos
- data['averageExcessOfDistanceBetweenClicks'] tem valores muito pequenos por causa dos outliers
- Normalizar coluna output?
'''
