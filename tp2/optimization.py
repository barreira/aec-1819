import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


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

normalizer = preprocessing.Normalizer(norm='l1')  # ou 'l2'
values_normalized = normalizer.transform(data.values)
data = pd.DataFrame(values_normalized, columns=data.columns)

# Feature selection

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

# Criar melhor modelo (AdaBoost)

clf_model = AdaBoostClassifier(RandomForestClassifier(max_depth=200, criterion="entropy"), n_estimators=50)

# K-Fold Cross-validation e Calcular resultados

cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
scores = cross_val_score(clf_model, data, target, cv=cv)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
