from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

__author__ = "André Pereira, Carlos Lemos, João Barreira, Rafael Braga"
__email__ = "pg38923@alunos.uminho.pt, pg38410@alunos.uminho.pt, a73831@alunos.uminho.pt, a61799@alunos.uminho.pt"

# Ler dados do ficheiro .csv

data = pd.read_csv("datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Retirar coluna de output 'PSS_Stress' do conjunto de dados (para variável auxiliar)

target = data['PSS_Stress']
data = data.drop('PSS_Stress', 1)

# Missing Data Filtering

print(data.isnull().any(axis=1).sum())  # número de registos que possuem pelo menos um valor 'NaN'

data = data.fillna(data.median())  # substituir NaN por valor da mediana
# data = data.fillna(data.mean())  # substituir NaN por valor da média
# data = data.dropna()  # descartar registos que possuem NaN

# Feature selection

selector = SelectKBest(f_classif, k=5)  # teste para k = 1..10
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

for idx, (ci, cn) in enumerate(zip(cols, cols_names)):
    print("*" * (len(cols) - idx) + " " * idx, ci, cn)

data = data[cols_names]

# Comparar resultados entre MinMaxScaler e RobustScaler:

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler = preprocessing.RobustScaler()

values_standardized = scaler.fit_transform(data.values)
data = pd.DataFrame(values_standardized, columns=data.columns)

clf_model = SVC()

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25)
clf_model.fit(data_train, target_train)
score = clf_model.score(data_test, target_test)
print(score)
