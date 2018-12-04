import pandas as pd

# Ler dados do ficheiro .csv

data = pd.read_csv("../datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Missing Data Filtering

print(data.isnull().any(axis=1).sum())  # número de registos que possuem pelo menos um valor 'NaN'

data = data.fillna(data.median())  # substituir NaN por valor da mediana
# data = data.fillna(data.mean())  # substituir NaN por valor da média
# data = data.dropna()             # descartar registos que possuem NaN
