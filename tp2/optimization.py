import pandas as pd
import scipy
import scipy.stats as ss
import numpy as np

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

__author__ = "André Pereira, Carlos Lemos, João Barreira, Rafael Braga"
__email__ = "pg38923@alunos.uminho.pt, pg38410@alunos.uminho.pt, a73831@alunos.uminho.pt, a61799@alunos.uminho.pt"


# Imprime no terminal o top-X (por defeito = 3) de resultados da hiper-parametrização do modelo
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (+/- {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}\n".format(results['params'][candidate]))


# Ler dados do ficheiro .csv

data = pd.read_csv("datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Retirar coluna de output 'PSS_Stress' do conjunto de dados (para variável auxiliar)

target = data['PSS_Stress']
data = data.drop('PSS_Stress', 1)

# Substituir NaN por valor da mediana

data = data.fillna(data.median())

# Feature selection "manual" (melhores resultados)

# cols = [9, 12]
# cols_names = list(data.columns[cols])
# data = data[cols_names]

selector = SelectKBest(f_classif, k=5)
selector.fit(data, target)
cols = selector.get_support(indices=True)
cols_names = list(data.columns[cols])

# Standardizar

robust_scaler = preprocessing.RobustScaler()
values_standardized = robust_scaler.fit_transform(data.values)
data = pd.DataFrame(values_standardized, columns=data.columns)

# Hyper-parametrização

clf_model = SVC()

param_dist = {'C': scipy.stats.expon(scale=100),
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': ss.randint(1, 5),
              'gamma': ['auto', 'scale'],
              'coef0': np.random.uniform(low=0.0, high=2.0, size=(10,)),
              'shrinking': [True, False],
              'probability': [True, False],
              'tol': np.random.uniform(low=0.0, high=2.0, size=(10,)),
              'decision_function_shape': ['ovo', 'ovr']}

# Cálculo dos resultados (com hiper-parameterização a 50 iterações e 5-fold cross validation)

rs = RandomizedSearchCV(clf_model, param_distributions=param_dist, n_iter=50, cv=5)
rs.fit(data, target)
report(rs.cv_results_)
