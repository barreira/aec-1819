import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as ss
import numpy as np


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Ler dados do ficheiro .csv
from sklearn.svm import SVC

data = pd.read_csv("../datasets/Dataset_MousePSS.csv", sep=';')

# Remover ExamID e StudyID

data = data.drop('ExamID', 1)
data = data.drop('StudyID', 1)

# Retirar coluna de output 'PSS_Stress' do conjunto de dados (para variável auxiliar)

target = data['PSS_Stress']
data = data.drop('PSS_Stress', 1)

# Substituir NaN por valor da mediana

data = data.fillna(data.median())

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

# Hyper-parameterização

clf_model = SVC()

param_dist = {'C': ss.randint(1,10),
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': ss.randint(1, 20),
              'gamma': ['auto', 'scale'],
              # 'coef0': np.random.uniform(low=0.0, high=20.0, size=(10,)),
              'coef0': ss.randint(1,10),
              'shrinking': [True, False],
              'probability': [True, False],
              'tol': ss.uniform,
              'decision_function_shape' : ['ovo', 'ovr']}

n_iter = 20
rs = RandomizedSearchCV(clf_model, param_distributions=param_dist, n_iter=n_iter, cv=5)
rs.fit(data, target)
report(rs.cv_results_)
