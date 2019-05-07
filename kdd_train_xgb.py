# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

from stacked_generalizer import StackedGeneralizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('KDDTrain+.csv', header=None)
testset = pd.read_csv('KDDTest+.csv', header=None)
joined_set = pd.concat([dataset, testset])

"""map_14 = {0: 0, 1: 1, 2: 0}
joined_set[14] = joined_set[14].map(map_14)"""

joined_set = joined_set.drop(19, 1)

#dataset = dataset.apply(lambda col: pd.factorize(col)[0])
factorize_columns = [1, 2, 3, 41]

for col in factorize_columns:
    joined_set[col] = pd.factorize(joined_set[col])[0]

X = joined_set.iloc[:, 0:-2].values
y = joined_set.iloc[:, [-2]].values

for i in range(y.shape[0]):
    if y[i] > 0:
        y[i] = 1
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train = X[0:dataset.shape[0], :]
X_test = X[dataset.shape[0]:, :]
y_train = y[0:dataset.shape[0], :]
y_test = y[dataset.shape[0]:, :]











# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_c = sc.fit_transform(X_train)
X_test_c = sc.transform(X_test)


#X_train_c = X_train
#X_test_c = X_test



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=1)
y_clust_pred = kmeans.fit_predict(X_train_c).reshape(-1, 1)
X_clus = np.append(X_train_c, y_train, axis = 1)
X_clus = np.append(X_clus, y_clust_pred, axis = 1)


splits = [X_clus[X_clus[:,-1]==k] for k in np.unique(X_clus[:,-1])]


classifiers = []
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

index = 0
for xset in splits:
    """base_models = [XGBClassifier(n_estimators=5, n_jobs=-1, reg_alpha=0.5, reg_lambda=0.1),
                   GradientBoostingClassifier(n_estimators=20),
               RandomForestClassifier(max_depth=10, n_jobs=-1, criterion='entropy'),
               ExtraTreesClassifier(max_depth=10, n_estimators=5, n_jobs=-1, criterion='entropy')]

# define blending model
    blending_model = LogisticRegression()
    classifier = StackedGeneralizer(base_models, blending_model, 
	                    n_folds=5, verbose=False)"""
    
    classifier = GradientBoostingClassifier(max_depth = 2)

    if index != 5:
        classifier.fit(xset[:, :-3], xset[:, -2])

    classifiers.append(classifier)
    index += 1
    print(index)


y_pred_clus = kmeans.predict(X_test_c).reshape(-1, 1)
X_test_clus = np.append(X_test_c, y_test, axis = 1)
X_test_clus = np.append(X_test_clus, y_pred_clus, axis = 1)

test_splits = [X_test_clus[X_test_clus[:,-1]==k] for k in np.unique(X_test_clus[:,-1])]

y_test_actual = []
y_pred = []

for i in range(len(test_splits)):
    y_test_actual += list(test_splits[i].T[120])
    if i == 5:
        y_pred += [1] * len(test_splits[i])
        continue
    """il = list(classifiers[i].predict(test_splits[i][:, :-3]))

    if il[0].shape[0] == 1:
        y_pred += [1] * len(il)
    else:
        y_pred += [ils[0] < ils[1] for ils in il]"""
    
    y_pred += list(classifiers[i].predict(test_splits[i][:, :-3]))


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

recall_b_b = recall_score(y_test_actual, y_pred, average = 'weighted')
prec_b_b = precision_score(y_test_actual, y_pred, average = 'weighted')
f1_b_b = f1_score(y_test_actual, y_pred, average = 'weighted')
acc_b_b = accuracy_score(y_test_actual, y_pred)
