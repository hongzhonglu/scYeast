import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os


os.chdir('../data')

data = pd.read_csv('alignment_pressure_data_1_with_labels.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].apply(ast.literal_eval).values
y = np.array([np.array(label) for label in y])

y_labels = np.argmax(y, axis=1)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in splitter.split(X, y_labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_labels[train_index], y_labels[test_index]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
print(f"训练准确率: {train_score}")
print(f"测试准确率: {test_score}")
