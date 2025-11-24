import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import torch

os.chdir('../data')
device = torch.device('cuda')
data = pd.read_csv('alignment_pressure_data_1_with_labels.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].apply(ast.literal_eval).values
y = np.array([np.array(label) for label in y])
y_labels = np.argmax(y, axis=1)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in splitter.split(X, y_labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_labels[train_index], y_labels[test_index]

# clf = RandomForestClassifier()
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"训练准确率: {train_score}")
print(f"测试准确率: {test_score}")