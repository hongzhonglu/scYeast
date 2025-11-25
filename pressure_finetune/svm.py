from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
import os
import pandas as pd
import numpy as np
import ast

os.chdir('../data')

data = pd.read_csv('alignment_pressure_data_1_with_labels.csv')
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
labels = np.array([ast.literal_eval(item) for item in labels])
labels = np.argmax(labels, axis=1)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in splitter.split(features, labels):
    features_train, features_test = features.iloc[train_index], features.iloc[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
model = svm.SVC(C=1, kernel='rbf', gamma=10, decision_function_shape='ovr')
model.fit(features_train, labels_train)
train_score = model.score(features_train, labels_train)
test_score = model.score(features_test, labels_test)
print(train_score)
print(test_score)