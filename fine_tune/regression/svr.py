import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import csv


# os.chdir('../data')
data = pd.read_csv('processed_yeast9_pnas_data_ori.csv')

# 初始化保存结果的CSV文件
ml_results_csv = 'ml_results_SVR.csv'
ml_predictions_csv = 'ml_predictions_SVR.csv'

# 创建结果CSV文件头
with open(ml_results_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Experiment_ID', 'MSE', 'R2'])

# 创建预测结果CSV文件头
with open(ml_predictions_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Experiment_ID', 'Sample_ID', 'True_Label', 'Predicted_Label'])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# 保存结果
with open(ml_results_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SVR', mse, r2])

# 保存预测结果
with open(ml_predictions_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    for sample_id, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
        writer.writerow([sample_id, true_label, pred_label])

print(f"\n实验完成！")
print(f"结果保存在: {ml_results_csv}")
print(f"预测结果保存在: {ml_predictions_csv}")
