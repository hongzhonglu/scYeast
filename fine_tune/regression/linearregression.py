import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 引入线性回归模型
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. 设置文件路径和加载数据
os.chdir('../data')  # 修改为你自己的路径
data = pd.read_csv('processed_yeast9_pnas_data_ori.csv')  # 替换为你的文件名

# 2. 提取特征和标签
X = data.iloc[:, :-1].values  # 假设标签在最后一列，特征在前面
y = data.iloc[:, -1].values  # 标签是最后一列

# 3. 标准化特征数

# 4. 切分数据集：训练集 80%，测试集 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 创建并训练线性回归模型
model = LinearRegression()  # 使用线性回归模型
model.fit(X_train, y_train)

# 6. 在测试集上进行预测
y_pred = model.predict(X_test)

# 7. 评估模型：计算均方误差（MSE）和 R² 分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 8. 输出结果
print("Mean Squared Error:", mse)
print("R² Score:", r2)