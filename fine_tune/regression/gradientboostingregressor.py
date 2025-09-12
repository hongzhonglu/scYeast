import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
# 1. 设置文件路径和加载数据

os.chdir('../data')  # 修改为你自己的路径
data = pd.read_csv('processed_yeast9_pnas_data_ori.csv')  # 替换为你的文件名
for i in range(10):
    # 2. 提取特征和标签
    X = data.iloc[:, :-1].values  # 假设标签在最后一列，特征在前面
    y = data.iloc[:, -1].values  # 标签是最后一列

    # 4. 切分数据集：训练集 80%，测试集 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,  # 学习率
        max_depth=3,        # 每棵树的最大深度
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # print("\n梯度提升回归:")
    # print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))