import pandas as pd
from sklearn.metrics import r2_score

# 假设你的csv文件名为"your_file.csv"
df = pd.read_csv(r"C:\Users\Wenb1n\Desktop\R2_fig.csv")

# 假设真实标签在'y_true'这一列，预测输出在'y_pred'这一列
y_true = df['label']
y_pred = df['with']

# 计算R2分数
r2 = r2_score(y_true, y_pred)

print(f"R2分数：{r2:.4f}")


