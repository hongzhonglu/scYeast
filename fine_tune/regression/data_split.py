import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def create_and_save_data_splits(data_file, num_experiments=20):
    """
    创建并保存数据集划分
    """
    # 加载原始数据
    data = pd.read_csv(data_file)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 创建保存划分数据的文件夹
    splits_dir = 'data_splits'
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    print(f"创建 {num_experiments} 个数据集划分...")

    for i in range(num_experiments):
        print(f"创建划分 {i + 1}/{num_experiments}")

        # 第一次划分：训练集 + 临时集 (80% + 20%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, random_state=i
        )

        # 第二次划分：验证集 + 测试集 (10% + 10%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=i
        )

        # 保存训练集
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.to_csv(f'{splits_dir}/train_split_{i}.csv', index=False)

        # 保存验证集
        val_data = pd.concat([X_val, y_val], axis=1)
        val_data.to_csv(f'{splits_dir}/val_split_{i}.csv', index=False)

        # 保存测试集
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv(f'{splits_dir}/test_split_{i}.csv', index=False)

        print(f"  训练集大小: {len(train_data)}")
        print(f"  验证集大小: {len(val_data)}")
        print(f"  测试集大小: {len(test_data)}")

    print(f"\n所有数据集划分保存完成！保存路径: {splits_dir}/")

    # 保存划分信息
    split_info = {
        'num_experiments': num_experiments,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'original_data_file': data_file,
        'total_samples': len(data)
    }

    import json
    with open(f'{splits_dir}/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=4)

    return splits_dir


if __name__ == '__main__':
    # 创建数据集划分
    data_file = 'processed_yeast9_pnas_data_ori.csv'
    splits_dir = create_and_save_data_splits(data_file, num_experiments=20)

