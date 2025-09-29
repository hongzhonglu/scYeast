import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
import csv
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import time
sys.path.append('../model_construction')
from read_checkpoint import DSgraphNetWO_Output, DSgraphNet_WO_knowlegde


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'label': self.labels[idx]}
        return sample



class AdditionalNetwork(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(AdditionalNetwork, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=200, out_channels=1, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1d2 = nn.Conv1d(in_channels=5812, out_channels=512, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=4, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x = self.conv1d2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        x = self.conv1d3(x)
        return x.squeeze(-1)


class CombinedNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.net1(x)
        x = self.net2(x)
        return x


def read_pressure_fine_tune_data(batch_size):
    age_dataset = pd.read_csv('/home/lulab/scyeast/tmp7-5/alignment_pressure_data_1_with_labels.csv')
    features = age_dataset.iloc[:, :-1]
    labels = age_dataset.iloc[:, -1]
    labels = np.array([ast.literal_eval(item) for item in labels])
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, temp_index in splitter1.split(features, labels):
        features_train, features_temp = features.iloc[train_index], features.iloc[temp_index]
        labels_train, labels_temp = labels[train_index], labels[temp_index]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    for val_index, test_index in splitter2.split(features_temp, labels_temp):
        features_val = features_temp.iloc[val_index]
        labels_val = labels_temp[val_index]
        features_test = features_temp.iloc[test_index]
        labels_test = labels_temp[test_index]

    train_dataset = CustomDataset(features_train, labels_train)
    val_dataset = CustomDataset(features_val, labels_val)
    test_dataset = CustomDataset(features_test, labels_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    os.chdir('../tmp7-5')
    device = 'cuda'
    
    # 存储所有实验的预测结果和真实值
    all_predictions = []
    all_true_labels = []
    
    for i in range(10):
        print(f"开始第 {i+1} 次实验...")
        train_data, val_data, test_data = read_pressure_fine_tune_data(3)
        torch.set_default_dtype(torch.float32)
        net = DSgraphNetWO_Output().to(device).to(torch.float32)
        # net = DSgraphNet_WO_knowlegde().to(device).to(torch.float32)
        modified_state_dict = net.state_dict()
        checkpoint = torch.load('/home/lulab/scyeast/tmp7-5/final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth')
        checkpoint_state_dict = checkpoint['net']
        for name, param in checkpoint_state_dict.items():
            if name in modified_state_dict:
                modified_state_dict[name].copy_(param)
        net.load_state_dict(modified_state_dict)
        net.eval()
        addition_net = AdditionalNetwork().to(device).to(torch.float32)
        combined_model = CombinedNet(net, addition_net)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(combined_model.parameters(), lr=0.0005, weight_decay=0.01)
        max_epoch = 6
        best_val_acc = 0.0
        best_model_state = None
        for epoch in range(max_epoch):
            combined_model.train()
            for batch_id, batch in enumerate(train_data):
                optimizer.zero_grad()
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y = torch.argmax(y, dim=1)
                y_out = combined_model(x)
                loss = loss_func(y_out, y)
                loss.backward()
                optimizer.step()
                print('epoch=%d, batch_id=%d, loss=%f' % (epoch, batch_id, loss))
            combined_model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_data:
                    x = batch['features'].to(device)
                    y = batch['label'].to(device)
                    y_out = combined_model(x)
                    predicted = torch.argmax(y_out, dim=1)
                    true = torch.argmax(y, dim=1)
                    val_correct += (predicted == true).sum().item()
                    val_total += y.shape[0]
            val_acc = val_correct / val_total
            print(f'Validation Accuracy after epoch {epoch}: {val_acc:.4f}')
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_state = combined_model.state_dict()
        combined_model.load_state_dict(best_model_state)
        torch.save(best_model_state, f'pressure_fine_tune_best_model_run_{i+1}.pth')
        
        # 在测试集上进行预测并保存结果
        combined_model.eval()
        test_predictions = []
        test_true_labels = []
        num_correct = 0
        total_num = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_data):
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y_out = combined_model(x)
                
                # 保存原始预测概率（用于ROC曲线）
                test_predictions.extend(y_out.cpu().numpy())
                
                # 保存真实标签
                test_true_labels.extend(y.cpu().numpy())
                
                # 计算准确率
                predicted_indices = torch.argmax(y_out, dim=1)
                true_indices = torch.argmax(y, dim=1)
                correct_predictions = (predicted_indices == true_indices)
                num_correct = num_correct + correct_predictions.sum().item()
                total_num = total_num + y.shape[0]
        
        print(f"第 {i+1} 次实验测试准确率: {num_correct}/{total_num} = {num_correct/total_num:.4f}")
        
        # 将当前实验的结果添加到总结果中
        all_predictions.append(test_predictions)
        all_true_labels.append(test_true_labels)
        
        torch.cuda.empty_cache()
    
    # 保存所有实验的预测结果和真实值
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # 保存为numpy文件
    np.save('pressure_test_predictions.npy', all_predictions)
    np.save('pressure_test_true_labels.npy', all_true_labels)
    
    # 计算并保存平均预测结果
    mean_predictions = np.mean(all_predictions, axis=0)
    mean_true_labels = np.mean(all_true_labels, axis=0)
    
        # for batch_id, batch in enumerate(test_data):
        #     x = batch['features'].to(device)
        #     y = batch['label'].to(device)
        #     y_out = combined_model(x)
        #     predicted_indices = torch.argmax(y_out, dim=1)
        #     true_indices = torch.argmax(y, dim=1)
        #     correct_predictions = (predicted_indices == true_indices)
        #     num_correct = num_correct + correct_predictions.sum().item()
        #     total_num = total_num + y.shape[0]
        # # print(total_num)
        # print(num_correct)
        # torch.cuda.empty_cache()
#%%
# 读取保存的预测结果和真实标签
import numpy as np
all_predictions = np.load('../tmp7-5/results/pressure_test_predictions.npy')
all_true_labels = np.load('../tmp7-5/results/pressure_test_true_labels.npy')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

# 将三维数据在第一个维度上展开，而不是取平均
# all_predictions shape: (n_experiments, n_samples, n_classes)
# all_true_labels shape: (n_experiments, n_samples, n_classes)
y_score = all_predictions.reshape(-1, all_predictions.shape[-1])  # 展开所有实验的预测结果
y_test = all_true_labels.reshape(-1, all_true_labels.shape[-1])  # 展开所有实验的真实标签

# 获取类别数
n_classes = y_score.shape[1]

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算micro-average ROC曲线和ROC面积（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 计算macro-average ROC曲线和ROC面积（方法一）
# 首先聚合所有假阳性率
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# 然后在这个点上插值所有ROC曲线
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# 最后平均并计算AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 绘制所有ROC曲线
lw = 2
plt.figure(figsize=(10, 8))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 打印每个类别的AUC值
print("AUC values for each class:")
for i in range(n_classes):
    print(f"Class {i}: {roc_auc[i]:.3f}")
print(f"Micro-average AUC: {roc_auc['micro']:.3f}")
print(f"Macro-average AUC: {roc_auc['macro']:.3f}")


# %%
