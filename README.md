# scYeast: Deep Learning Framework for Yeast Transcriptome and Proteome Analysis

本项目是一个用于酵母转录组和蛋白质组数据分析的深度学习框架，包括基于转录组数据的预训练、下游任务微调（如压力响应预测）以及基于蛋白质组数据的预训练和下游任务。

## 项目结构

```
scYeast/
├── model_construction/        # 转录组预训练模型构建
├── models/                    # 预训练模型文件
├── pressure_finetune/         # 压力响应微调任务
├── proteome_pretrain/         # 蛋白质组预训练
├── proteome_finetune/         # 蛋白质组下游任务微调
├── fine_tune/                 # 其他微调任务
├── analysis/                  # 分析脚本
├── zero_shot/                 # 零样本学习任务
└── requirements.txt           # Python依赖包
```

## 重要提示：处理超大文件

由于GitHub单文件限制为2GB，项目中的超大文件已经被分割：

### 合并分割的文件

克隆仓库后，需要合并以下分割文件：

#### 1. 合并蛋白质相互作用数据（7.4GB）

```bash
cd proteome_finetune/data
bash merge_pro_interaction.sh
# 合并完成后可以删除分割文件节省空间
# rm pro_interaction.pkl.part_*
```

#### 2. 合并对齐数据（2.4GB）

```bash
cd proteome_pretrain/data
bash merge_alignment_data.sh
# 合并完成后可以删除分割文件节省空间
# rm alignment_data.txt.part_*
```

**注意**：合并过程需要足够的磁盘空间（至少15GB）。

## 环境安装

### 1. 创建Conda环境

```bash
conda create -n scyeast python=3.9
conda activate scyeast
```

### 2. 安装依赖包

```bash
# 基于requirements.txt安装（conda list --export格式）
conda install --file requirements.txt

# 或者使用pip安装主要依赖
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn scipy
pip install matplotlib seaborn
pip install goatools  # 用于GO富集分析
```

### 3. 安装Git LFS（用于大文件管理）

如果需要克隆完整的模型和数据文件：

```bash
# 安装Git LFS
git lfs install

# 克隆仓库（会自动下载LFS文件）
git clone https://github.com/hongzhonglu/scYeast.git
cd scYeast
```

## 预训练模型

项目使用以下预训练模型：

- `models/final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth` - 包含知识图谱的转录组预训练模型
- `models/final_checkpoint_spearman0.5_wo_knowledge_huber_v4.pth` - 不包含知识图谱的转录组预训练模型

## 使用指南

### 1. 压力响应预测任务 (Pressure Fine-tuning)

压力响应预测任务用于预测酵母在不同压力条件下的表型响应。

#### 运行环境准备

```bash
cd pressure_finetune
```

#### 数据文件

- `alignment_pressure_data_1_with_labels.csv` - 压力响应数据集（包含特征和标签）
- `GSE201387_yeast.tpm.tsv` - TPM表达数据
- `pressure_DEGs.xlsx` - 差异表达基因
- `log(tpm+1)d2_pressure_data_1.csv` - 预处理后的表达数据

#### 运行微调训练

##### 使用预训练模型进行微调

```bash
python pressure_fine_tune.py
```

该脚本会：
- 从`alignment_pressure_data_1_with_labels.csv`加载数据
- 加载预训练模型 `../models/final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth`
- 进行10次独立实验以评估模型稳定性
- 保存每次实验的最佳模型到 `pressure_fine_tune_best_model_run_{i}.pth`
- 保存预测结果到 `results/` 目录

##### 不使用预训练模型（从头训练）

```bash
python pressure_no_pretrain.py
```

该脚本用于对比实验，评估预训练的效果。

##### 不使用知识图谱的预训练模型

```bash
python pressure_wo_know.py
```

#### 结果分析

训练完成后，结果文件保存在 `results/` 目录：
- `pressure_test_predictions.npy` - 测试集预测概率
- `pressure_test_true_labels.npy` - 测试集真实标签
- `pressure_fine_tune_best_model_run_*.pth` - 各次实验的最佳模型

使用ROC曲线可视化结果：

```bash
python roc_paint.py
```

#### 其他机器学习基线模型

项目还提供了多种机器学习方法用于对比：

```bash
python svm.py           # 支持向量机
python knn.py           # K近邻
python decisiontree.py  # 决策树
python naive_bayes.py   # 朴素贝叶斯
```

### 2. 蛋白质组预训练 (Proteome Pre-training)

蛋白质组预训练基于转录组预训练模型，进一步使用蛋白质组数据进行训练。

#### 运行环境准备

```bash
cd proteome_pretrain
```

#### 数据文件

数据文件位于 `data/` 目录：
- `alignment_data.txt` - 对齐的蛋白质组数据 (2.4GB, 使用Git LFS)
- `scaled_data_1_pro.pkl` - 标准化后的蛋白质数据 (85MB, 使用Git LFS)
- `gene2vec_dim_200_iter_9spearman0.5.txt` - 基因向量表示
- `filter_proindex.pkl` - 过滤后的蛋白质索引
- `final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth` - 转录组预训练模型

#### 运行预训练

##### 基于转录组预训练模型继续训练

```bash
python "DSgraph_main_v4 _pro_test.py"
```

该脚本会：
- 加载转录组预训练模型
- 使用蛋白质组数据进行继续训练
- 保存训练好的模型为 `final_checkpoint_spearman0.5_w_knowledge_v4pro.pth`
- 输出训练日志到 `output_pro.txt`

##### 不使用转录组预训练（从头训练）

```bash
python "DSgraph_main_v4 _pro_nopretrain.py"
```

用于对比实验，评估转录组预训练的迁移学习效果。

#### 模型架构

蛋白质组预训练使用以下核心模块：
- `DSgraph_pro.py` - 主模型架构
- `embedding_pro.py` - 嵌入层
- `self_attention_pro.py` - 自注意力机制
- `gated_fusion.py` - 门控融合层
- `gene2vec.py` - 基因向量化
- `load_data.py` - 数据加载工具

### 3. 蛋白质组下游任务微调 (Proteome Fine-tuning)

基于蛋白质组预训练模型进行下游任务微调，包括生长速率预测、核糖体占用率(RPF)预测和蛋白质周转率(Turnover)预测。

#### 运行环境准备

```bash
cd proteome_finetune
```

#### 数据文件

数据文件位于 `data/` 目录（使用Git LFS）：
- `pro_interaction.pkl` - 蛋白质相互作用数据 (7.4GB)
- `scaled_data_1_pro.pkl` - 标准化蛋白质数据 (85MB)
- `growth_tensor.pkl` - 生长数据
- `pretrained_features_max.pkl` - 预训练特征（最大值）
- `pretrained_features_min.pkl` - 预训练特征（最小值）
- 其他pkl文件：基因索引、样本索引、过滤器等

#### 运行微调任务

##### 生长速率预测

```bash
python growth_finetune.py
```

##### 核糖体占用率(RPF)预测

```bash
python rpf_finetune.py
```

该脚本使用K折交叉验证，结果保存在 `rpf_kfold_results/`。

##### 蛋白质周转率预测

```bash
python turnover_finetune.py
```

该脚本使用K折交叉验证，结果保存在 `turnover_kfold_results/`。

#### 结果文件

- `rpf_kfold_results/` - RPF预测的K折验证结果
  - `fold_*/model.pkl` - 各折的模型
  - `fold_*/predictions.csv` - 各折的预测结果
  - `round_results.csv` - 汇总结果

- `turnover_kfold_results/` - Turnover预测的K折验证结果（结构同上）

- `ml_model_results/` - 机器学习模型结果
  - `extra_tree_fold_*.pkl` - Extra Trees模型 (169MB, 使用Git LFS)
  - `fold_*_predictions.csv` - 预测结果

## 分析工具

项目提供了多种分析脚本（位于 `analysis/` 目录）：

### GO富集分析

```bash
cd analysis
python GO_weighted_enrich.py  # 加权GO富集分析
python GO_paint.py             # GO富集结果可视化
```

### 性能评估

```bash
python r2_scatter_paint.py     # R²散点图
python loss_compare.py         # 损失函数对比
```

## 零样本学习 (Zero-shot Learning)

位于 `zero_shot/` 目录，包括：
- `gene_embedding_analysis.py` - 基因嵌入分析
- `gene_pair_analysis.py` - 基因对分析
- `GRN.py` - 基因调控网络分析

## 注意事项

### 大文件管理

本项目使用Git LFS管理大文件（>50MB）：
- 模型文件 (*.pth)
- 数据文件 (*.pkl, *.npy)
- 大型文本文件 (alignment_data.txt)

确保已安装Git LFS：
```bash
git lfs install
```

### GPU要求

大部分训练脚本需要GPU支持。如果只有CPU，可以修改脚本中的：
```python
device = 'cuda'  # 改为 'cpu'
```

### 内存要求

蛋白质组相关任务需要较大内存，建议至少32GB RAM。

## 文件路径说明

所有脚本已经更新为使用相对路径，确保在对应目录下运行：
- `pressure_finetune/` 下的脚本需要在该目录运行
- `proteome_pretrain/` 下的脚本需要在该目录运行
- `proteome_finetune/` 下的脚本需要在该目录运行

## 常见问题

### Q: 运行时提示找不到模块？
A: 确保已激活conda环境并安装所有依赖：
```bash
conda activate scyeast
pip install -r requirements.txt  # 如果有pip格式的requirements
```

### Q: Git LFS下载速度慢？
A: 可以使用代理或从发布页面单独下载大文件。

### Q: 显存不足？
A: 可以减小batch_size，或使用CPU运行（速度会变慢）。

## 引用

如果使用本项目，请引用我们的论文（待发表）。

## 许可证

[待添加许可证信息]

## 联系方式

如有问题，请提交GitHub Issue或联系项目维护者。

---

**更新日期**: 2025-11-24

