import random
import os
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


def input_data_process():
    gene_pairs = []
    with open('gene2vec_input_co_expression.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            gene_pairs.append(row)
    random.shuffle(gene_pairs)
    return gene_pairs


def outputtxt(embeddings_file):
    model = KeyedVectors.load(embeddings_file)
    wordVector = model.wv
    vocabulary, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    wv = np.asarray(wv)
    index = 0
    matrix_txt_file = embeddings_file+".txt"  # gene2vec matrix txt file address
    with open(matrix_txt_file, 'w') as out:
        for i in wv[:]:
            out.write(str(vocabulary[index]) + "\t")
            index = index + 1
            for j in i:
                out.write(str(j) + " ")
            out.write("\n")
    out.close()


def gene2vec_training(gene_pairs):
    dimension = 50
    num_workers = 8
    sg = 1
    max_iter = 10
    window_size = 1
    txtoutput = True
    export_dir = '../embedding_result/'
    for current_iter in range(max_iter):
        if current_iter == 0:
            print('gene2vec dimension ' + str(dimension) + ' iteration ' + str(current_iter) + ' start')
            model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1,
                                           workers=num_workers, iter=1, sg=sg)
            model.save(export_dir + 'gene2vec_dim_' + str(dimension) + '_iter_' + str(current_iter))
            if txtoutput:
                outputtxt(export_dir + 'gene2vec_dim_' + str(dimension) + '_iter_' + str(current_iter))
            print('gene2vec dimension ' + str(dimension) + ' iteration' + str(current_iter) + ' done')
            del model
        else:
            random.shuffle(gene_pairs)
            print('gene2vec dimension ' + str(dimension) + ' iteration ' + str(current_iter) + ' start')
            model = gensim.models.Word2Vec.load(export_dir+"gene2vec_dim_"+str(dimension)+'_iter_'+str(current_iter-1))
            model.train(gene_pairs, total_examples=model.corpus_count, epochs=model.iter)
            model.save(export_dir+'gene2vec_dim_'+str(dimension)+'_iter_'+str(current_iter))
            if txtoutput:
                outputtxt(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter))
            print("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " done")
            del model


def read_gene2vec_result():
    os.chdir('../embedding_result')
    gene2vec_result = {}
    with open('gene2vec_dim_200_iter_9spearman0.5.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            vec = []
            for j in range(1, len(row)):
                vec.append(float(row[j]))
            gene2vec_result[row[0]] = vec
    os.chdir('../data')
    with open('alignment_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            name_list = []
            for j in range(len(row)):
                name_list.append(row[j])
            break
    gene2vec_weight = []
    for i in range(len(name_list)):
        gene2vec_weight.append(gene2vec_result[name_list[i]])
    gene2vec_weight = np.array(gene2vec_weight)
    return gene2vec_weight


if __name__ == '__main__':
    os.chdir('../data')
    gene_pairs = input_data_process()
    gene2vec_training(gene_pairs)
    read_gene2vec_result()


# ### **代码解析：`gene2vec.py`**
#
# ---
#
# #### **1. 功能概述**
# 该代码用于训练基因的嵌入表示（Gene2Vec），核心功能包括：
# - 从共表达数据中加载基因对，并训练 Word2Vec 模型。
# - 保存和导出嵌入向量结果。
# - 读取训练好的基因向量，生成与预处理数据对齐的权重矩阵。
#
# ---
#
# #### **2. 核心函数解析**
#
# ##### **2.1 `input_data_process()`**
# - **功能**：
#   从 `gene2vec_input_co_expression.txt` 读取基因对（每行一个基因），并随机打乱顺序。
# - **实现细节**：
#   - 文件每行为单个基因名称（而非基因对），需注意与 `co_expression_from_data.txt` 的格式差异。
#   - 返回打乱后的基因列表 `gene_pairs`。
#
# ##### **2.2 `outputtxt(embeddings_file)`**
# - **功能**：
#   将训练好的 Word2Vec 模型转换为文本格式，保存基因名称及其对应的嵌入向量。
# - **实现细节**：
#   - 使用 `KeyedVectors` 加载模型，提取词汇表和嵌入矩阵。
#   - 写入格式：每行为 `基因名称\t 向量值`（空格分隔）。
#
# ##### **2.3 `gene2vec_training(gene_pairs)`**
# - **功能**：
#   训练 Word2Vec 模型生成基因嵌入，支持多轮迭代训练。
# - **参数**：
#   - `dimension=50`：嵌入向量维度。
#   - `num_workers=8`：并行线程数。
#   - `sg=1`：使用 Skip-gram 算法（若为 0 则使用 CBOW）。
#   - `max_iter=10`：最大训练轮次。
#   - `window_size=1`：上下文窗口大小。
# - **训练流程**：
#   1. **首次迭代**：初始化模型并训练。
#   2. **后续迭代**：加载上一轮模型继续训练，每次打乱输入数据。
#   3. **保存结果**：每轮保存模型文件（`.model`）和文本格式向量（`.txt`）。
#
# ##### **2.4 `read_gene2vec_result()`**
# - **功能**：
#   读取训练好的基因向量，生成与预处理数据对齐的权重矩阵。
# - **实现细节**：
#   - 从 `gene2vec_dim_200_iter_9spearman0.5.txt` 加载嵌入结果（文件名需与实际一致）。
#   - 根据 `alignment_data.txt` 中的基因顺序，构建 `gene2vec_weight` 矩阵（形状 `[num_genes, 200]`）。
#
# ---
#
# #### **3. 主程序逻辑**
# - **注释部分**：
#   ```python
#   os.chdir('../data')
#   # gene_pairs = input_data_process()
#   # gene2vec_training(gene_pairs)
#   # read_gene2vec_result()
#   ```
#   - 切换到 `../data` 目录，依次执行数据加载、模型训练和结果读取（需取消注释运行）。
#
# ---
#
# #### **4. 关键设计细节**
# - **嵌入训练配置**：
#   - 使用 **Skip-gram 算法**（`sg=1`），适合捕捉基因共现的局部模式。
#   - 窗口大小为 **1**，表示仅考虑相邻基因的共现关系。
#   - 嵌入维度 **50**（可调整 `dimension` 参数）。
# - **迭代训练**：
#   - 每轮迭代后打乱数据顺序，避免模型陷入局部最优。
#   - 支持断点续训（加载历史模型继续训练）。
#
# ---
#
# #### **5. 代码架构图**
# ```plaintext
# gene2vec.py
# ├─ input_data_process()
# │  └─ 加载并打乱基因列表
# │
# ├─ outputtxt(embeddings_file)
# │  └─ 将模型转换为文本格式
# │
# ├─ gene2vec_training(gene_pairs)
# │  └─ 多轮迭代训练 Word2Vec 模型
# │
# ├─ read_gene2vec_result()
# │  └─ 读取嵌入结果并生成权重矩阵
# │
# └─ 主程序（注释）
#    └─ 按需调用上述函数
# ```
#
# ---
#
# #### **6. 应用场景**
# - **基因表示学习**：将基因映射为低维向量，用于下游任务（如基因关系预测、分类）。
# - **多轮迭代优化**：通过增量训练逐步提升嵌入质量。
# - **数据对齐**：确保嵌入向量与预处理数据中的基因顺序一致，便于模型输入。