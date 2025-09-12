import time

import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from itertools import combinations


def load_gene_names(file_path='relation_matrix.txt'):
    gene_names = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'Q0045':  # 特定标识符
                gene_names = row
                break

    assert len(gene_names) == 5812, f"基因名称数量不匹配，期望5812个，实际{len(gene_names)}个"
    return gene_names


def process_age_stage(stage, base_path, output_base_path, gene_names):
    """
    处理特定年龄阶段的基因嵌入数据
    """
    # 确保输出路径存在
    os.makedirs(output_base_path, exist_ok=True)

    # 获取该年龄阶段的所有嵌入数据文件
    embedding_files = [
        f for f in os.listdir(base_path)
        if f.startswith(f'age_gene_embedding_output_{stage}')
           and f.endswith('.npy')
    ]

    # 处理每个嵌入数据文件
    for embedding_file in tqdm(embedding_files, desc=f"Processing {stage} stage"):
        # 完整文件路径
        full_embedding_path = os.path.join(base_path, embedding_file)

        # 加载基因嵌入数据
        gene_embeddings = np.load(full_embedding_path)

        # 确保数据形状正确 (5812, 200)
        assert gene_embeddings.shape == (5812, 200), f"Unexpected shape for {embedding_file}"

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(gene_embeddings)

        # 保存相似度矩阵为CSV
        similarity_matrix_df = pd.DataFrame(
            similarity_matrix,
            columns=gene_names,
            index=gene_names
        )
        similarity_matrix_filename = f'gene_similarity_matrix_{embedding_file.replace(".npy", ".csv")}'
        similarity_matrix_path = os.path.join(output_base_path, similarity_matrix_filename)
        similarity_matrix_df.to_csv(similarity_matrix_path)

        # 可选：生成每个基因的top相似基因信息（如果需要）
        top_similar_genes = []
        for i in range(len(gene_names)):
            # 获取排序后的相似基因索引（不包括自身）
            similar_indices = np.argsort(-similarity_matrix[i, :])
            similar_indices = similar_indices[similar_indices != i]

            top_similar_genes.append({
                'Gene_Name': gene_names[i],
                'Top_1000_Similar_Genes': [gene_names[idx] for idx in similar_indices[:1000]],
                'Top_1000_Similarities': list(similarity_matrix[i, similar_indices[:1000]])
            })

            # 保存top相似基因信息
        top_similar_df = pd.DataFrame(top_similar_genes)
        top_similar_filename = f'top_similar_genes_{embedding_file.replace(".npy", ".csv")}'
        top_similar_path = os.path.join(output_base_path, top_similar_filename)
        top_similar_df.to_csv(top_similar_path, index=False)


def extract_high_similarity_gene_pairs(folder_path, similarity_threshold=0.9):
    """
    提取不同样本中高相似度的基因对
    """
    # 按生长状态分组的相似度矩阵文件
    stages = ['early age', 'late age', 'young']
    stage_gene_pairs = {}

    for stage in stages:
        # 找出该生长状态下的所有相似度矩阵文件
        stage_files = [
            f for f in os.listdir(folder_path)
            if f.startswith(f'gene_similarity_matrix_age_gene_embedding_output_{stage}')
               and f.endswith('.csv')
        ]

        # 存储该生长状态下所有样本的高相似度基因对
        stage_sample_pairs = []

        for file in stage_files:
            start_time = time.time()
            file_path = os.path.join(folder_path, file)

            similarity_matrix = pd.read_csv(file_path, index_col=0).values
            gene_names = pd.read_csv(file_path, index_col=0).columns

            high_sim_indices = np.argwhere(
                (similarity_matrix >= similarity_threshold) &
                (np.triu(np.ones_like(similarity_matrix), k=1).astype(bool))
            )

            high_similarity_pairs = {
                tuple(sorted((gene_names[i], gene_names[j])))
                for i, j in high_sim_indices
            }

            stage_sample_pairs.append(high_similarity_pairs)

            end_time = time.time()
            print(f"{file} 处理时间: {end_time - start_time:.2f}s")

            # 计算所有样本的交集
        stage_gene_pairs[stage] = set.intersection(*stage_sample_pairs)

    return stage_gene_pairs


def compare_gene_pairs_across_stages(stage_gene_pairs):
    """
    深入比较不同生长状态间的基因对差异
    """
    stages = list(stage_gene_pairs.keys())
    comparison_results = {}

    # 两两比较
    for i in range(len(stages)):  # 0,1,2
        for j in range(i + 1, len(stages)):
            stage1, stage2 = stages[i], stages[j]

            # 计算基本交集和并集
            intersection = stage_gene_pairs[stage1] & stage_gene_pairs[stage2]
            union = stage_gene_pairs[stage1] | stage_gene_pairs[stage2]

            # 差异分析
            unique_to_stage1 = stage_gene_pairs[stage1] - stage_gene_pairs[stage2]
            unique_to_stage2 = stage_gene_pairs[stage2] - stage_gene_pairs[stage1]

            # # 计算三状态特异性基因对
            # all_stages = set.union(*stage_gene_pairs.values())
            # stage_sets = [stage_gene_pairs[stage] for stage in stages]

            # 找出仅在一个状态存在的基因对
            unique_gene_pairs = {}
            for k, current_stage in enumerate(stages):
                other_stages = stages[:k] + stages[k + 1:]
                unique_to_current_stage = stage_gene_pairs[current_stage] - set.union(
                    *[stage_gene_pairs[s] for s in other_stages])
                unique_gene_pairs[current_stage] = unique_to_current_stage

            comparison_results[f'{stage1}__vs__{stage2}'] = {
                # 基本统计
                'intersection_count': len(intersection),
                'union_count': len(union),
                'jaccard_index': len(intersection) / len(union) if len(union) > 0 else 0,

                # 各状态特异性基因对
                'unique_to_stage1_count': len(unique_to_stage1),
                'unique_to_stage2_count': len(unique_to_stage2),

                # 详细差异信息
                'unique_to_stage1': unique_to_stage1,
                'unique_to_stage2': unique_to_stage2,

                # 三状态特异性分析
                'stage_specific_gene_pairs': unique_gene_pairs
            }

    return comparison_results


def save_gene_pairs_comparison(stage_gene_pairs, comparison_results, output_folder):
    """
    保存详细的基因对比较结果
    """
    os.makedirs(output_folder, exist_ok=True)

    # 保存每个生长状态的基因对
    for stage, gene_pairs in stage_gene_pairs.items():
        pairs_df = pd.DataFrame(list(gene_pairs), columns=['Gene1', 'Gene2'])
        output_path = os.path.join(output_folder, f'{stage}_high_similarity_gene_pairs.csv')
        pairs_df.to_csv(output_path, index=False)
        print(f"{stage}：找到 {len(gene_pairs)} 个高相似度基因对")

        # 保存比较结果的详细信息
    for comparison, result in comparison_results.items():
        # 1. 创建比较结果摘要DataFrame
        summary_data = {
            'Metric': [
                'jiao',
                'bin',
                'Jaccard',
                f'仅在{comparison.split("_vs_")[0]}中的基因对数量',
                f'仅在{comparison.split("_vs_")[1]}中的基因对数量'
            ],
            'Value': [
                result['intersection_count'],
                result['union_count'],
                result['jaccard_index'],
                result['unique_to_stage1_count'],
                result['unique_to_stage2_count']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_output_path = os.path.join(output_folder, f'{comparison}_summary.csv')
        summary_df.to_csv(summary_output_path, index=False)

        # 2. 保存特异性基因对
        for stage, unique_pairs in result['stage_specific_gene_pairs'].items():
            unique_pairs_df = pd.DataFrame(list(unique_pairs), columns=['Gene1', 'Gene2'])
            unique_pairs_output_path = os.path.join(output_folder, f'{comparison}_{stage}_unique_gene_pairs.csv')
            unique_pairs_df.to_csv(unique_pairs_output_path, index=False)

            # 打印总体比较结果摘要
    print("\n生长状态间基因对比较结果:")
    for comparison, result in comparison_results.items():
        print(f"{comparison}:")
        print(f"  交集基因对数量: {result['intersection_count']}")
        print(f"  并集基因对数量: {result['union_count']}")
        print(f"  Jaccard相似性指数: {result['jaccard_index']:.4f}")
        print(f"  特异性基因对:")
        for stage, unique_pairs in result['stage_specific_gene_pairs'].items():
            print(f"    {stage}: {len(unique_pairs)} 个")


def verify_matrix_properties(folder_path):
    """
    验证相似度矩阵的属性

    参数:
    folder_path (str): 存储相似度矩阵的文件夹路径
    """
    # 按生长状态分组的相似度矩阵文件
    stages = ['early age', 'late age', 'young']

    for stage in stages:
        # 找出该生长状态下的所有相似度矩阵文件
        stage_files = [
            f for f in os.listdir(folder_path)
            if f.startswith(f'gene_similarity_matrix_age_gene_embedding_output_{stage}')
               and f.endswith('.csv')
        ]

        print(f"\n{stage} 阶段的矩阵验证:")
        for file in stage_files:
            file_path = os.path.join(folder_path, file)
            similarity_df = pd.read_csv(file_path, index_col=0)

            # 验证矩阵维度
            print(f"文件 {file}:")
            print(f"矩阵维度: {similarity_df.shape}")

            # 验证对称性
            is_symmetric = np.allclose(similarity_df, similarity_df.T, atol=1e-8)
            print(f"是否对称: {is_symmetric}")

            # 验证对角线
            diagonal_values = similarity_df.values.diagonal()
            print(f"对角线值范围: {diagonal_values.min()} - {diagonal_values.max()}")

            # 抽样检查
            sample_row = np.random.randint(0, similarity_df.shape[0])
            sample_col = np.random.randint(0, similarity_df.shape[0])
            print(f"随机抽查位置 [{sample_row}, {sample_col}] 的值: {similarity_df.iloc[sample_row, sample_col]}")


def main():
    # 配置路径 - 请根据实际情况修改
    # os.chdir('C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data')
    base_path = 'C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data/age_gene_embedding_output'  # 基因嵌入数据路径
    output_base_path = 'C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data/similarity_results'  # 输出路径
    gene_names_path = 'C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data/processed_data.txt'

    gene_names = load_gene_names(gene_names_path)

    # 年龄阶段
    stages = ['early age', 'late age', 'young']

    # # 处理每个年龄阶段
    # for stage in stages:
    #     process_age_stage(
    #         stage,
    #         base_path,
    #         output_base_path,
    #         gene_names
    #     )
    #
    # print("余弦相似度分析完成！")


    # 配置路径
    similarity_matrix_folder = output_base_path
    output_analyse_folder = 'C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data/high_similarity_gene_pairs'


    # # 验证矩阵属性
    # verify_matrix_properties(similarity_matrix_folder)

    # 提取高相似度基因对
    stage_gene_pairs = extract_high_similarity_gene_pairs(
        similarity_matrix_folder,
        similarity_threshold=0.9
    )

    # 比较不同生长状态的基因对
    comparison_results = compare_gene_pairs_across_stages(stage_gene_pairs)

    # 保存结果
    save_gene_pairs_comparison(stage_gene_pairs, comparison_results, output_analyse_folder)


if __name__ == '__main__':
    main()
