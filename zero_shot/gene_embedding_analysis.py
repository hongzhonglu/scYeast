# import pandas as pd
# import numpy as np
# import os
#
#
# # 设定高相似度的阈值
# threshold = 0.9  # 这里可以调整阈值，根据您的需求
# os.chdir('C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data')
# # 加载余弦相似度矩阵
# young_similarity_matrix = pd.read_csv("gene_similarity_matrix_age_gene_embedding_output_young.csv", header=None)
# early_age_similarity_matrix = pd.read_csv("gene_similarity_matrix_age_gene_embedding_output_early_age.csv", header=None)
# late_age_similarity_matrix = pd.read_csv("gene_similarity_matrix_age_gene_embedding_output_late_age.csv", header=None)
#
# # 函数：提取高相似度基因
# def extract_high_similarity_genes(similarity_matrix, threshold):
#     high_similarity_genes = {}
#     # 从相似度矩阵中提取高相似度基因（排除对角线）
#     for i in range(similarity_matrix.shape[0]):
#         similar_genes = np.where(similarity_matrix.iloc[i].values > threshold)[0]
#         high_similarity_genes[i] = list(similar_genes)
#     return high_similarity_genes
#
# young_high_genes = extract_high_similarity_genes(young_similarity_matrix, threshold)
# early_age_high_genes = extract_high_similarity_genes(early_age_similarity_matrix, threshold)
# late_age_high_genes = extract_high_similarity_genes(late_age_similarity_matrix, threshold)
#
# # 提取高相似度基因的集合
# young_set = set(young_high_genes.keys())
# early_age_set = set(early_age_high_genes.keys())
# late_age_set = set(late_age_high_genes.keys())
#
# # 计算交集
# intersection_young_early_age = young_set.intersection(early_age_set)
# intersection_young_late_age = young_set.intersection(late_age_set)
# intersection_early_age_late_age = early_age_set.intersection(late_age_set)
#
# # 计算并集
# union_all = young_set.union(early_age_set).union(late_age_set)
#
# # 输出结果
# print("年轻状态下的高相似度基因:", young_set)
# print("早期状态下的高相似度基因:", early_age_set)
# print("晚期状态下的高相似度基因:", late_age_set)
# print("年轻状态与早期状态交集基因:", intersection_young_early_age)
# print("年轻状态与晚期状态交集基因:", intersection_young_late_age)
# print("早期状态与晚期状态交集基因:", intersection_early_age_late_age)
# print("三个状态下的基因并集:", union_all)

import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
def average_similarity_matrices(folder_path, age_state):

    pattern = os.path.join(folder_path, f'gene_similarity_matrix_age_gene_embedding_output_{age_state}_*.csv')

    similarity_files = glob.glob(pattern)

    similarity_matrices = []
    gene_names = None

    # 读取并存储所有矩阵
    for file in similarity_files:
        matrix_df = pd.read_csv(file, index_col=0)

        if gene_names is None:
            gene_names = matrix_df.index.tolist()

            # 确保所有矩阵的基因名称顺序一致
        matrix_df = matrix_df.reindex(index=gene_names, columns=gene_names)

        # 只提取数值矩阵部分
        matrix_values = matrix_df.values
        similarity_matrices.append(matrix_values)

        # 计算平均矩阵
    average_matrix = np.mean(similarity_matrices, axis=0)

    # 转换为DataFrame，保留基因名称
    average_df = pd.DataFrame(average_matrix, index=gene_names, columns=gene_names)

    # 保存平均矩阵
    output_filename = os.path.join(folder_path, f'!!!average_gene_similarity_matrix_{age_state}.csv')
    average_df.to_csv(output_filename)

    print(f"已保存 {age_state} 状态的平均相似度矩阵到 {output_filename}")

    return average_df


def extract_unique_genes(input_file, output_file):
    # 读取基因对文件
    df = pd.read_csv(input_file)

    # 提取所有唯一基因
    unique_genes = set(df['Gene1']).union(set(df['Gene2']))

    # 将唯一基因转换为DataFrame
    unique_genes_df = pd.DataFrame(list(unique_genes), columns=['Gene'])

    # 保存唯一基因到新文件
    unique_genes_df.to_csv(output_file, index=False)

    print(f"从 {input_file} 中提取了 {len(unique_genes)} 个唯一基因")

    return unique_genes


def find_state_specific_genes(gene_sets):
    state_specific_genes = {}
    states = list(gene_sets.keys())

    for i, state in enumerate(states):
        # 创建其他状态的并集
        other_states_genes = set().union(*[gene_sets[other_state] for other_state in states[:i] + states[i + 1:]])

        # 找出仅存在于当前状态的基因
        specific_genes = gene_sets[state] - other_states_genes
        state_specific_genes[state] = specific_genes

    return state_specific_genes


def bin_similarity_values(values):
    """
    对相似度值进行分箱处理
    < 0.3: 0
    0.3-0.8: 0.5
    > 0.8: 1
    """
    binned_values = []
    for val in values:
        if val < 0.5:
            binned_values.append(0)
        elif val <= 0.9:
            binned_values.append(0.5)
        else:
            binned_values.append(1)
    return binned_values


def load_state_specific_genes(folder_path, age_states):
    state_specific_genes = {}
    all_specific_genes = []  # 使用列表保持顺序

    # 定义状态顺序
    ordered_states = age_states

    for state in ordered_states:
        file_path = os.path.join(folder_path, f'E:/lulab电脑/PycharmProjects/scYeast/DSGraph/data/high_similarity_gene_pairs/state_specific_genes_{state}.csv')
        genes_df = pd.read_csv(file_path)
        state_specific_genes[state] = genes_df['Gene'].tolist()

        # 按顺序添加，避免重复
        for gene in genes_df['Gene'].tolist():
            if gene not in all_specific_genes:
                all_specific_genes.append(gene)

    return state_specific_genes, all_specific_genes


def create_combined_heatmaps(folder_path, age_states, all_specific_genes):

    # 存储每个状态的热力图数据
    heatmap_data = {}

    for state in age_states:
        # 加载余弦相似度矩阵
        similarity_matrix_file = os.path.join(folder_path, f'E:/lulab电脑/PycharmProjects/scYeast/DSGraph/data/similarity_results/!!!average_gene_similarity_matrix_{state}.csv')
        similarity_matrix_df = pd.read_csv(similarity_matrix_file, index_col=0)

        # 确保基因存在于矩阵中
        existing_genes = [gene for gene in all_specific_genes if gene in similarity_matrix_df.index]
        print(len(existing_genes), len(similarity_matrix_df.index))
        if not existing_genes:
            print(f"{state} 状态没有找到匹配的基因")
            continue

        # 提取子矩阵
        sub_matrix = similarity_matrix_df.loc[existing_genes, existing_genes]

        # 存储热力图数据
        heatmap_data[state] = sub_matrix

        # 绘制热力图
        plt.figure(figsize=(16, 14))
        sns.heatmap(sub_matrix,
                    cmap='YlGnBu',
                    annot=False,
                    cbar_kws={'label': 'Cosine Similarity'})

        # 保存热力图
        # output_file = os.path.join(folder_path, f'1_heatmap_all_specific_genes_{state}.png')
        # plt.savefig(output_file, dpi=600)
        plt.close()

        # print(f"已保存 {state} 状态的热力图到 {output_file}")

    return heatmap_data


def create_side_by_side_heatmap(heatmap_data, folder_path):
    fig, axes = plt.subplots(1, 3, figsize=(38, 14), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # 确定全局颜色映射的最大最小值
    vmin = min(matrix.values.min() for matrix in heatmap_data.values())
    vmax = max(matrix.values.max() for matrix in heatmap_data.values())
    # 定义区间
    intervals = [0, 0.5, 0.9, 1.5]

    def segment_matrix(matrix, intervals):
        """
        根据指定区间对矩阵进行分区
        """
        segmented_matrix = np.zeros_like(matrix.values)
        aa = [0.26, 0.5, 0.95]
        for i in range(len(intervals) - 1):
            mask = (matrix.values >= intervals[i]) & (matrix.values < intervals[i + 1])
            segmented_matrix[mask] = aa[i]

        return segmented_matrix

    for i, (state, matrix) in enumerate(heatmap_data.items()):
        print(matrix.values.min(), matrix.values.max())
        segmented_matrix = segment_matrix(matrix, intervals)
        # 创建热力图  CMRmap_r  YlGnBu
        im = sns.heatmap(segmented_matrix,
                    cmap=['#FFF0F5', '#87CEFF', '#7A67EE'],
                    ax=axes[i],
                    cbar=False,
                    xticklabels=False,
                    yticklabels=False
                    )
        # 为最后一个图添加颜色棒
        if i == len(heatmap_data) - 1:
            # 在最后一个子图旁边添加颜色棒
            cbar = fig.colorbar(im.collections[0],
                                ax=axes[i],
                                aspect=28,  # 控制颜色棒的长度和宽度比
                                pad=0.08,
                                ticks=[]
                                )  # 控制颜色棒与热力图的距离

    plt.tight_layout(pad=3)
    output_file = os.path.join(folder_path, '1_side_by_side_heatmaps_all_specific_genes.png')
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_stage_similarity_averages(heatmap_data, state_specific_genes, all_specific_genes):
    """
    计算热力图中不同年龄阶段基因区块的平均余弦相似度
    """
    for state in heatmap_data.keys():
        similarity_matrix = heatmap_data[state]

        # 创建基因到阶段的映射
        gene_to_stage = {}
        stage_start_idx = {}
        current_idx = 0

        # 按顺序记录每个阶段的基因及其在矩阵中的位置
        for stage in ['young', 'early age', 'late age']:
            stage_genes_in_matrix = []
            for gene in state_specific_genes[stage]:
                if gene in all_specific_genes and gene in similarity_matrix.index:
                    gene_to_stage[gene] = stage
                    stage_genes_in_matrix.append(gene)

            if stage_genes_in_matrix:
                stage_start_idx[stage] = current_idx
                current_idx += len(stage_genes_in_matrix)

        # 计算各阶段区块的平均相似度
        print(f"\n=== {state} 状态的基因相似度分析 ===")

        # 找到每个阶段基因在矩阵中的索引范围
        stage_ranges = {}
        start_idx = 0

        for stage in ['young', 'early age', 'late age']:
            stage_genes = [gene for gene in state_specific_genes[stage]
                           if gene in similarity_matrix.index]
            if stage_genes:
                end_idx = start_idx + len(stage_genes)
                stage_ranges[stage] = (start_idx, end_idx)
                print(f"{stage} 阶段: 基因数量 {len(stage_genes)}, 矩阵位置 [{start_idx}:{end_idx}]")
                start_idx = end_idx

        # 计算同一阶段内基因的平均相似度
        print(f"\n同一阶段内基因的平均相似度:")
        for stage, (start, end) in stage_ranges.items():
            if end - start > 1:  # 需要至少2个基因才能计算相似度
                stage_matrix = similarity_matrix.iloc[start:end, start:end]
                # 获取上三角矩阵的值（排除对角线）
                upper_triangle = []
                for i in range(stage_matrix.shape[0]):
                    for j in range(i + 1, stage_matrix.shape[1]):
                        upper_triangle.append(stage_matrix.iloc[i, j])

                # if upper_triangle:
                #     avg_similarity = np.mean(upper_triangle)
                #     print(f"  {stage}: {avg_similarity:.4f}")

                if upper_triangle:
                    # 原始平均值
                    original_avg = np.mean(upper_triangle)

                    # 分箱后的值
                    binned_values = bin_similarity_values(upper_triangle)
                    binned_avg = np.mean(binned_values)

                    print(f"  {stage}: 原始平均值 {original_avg:.4f}, 分箱后平均值 {binned_avg:.4f}")

        # # 计算不同阶段间基因的平均相似度
        # print(f"\n不同阶段间基因的平均相似度:")
        # stage_list = list(stage_ranges.keys())
        #
        # for i, stage1 in enumerate(stage_list):
        #     for j, stage2 in enumerate(stage_list):
        #         if i < j:  # 避免重复计算
        #             start1, end1 = stage_ranges[stage1]
        #             start2, end2 = stage_ranges[stage2]
        #
        #             # 提取两个阶段间的相似度矩阵
        #             cross_matrix = similarity_matrix.iloc[start1:end1, start2:end2]
        #             avg_similarity = cross_matrix.values.mean()
        #             print(f"  {stage1} vs {stage2}: {avg_similarity:.4f}")

        # 计算整体平均相似度（排除对角线）
        n = similarity_matrix.shape[0]
        if n > 1:
            upper_triangle_all = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_triangle_all.append(similarity_matrix.iloc[i, j])
            overall_avg = np.mean(upper_triangle_all)
            binned_values_all = bin_similarity_values(upper_triangle_all)
            binned_avg_all = np.mean(binned_values_all)

            print(f"整体原始平均值 {overall_avg:.4f}, 分箱后平均值 {binned_avg_all:.4f}")




def main():

    folder_path = 'C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data'

    # 处理三种年龄状态
    age_states = ['young', 'early age', 'late age']

    # # 存储每个状态的平均矩阵
    # average_matrices = {}
    #
    # for state in age_states:
    #     average_matrices[state] = average_similarity_matrices(folder_path, state)
    #
    #     # 打印每个状态的平均矩阵信息
    #     print(f"{state} 状态平均矩阵形状:", average_matrices[state].shape)
    #     print(f"{state} 状态基因名称示例:", average_matrices[state].index[:5].tolist())



    # # 存储每个状态的唯一基因
    # unique_genes = {}
    #
    # # 提取并保存每个状态的唯一基因
    # for state in age_states:
    #     input_file = os.path.join(folder_path, f'{state}_high_similarity_gene_pairs.csv')
    #     output_file = os.path.join(folder_path, f'unique_genes_{state}.csv')
    #
    #     unique_genes[state] = set(extract_unique_genes(input_file, output_file))
    #
    #     # 找出每个状态特有的基因
    # state_specific_genes = find_state_specific_genes(unique_genes)
    #
    # # 保存每个状态特有的基因
    # for state, genes in state_specific_genes.items():
    #     specific_genes_file = os.path.join(folder_path, f'state_specific_genes_{state}.csv')
    #     specific_genes_df = pd.DataFrame(list(genes), columns=['Gene'])
    #     specific_genes_df.to_csv(specific_genes_file, index=False)
    #
    #     print(f"{state} 状态特有的基因数量: {len(genes)}")
    #     print(f"已保存 {state} 状态特有的基因到 {specific_genes_file}")
    #
    #     # 打印每个状态的基因数量
    # for state, genes in unique_genes.items():
    #     print(f"{state} 状态基因总数: {len(genes)}")

    # 加载每个状态特有的基因
    state_specific_genes, all_specific_genes = load_state_specific_genes(folder_path, age_states)

    # 打印每个状态特有基因的数量和总数
    for state, genes in state_specific_genes.items():
        print(f"{state} 状态特有基因数量: {len(genes)}")
    print(f"所有状态特有基因总数: {len(all_specific_genes)}")

    # 为所有特有基因创建热力图
    heatmap_data = create_combined_heatmaps(folder_path, age_states, all_specific_genes)

    # 创建并排热力图
    # create_side_by_side_heatmap(heatmap_data, folder_path)

    calculate_stage_similarity_averages(heatmap_data, state_specific_genes, all_specific_genes)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()









