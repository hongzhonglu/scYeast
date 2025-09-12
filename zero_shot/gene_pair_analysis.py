import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import matplotlib as mpl

# 设置全局字体和图形参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# 设置工作目录
os.chdir(r'C:\Users\Wenb1n\PycharmProjects\scYeast\DSGraph\data\high_similarity_gene_pairs')

# 读取三个状态的高相似基因对文件
young_pairs = pd.read_csv('young_high_similarity_gene_pairs.csv')
early_age_pairs = pd.read_csv('early age_high_similarity_gene_pairs.csv')
late_age_pairs = pd.read_csv('late age_high_similarity_gene_pairs.csv')

# 定义标准化基因对的函数（确保基因对没有先后关系）
def standardize_gene_pair(row):
    return tuple(sorted([row['Gene1'], row['Gene2']]))

# 将基因对转换为标准化的元组集合
young_gene_pairs = set(young_pairs.apply(standardize_gene_pair, axis=1))
early_age_gene_pairs = set(early_age_pairs.apply(standardize_gene_pair, axis=1))
late_age_gene_pairs = set(late_age_pairs.apply(standardize_gene_pair, axis=1))

# 执行集合运算
only_young = young_gene_pairs - early_age_gene_pairs - late_age_gene_pairs
only_early_age = early_age_gene_pairs - young_gene_pairs - late_age_gene_pairs
only_late_age = late_age_gene_pairs - young_gene_pairs - early_age_gene_pairs

young_and_early_age = young_gene_pairs & early_age_gene_pairs - late_age_gene_pairs
young_and_late_age = young_gene_pairs & late_age_gene_pairs - early_age_gene_pairs
early_age_and_late_age = early_age_gene_pairs & late_age_gene_pairs - young_gene_pairs

all_three_states = young_gene_pairs & early_age_gene_pairs & late_age_gene_pairs

# 打印结果
print("仅存在于Young状态的基因对数量:", len(only_young))
print("仅存在于Early Age状态的基因对数量:", len(only_early_age))
print("仅存在于Late Age状态的基因对数量:", len(only_late_age))
print("同时存在于Young和Early Age状态的基因对数量:", len(young_and_early_age))
print("同时存在于Young和Late Age状态的基因对数量:", len(young_and_late_age))
print("同时存在于Early Age和Late Age状态的基因对数量:", len(early_age_and_late_age))
print("存在于所有三种状态的基因对数量:", len(all_three_states))

# 创建图形
plt.figure(figsize=(12, 8), dpi=600)

# 自定义颜色方案
colors = ['#9AFF9A', '#00F5FF', '#FFA54F']  # mediumseagreen, dodgerblue, tomato
outline_colors = ['#008B45', '#008B8B', '#8B6914']  # 对应的深色轮廓线
# 绘制韦恩图
v = venn3([early_age_gene_pairs,young_gene_pairs, late_age_gene_pairs],

          set_colors=colors)


# 去除坐标轴
plt.axis('off')
# 为每个圆添加对应颜色的轮廓线
for patch in v.patches:
    patch.set_alpha(0.6)    # 填充区域透明度
print(len(v.patches))
# 为圆形区域添加轮廓线
for i in range(7):
    v.patches[i].set_edgecolor('#696969')
    v.patches[i].set_linewidth(1.5)

# # 调整布局
# plt.tight_layout()

# 保存高质量图片
os.makedirs('gene_pair_analysis', exist_ok=True)
plt.savefig('gene_pair_analysis/gene_pairs_venn_diagram.png',
            dpi=600,
            bbox_inches='tight')
plt.close()

# # 创建结果DataFrame并保存
# def create_gene_pair_df(gene_pairs):
#     return pd.DataFrame(list(gene_pairs), columns=['Gene1', 'Gene2'])
#
# # 保存各类基因对
# only_young_df = create_gene_pair_df(only_young)
# only_early_age_df = create_gene_pair_df(only_early_age)
# only_late_age_df = create_gene_pair_df(only_late_age)
# young_and_early_age_df = create_gene_pair_df(young_and_early_age)
# young_and_late_age_df = create_gene_pair_df(young_and_late_age)
# early_age_and_late_age_df = create_gene_pair_df(early_age_and_late_age)
# all_three_states_df = create_gene_pair_df(all_three_states)
#
# # 创建输出文件夹
# os.makedirs('gene_pair_analysis', exist_ok=True)
#
# # 保存结果
# only_young_df.to_csv('gene_pair_analysis/only_young_gene_pairs.csv', index=False)
# only_early_age_df.to_csv('gene_pair_analysis/only_early_age_gene_pairs.csv', index=False)
# only_late_age_df.to_csv('gene_pair_analysis/only_late_age_gene_pairs.csv', index=False)
# young_and_early_age_df.to_csv('gene_pair_analysis/young_and_early_age_gene_pairs.csv', index=False)
# young_and_late_age_df.to_csv('gene_pair_analysis/young_and_late_age_gene_pairs.csv', index=False)
# early_age_and_late_age_df.to_csv('gene_pair_analysis/early_age_and_late_age_gene_pairs.csv', index=False)
# all_three_states_df.to_csv('gene_pair_analysis/all_three_states_gene_pairs.csv', index=False)

# 创建汇总结果
summary_data = {
    '分类': [
        '仅存在于Young状态的基因对',
        '仅存在于Early Age状态的基因对',
        '仅存在于Late Age状态的基因对',
        '同时存在于Young和Early Age状态的基因对',
        '同时存在于Young和Late Age状态的基因对',
        '同时存在于Early Age和Late Age状态的基因对',
        '存在于所有三种状态的基因对'
    ],
    '数量': [
        len(only_young),
        len(only_early_age),
        len(only_late_age),
        len(young_and_early_age),
        len(young_and_late_age),
        len(early_age_and_late_age),
        len(all_three_states)
    ]
}

# summary_df = pd.DataFrame(summary_data)
# summary_df.to_csv('gene_pair_analysis/gene_pair_summary.csv', index=False)

print("分析完成，结果已保存在 gene_pair_analysis 文件夹中")



