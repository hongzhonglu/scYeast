import os
import pandas as pd

# 设置工作目录
os.chdir(r'C:\Users\Wenb1n\PycharmProjects\scYeast\DSGraph\data')

# 读取基因名称对照表
name_list = pd.read_table('expression_standard_and_systematic_name_list.txt')

# 读取要处理的基因列表文件
gene_list = pd.read_csv('high_similarity_gene_pairs/state_specific_genes_young.csv')

# 创建一个空列表来存储结果
result_data = []

# 遍历基因列表
for gene in gene_list['Gene']:
    # 检查是否为系统名
    if gene in name_list['systematic_name'].values:
        systematic_name = gene
        # 查找对应的标准名
        standard_name = name_list.loc[name_list['systematic_name'] == gene, 'standard_name'].values
        standard_name = standard_name[0] if len(standard_name) > 0 else ''

        # 检查是否为标准名
    elif gene in name_list['standard_name'].values:
        standard_name = gene
        # 查找对应的系统名
        systematic_name = name_list.loc[name_list['standard_name'] == gene, 'systematic_name'].values
        systematic_name = systematic_name[0] if len(systematic_name) > 0 else ''

        # 如果在对照表中没有找到
    else:
        systematic_name = gene
        standard_name = ''
        print(f"Warning: Gene {gene} not found in the name list")

        # 添加到结果列表
    result_data.append({
        'systematic_name': systematic_name,
        'standard_name': standard_name,
        'original_gene': gene
    })

# 使用列表创建DataFrame
result_df = pd.DataFrame(result_data)

# 保存结果
result_df.to_csv('high_similarity_gene_pairs/genename_conversion_young.csv', index=False)

print("Conversion complete. Results saved to gene_name_conversion.csv")
















