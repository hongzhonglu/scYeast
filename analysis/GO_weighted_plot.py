#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置全局参数，参考go_paint.py
plt.rcParams.update({
    "font.family": "Arial", 
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.titlesize": 14,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.8,
    "grid.linestyle": '--',
    "legend.frameon": False
})

def split_term_label(term, max_words=6):
    """
    如果GO术语超过max_words个单词（以空格、-、,为分隔），则拆分成两行，分隔符保留
    """
    import re
    # 匹配单词和分隔符（空格、-、,）
    tokens = re.findall(r'\w+|[ ,\-]', term)
    word_count = 0
    split_index = None
    for i, token in enumerate(tokens):
        if re.match(r'\w+', token):
            word_count += 1
            if word_count == max_words:
                split_index = i + 1  # 包含当前单词
                break
    if split_index is not None and split_index < len(tokens):
        first_line = ''.join(tokens[:split_index])
        second_line = ''.join(tokens[split_index:])
        return first_line + '\n' + second_line
    else:
        return term

def plot_go_enrichment(pkl_file, title, output_file):
    """
    绘制GO富集图
    
    Parameters:
    pkl_file: pkl文件路径
    title: 图片标题
    output_file: 输出文件路径
    """
    # 读取pkl文件
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    df['Gene_Count'] = df['Genes'].apply(len)
    # 过滤条件：排除Count_in_bg小于等于2的GO条目
    df_filtered = df[df['Count_in_bg'] > 9].copy()
    df_filtered = df_filtered[df_filtered['Gene_Count'] > 5].copy()
    if len(df_filtered) == 0:
        print(f"警告：{pkl_file} 中没有满足条件的GO条目（Count_in_bg > 10）")
        return
    
    # 按照Enrichment从大到小排列，选取前10个
    df_sorted = df_filtered.sort_values('Enrichment', ascending=False).head(14)
    print(df_sorted)
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 创建颜色映射：蓝色（p值小，显著）到红色（p值大，不显著）
    colors = ['darkblue', 'blue', 'lightblue', 'lightcoral', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('pvalue_cmap', colors)
    
    # 归一化p值用于颜色映射
    p_values = df_sorted['P-value'].values
    norm = plt.Normalize(p_values.min(), p_values.max())
    
    # 绘制散点图
    scatter = ax.scatter(
        range(len(df_sorted)), 
        df_sorted['Enrichment'],
        s=df_sorted['Gene_Count'] * 20,  # 点的大小基于基因数量
        c=p_values,
        cmap=cmap,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 设置x轴标签（GO术语），如果术语大于4个单词则拆分成两行
    ax.set_xticks(range(len(df_sorted)))
    term_labels = [split_term_label(term, max_words=5) for term in df_sorted['Term']]
    ax.set_xticklabels(term_labels, rotation=45, ha='right', fontsize=20)
    
    # 设置y轴标签
    ax.set_ylabel('Enrichment ratio', fontsize=20)
    #ax.set_xlabel('GO Biological Process', fontsize=16)
    #ax.set_title(title, fontsize=18, pad=20)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('p-value', fontsize=20)
    
    # 创建图例显示点大小对应的基因数量，并增大各圆形之间的距离
    legend_elements = []
    gene_counts = [4, 8, 16, 32]  # 示例基因数量
    for count in gene_counts:
        legend_elements.append(plt.scatter([], [], s=count*20, c='gray', 
                                         alpha=0.5, edgecolors='black', 
                                         linewidth=0.5, label=f'{count}'))
    # 增大legend中各个圆形的距离
    ax.legend(handles=legend_elements, loc='upper right', title='Gene count', 
              title_fontsize=20, fontsize=20, borderpad=1, labelspacing=1.0, handletextpad=2.0)
    
    # # 添加图例
    # ax.legend(handles=legend_elements, loc='upper right', title='Gene Count', 
    #          title_fontsize=12, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"保存图片：{output_file}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print(f"\n{title} 统计信息：")
    print(f"总GO条目数：{len(results)}")
    print(f"满足条件的条目数（Count_in_bg > 2）：{len(df_filtered)}")
    print(f"展示的条目数：{len(df_sorted)}")
    print(f"Enrichment范围：{df_sorted['Enrichment'].min():.3f} - {df_sorted['Enrichment'].max():.3f}")
    print(f"P-value范围：{df_sorted['P-value'].min():.3e} - {df_sorted['P-value'].max():.3e}")

# 绘制三个任务的GO富集图
#%%
    # Growth任务
print("==== 绘制Growth任务GO富集图 ====")
plot_go_enrichment(
        'growth_go_enrichment_weighted2.pkl',
        'GO Enrichment Analysis (Growth)',
        'growth_go_enrichment_plot.svg'
    )
#%%
    # Turnover任务
print("\n==== 绘制Turnover任务GO富集图 ====")
plot_go_enrichment(
        'turnover_go_enrichment_weighted2.pkl',
        'GO Enrichment Analysis (Turnover)',
        'turnover_go_enrichment_plot.svg'
    )
#%%
    # RPF任务
print("\n==== 绘制RPF任务GO富集图 ====")
plot_go_enrichment(
        'rpf_go_enrichment_weighted2.pkl',
        'GO Enrichment Analysis (RPF)',
        'rpf_go_enrichment_plot.svg'
) 
# %%
