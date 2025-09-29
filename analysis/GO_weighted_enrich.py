#%%
import pandas as pd
import numpy as np
import pickle
import math
from pathlib import Path
from collections import defaultdict
from goatools.base import get_godag

# ========== 工具函数 ==========
def load_gene2go(go_anno_file):
    gene2go = pd.read_csv(go_anno_file, index_col=0).T.to_dict('list')
    gene2go = {k: list(filter(None, v)) for k, v in gene2go.items()}
    gene2go = {k: set(v) for k, v in gene2go.items()} 
    # 过滤空值，转为集合
    gene2go_clean = {
        gene: {go_id for go_id in go_set if not (isinstance(go_id, float) and pd.isna(go_id))}
        for gene, go_set in gene2go.items()
    }
    return gene2go_clean

def weighted_go_enrichment(sig_df, gene2go_clean, go_obo, out_pkl, gene2entry=None, weight_col='importance', pval_cutoff=0.05, min_enrich=1, permutation_n=2000, random_seed=38):
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    # 显著基因及其权重
    if gene2entry is not None:
        # 需要蛋白名->基因名，同时处理重复蛋白
        sig_weights = {}
        for prot, imp in zip(sig_df['protein'], sig_df[weight_col]):
            if prot in gene2entry:
                gene = gene2entry[prot]
                if gene in sig_weights:
                    sig_weights[gene] += imp  # 相同基因的权重相加
                else:
                    sig_weights[gene] = imp
    else:
        # 直接处理蛋白名，同时处理重复蛋白
        sig_weights = {}
        for prot, imp in zip(sig_df['protein'], sig_df[weight_col]):
            if prot in sig_weights:
                sig_weights[prot] += imp  # 相同蛋白的权重相加
            else:
                sig_weights[prot] = imp
    # 背景基因
    background = set(gene2go_clean.keys())
    # 统计每个GO term在显著集和背景集的加权出现次数
    go2sig_weight = defaultdict(float)
    go2bg_count = defaultdict(int)
    for gene, go_terms in gene2go_clean.items():
        for go in go_terms:
            go2bg_count[go] += 1
            if gene in sig_weights:
                go2sig_weight[go] += sig_weights[gene]
    # 总权重和总背景数
    total_sig_weight = sum(sig_weights.values())
    total_bg = len(background)
    # 置换检验
    results = []
    sig_genes = list(sig_weights.keys())  # 显著基因的基因名列表
    sig_importances = np.array([sig_weights[g] for g in sig_genes])  # 显著基因的importance数组
    for go in go2bg_count:
        w_in_study = go2sig_weight[go]  # 该GO在显著集中的加权和
        n_in_study = go2bg_count[go]    # 该GO在背景集中的基因数
        # enrichment: 显著集加权比例/背景比例
        enrich = (w_in_study / total_sig_weight) / (n_in_study / total_bg) if n_in_study > 0 and total_sig_weight > 0 else 0

        # 1. 找到sig_genes中哪些基因属于该GO
        go_sig_mask = np.array([g in gene2go_clean and go in gene2go_clean[g] for g in sig_genes])
        # go_sig_mask是一个布尔数组，表示每个显著基因是否属于该GO
        n_sig_go = go_sig_mask.sum()  # 属于该GO的显著基因数
        if n_sig_go == 0:
            continue  # 如果没有显著基因属于该GO，跳过

        # 2. 置换importance
        perm_enrich = []  # 存储每次置换的enrichment
        for _ in range(permutation_n):
            permuted = np.random.permutation(sig_importances)  # 随机打乱importance
            w_perm = permuted[go_sig_mask].sum()  # 取属于该GO的打乱后importance之和
            # 计算置换后的enrichment
            enrich_perm = (w_perm / total_sig_weight) / (n_in_study / total_bg) if n_in_study > 0 and total_sig_weight > 0 else 0
            perm_enrich.append(enrich_perm)
        perm_enrich = np.array(perm_enrich)  # 转为numpy数组

        # 3. p-value = 置换分布中大于等于实际enrich的比例
        pval = (np.sum(perm_enrich >= enrich) + 1) / (permutation_n + 1)  # 加1防止为0
        # 统计置换分布中有多少次enrichment大于等于实际观测值，+1是常用的连续性校正

        if pval < pval_cutoff and enrich > min_enrich:
            go_name = go_obo[go].name if go in go_obo else ''
            results.append({
                'GO_ID': go,
                'Term': go_name,
                'P-value': pval,
                'Enrichment': enrich,
                'Genes': [g for g in sig_weights if g in gene2go_clean and go in gene2go_clean[g]],
                'Weighted_in_study': w_in_study,
                'Count_in_bg': n_in_study
            })
    # 保存pkl
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"保存: {out_pkl}, 显著GO条目数: {len(results)}")
#%%
# ========== growth ==========
print("==== GROWTH ====")
growth_df = pd.read_csv('growth_top_mportant_proteins2.csv')
gene2go_clean = load_gene2go('/home/lulab/scyeast/data/go_annotations.csv')
go_obo = get_godag('go-basic.obo')
weighted_go_enrichment(growth_df, gene2go_clean, go_obo, 'growth_go_enrichment_weighted2.pkl')
#%%
import pickle
import pandas as pd

# 读取growth_go_enrichment_weighted.pkl文件
with open('growth_go_enrichment_weighted2.pkl', 'rb') as f:
    growth_results = pickle.load(f)

# 转为DataFrame便于展示
growth_df = pd.DataFrame(growth_results)

# 按Enrichment降序展示前10行
growth_df_sorted = growth_df.sort_values(by='Enrichment', ascending=False)
print("growth_go_enrichment_weighted.pkl Enrichment降序前10条结果：")
print(growth_df_sorted.head(30))

#%%
# ========== turnover ==========
print("==== TURNOVER ====")
turnover_df = pd.read_csv('turnover_top_mportant_proteins2.csv')
gene2go_clean = load_gene2go('../turnover/all_go_annotations.csv')
go_obo = get_godag('go-basic.obo')
with open('../turnover/gene2entry_dict.pkl', 'rb') as f:
    gene2entry = pickle.load(f)
weighted_go_enrichment(turnover_df, gene2go_clean, go_obo, 'turnover_go_enrichment_weighted2.pkl', gene2entry=gene2entry)
#%%
import pickle
import pandas as pd

# 读取turnover_go_enrichment_weighted.pkl文件
with open('turnover_go_enrichment_weighted2.pkl', 'rb') as f:
    turnover_results = pickle.load(f)

# 转为DataFrame便于展示
turnover_df = pd.DataFrame(turnover_results)

# 展示前10行
print("turnover_go_enrichment_weighted.pkl 前10条结果：")
print(turnover_df.head(20))
#%%
# ========== rpf ==========
print("==== RPF ====")
rpf_df = pd.read_csv('rpf_top_mportant_proteins2.csv')
gene2go_clean = load_gene2go('../turnover/all_go_annotations.csv')
go_obo = get_godag('go-basic.obo')
with open('../turnover/gene2entry_dict.pkl', 'rb') as f:
    gene2entry = pickle.load(f)
weighted_go_enrichment(rpf_df, gene2go_clean, go_obo, 'rpf_go_enrichment_weighted2.pkl', gene2entry=gene2entry) 
# %%
import pickle
import pandas as pd

# 读取rpf_go_enrichment_weighted.pkl文件
with open('rpf_go_enrichment_weighted2.pkl', 'rb') as f:
    rpf_results = pickle.load(f)

# 转为DataFrame便于展示
rpf_df = pd.DataFrame(rpf_results)

# 展示前10行
print("rpf_go_enrichment_weighted.pkl 前10条结果：")
print(rpf_df.head(20))

# %%
