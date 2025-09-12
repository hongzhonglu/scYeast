import numpy as np
from gene_name_alignment import get_name_from_scrna_database, get_gene2vec_input
import os
import time


def data_alignment():
    common_gene, gse125162_gene_name_list, gse242556_gene_name_list, _ = get_name_from_scrna_database()
    gene2vec_total_gene, modeling_total_gene = get_gene2vec_input(common_gene)
    with open('processed_data.txt', 'a') as file:
        for i in range(len(modeling_total_gene)):
            file.writelines(modeling_total_gene[i] + '\t')
        file.writelines('\n')
    with open('38225x6800_GSE125162.tsv', 'r') as f:
        gene_name_list = []
        for line in f:
            row = line.strip().split()
            if row[0] == '"YDL248W"':
                for j in range(len(row)):
                    if row[j] == '"Genotype"':
                        break
                    gene_name_list.append(row[j].replace('"', ''))
                break
    index = []
    for j in range(len(modeling_total_gene)):
        if modeling_total_gene[j] in gene_name_list:
            index.append(gene_name_list.index(modeling_total_gene[j])+1)
    with open('38225x6800_GSE125162.tsv', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == '"YDL248W"':
                continue
            for k in range(len(row)):
                if not row[k].isdigit():
                    row[k] = 0.0
                elif row[k].isdigit():
                    row[k] = float(row[k])
            non_zero = np.count_nonzero(row)
            if non_zero < 200:
                continue
            total = sum(row)
            factor = 10000/total
            with open('processed_data.txt', 'a') as file:
                for j in range(len(index)):
                    file.writelines(str(np.log1p(row[index[j]]*factor)) + '\t')
                file.writelines('\n')
    with open('173000x5843_GSE242556.tsv', 'r') as f:
        gene_name_list = []
        for line in f:
            row = line.strip().split()
            if row[0] == 'YAL068C':
                for j in range(len(row)):
                    if row[j] == 'Gene':
                        break
                    gene_name_list.append(row[j].replace('-', '.'))
                break
    index = []
    for j in range(len(modeling_total_gene)):
        if modeling_total_gene[j] in gene_name_list:
            index.append(gene_name_list.index(modeling_total_gene[j]))
    with open('173000x5843_GSE242556.tsv', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'YAL068C':
                continue
            for k in range(len(row)):
                if not row[k].isdigit():
                    row[k] = 0.0
                elif row[k].isdigit():
                    row[k] = float(row[k])
            non_zero = np.count_nonzero(row)
            if non_zero < 200:
                continue
            total = sum(row)
            factor = 10000 / total
            with open('processed_data.txt', 'a') as file:
                for j in range(len(index)):
                    file.writelines(str(np.log1p(row[index[j]]*factor)) + '\t')
                file.writelines('\n')
    return 0


def data_process():
    name_list = []
    each_gene_data = dict()
    with open('processed_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'Q0045':
                for j in range(len(row)):
                    name_list.append(row[j])
                    each_gene_data[row[j]] = []
            elif row[0] != 'Q0045':
                for k in range(len(row)):
                    each_gene_data[name_list[k]].append(float(row[k]))
    all_zero_gene = []
    max_expression = 0
    for i in range(len(name_list)):
        if any(num != 0 for num in each_gene_data[name_list[i]]):
            max_expression = max(max(each_gene_data[name_list[i]]), max_expression)
        else:
            all_zero_gene.append(name_list[i])
    return each_gene_data


if __name__ == '__main__':
    os.chdir('../data')
    # data_alignment()
    data_process()
