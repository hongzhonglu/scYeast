import os
import numpy as np


def get_name_from_scrna_database():
    gse125162_gene_name_list = []
    gse242556_gene_name_list = []
    expression_gene_name_list = []
    with open('38225x6800_GSE125162.tsv', 'r') as f:
        for line in f:
            row = line.strip().split()
            for j in range(len(row)):
                row[j] = row[j].replace('"', '')
                if row[j] == 'Genotype':
                    break
                if row[j][0] != 'Y' and row[j][0] != 'Q':
                    continue
                gse125162_gene_name_list.append(row[j])
            break
    gse125162_gene_name_list = list(np.unique(gse125162_gene_name_list))
    with open('173000x5843_GSE242556.tsv', 'r') as f:
        for line in f:
            row = line.strip().split()
            for j in range(len(row)):
                row[j] = row[j].replace('-', '.')
                if row[j] == 'Gene':
                    break
                if row[j][0] != 'Y' and row[j][0] != 'Q':
                    continue
                gse242556_gene_name_list.append(row[j])
            break
    gse242556_gene_name_list = list(np.unique(gse242556_gene_name_list))
    with open('expression_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'ORF':
                continue
            if row[0][0] == 'Y' or row[0][0] == 'Q':
                expression_gene_name_list.append(row[0])
    expression_gene_name_list = list(np.unique(expression_gene_name_list))
    common_gene = [gene for gene in gse125162_gene_name_list if gene in gse242556_gene_name_list]
    return common_gene, gse125162_gene_name_list, gse242556_gene_name_list, expression_gene_name_list


def get_gene2vec_input(common_gene):
    name_list = []
    with open('co_expression_from_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            name_list.append(row[0])
            name_list.append(row[1])
    name_list = list(np.unique(name_list))
    ex_gene = [gene for gene in common_gene if gene not in name_list]
    with open('gene2vec_input_co_expression.txt', 'a') as f:
        for i in range(len(ex_gene)):
            f.writelines(ex_gene[i] + '\n')
    gene2vec_total_gene = name_list + ex_gene
    modeling_total_gene = common_gene
    return gene2vec_total_gene, modeling_total_gene


if __name__ == '__main__':
    '''os.chdir('../data')
    common_gene, gse125162_gene_name_list, gse242556_gene_name_list, expression_gene_name_list = get_name_from_scrna_database()
    _, _ = get_gene2vec_input(common_gene)'''
