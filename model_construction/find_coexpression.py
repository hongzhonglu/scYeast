from scipy.stats import pearsonr
import numpy as np
import os


def get_strains_data():
    strains = []
    genes = []
    with open('expression_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'ORF':
                continue
            strains.append(row[1])
            if row[0][0] == 'Y':
                genes.append(row[0])
        strains = list(np.unique(strains))
        genes = list(np.unique(genes))
    return genes, strains


def get_the_matrix():
    genes, strains = get_strains_data()
    data_matrix = list(np.zeros((len(genes), len(strains))))
    with open('expression_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'ORF':
                continue
            if row[0][0] == 'Y':
                data_matrix[genes.index(row[0])][strains.index(row[1])] = float(row[2])
    return data_matrix


def calculate_pearsonr():
    genes, strains = get_strains_data()
    data_matrix = get_the_matrix()
    corr_matrix = list(np.zeros((len(genes), len(genes))))
    for i in range(len(genes)):
        for j in range(i, len(genes)):
            if genes[i] == genes[j]:
                continue
            data1 = list(data_matrix[i])
            data2 = list(data_matrix[j])
            for k in range(len(data1)-1, -1, -1):
                if data1[k] == 0 or data2[k] == 0:
                    data1.pop(k)
                    data2.pop(k)
            corr, _ = pearsonr(data1, data2)
            corr_matrix[i][j] = corr_matrix[j][i] = corr
            if abs(corr) >= 0.7:
                with open('co_expression_from_data.txt', 'a') as file:
                    file.writelines(genes[i] + '\t' + genes[j] + '\n')
    return corr_matrix


if __name__ == '__main__':
    os.chdir('../data')
    corr_matrix = calculate_pearsonr()
