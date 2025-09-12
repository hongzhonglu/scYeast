import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_relationship_file():
    relationship = dict()
    with open('expression_relationships.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            relationship[row[0]] = row[1].split(',')
    return relationship


def get_relation_matrix(relationship):
    name_list = []
    with open('processed_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == '0.0':
                break
            for j in range(len(row)):
                name_list.append(row[j].replace('.', '-'))
    relation_matrix = np.zeros((len(name_list), len(name_list)))
    tfs = list(relationship.keys())
    for i in range(len(tfs)):
        tgs = relationship[tfs[i]]
        for j in range(len(tgs)):
            if tfs[i] in name_list and tgs[j] in name_list:
                relation_matrix[name_list.index(tfs[i])][name_list.index(tgs[j])] = 1.
                relation_matrix[name_list.index(tgs[j])][name_list.index(tfs[i])] = 1.
    data = pd.DataFrame(relation_matrix)
    plot = sns.heatmap(data)
    # plt.show()
    with open('relation_matrix.txt', 'w') as f:
        f.write('\t')
        for i in range(len(name_list)):
            f.write(name_list[i] + '\t')
        f.write('\n')
        for i in range(len(name_list)):
            f.write(name_list[i])
            f.write('\t')
            for j in range(len(name_list)):
                f.write(str(relation_matrix[i][j]))
                f.write('\t')
            f.write('\n')
    return relation_matrix


if __name__ == '__main__':
    os.chdir('../data')
    # relationship = read_relationship_file()
    # get_relation_matrix(relationship)
