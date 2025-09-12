import pandas as pd
import os


def get_next_time_data(row_index):
    strain, time = row_index.split('_')
    time = float(time)
    next_time_data = df_gene_data[(df_gene_data['strain'] == strain) & (df_gene_data['time'] == time + 5)]

    if len(next_time_data) > 0:
        return 2 ** next_time_data.iloc[0]['log2_shrunken_timecourses']
    else:
        return None


if __name__ == '__main__':
    # os.chdir('../data')
    os.chdir('C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data/pert_train_data')
    df_matrix = pd.read_csv('interpolation_data_matrix_with_pert_data.csv', index_col=0)
    os.chdir('pert_data')
    gene_list = ['YOL154W']
    for i in range(len(gene_list)):
        df_gene_data = pd.read_csv(gene_list[i] + '_data.tsv', delimiter='\t')
        strain_times = df_matrix.index
        df_matrix = df_matrix[~df_matrix.index.str.endswith('_45')]
        df_matrix[gene_list[i] + '_output'] = df_matrix.index.map(get_next_time_data)
    df_matrix = df_matrix.fillna(1).astype(float)
    os.chdir('..')
    df_matrix.to_csv('pert_training_data_YOL154W.csv')

