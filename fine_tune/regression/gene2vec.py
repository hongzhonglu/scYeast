import random
import os
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


def input_data_process():
    gene_pairs = []
    with open('gene2vec_input_co_expression.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            gene_pairs.append(row)
    random.shuffle(gene_pairs)
    return gene_pairs


def outputtxt(embeddings_file):
    model = KeyedVectors.load(embeddings_file)
    wordVector = model.wv
    vocabulary, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    wv = np.asarray(wv)
    index = 0
    matrix_txt_file = embeddings_file+".txt"  # gene2vec matrix txt file address
    with open(matrix_txt_file, 'w') as out:
        for i in wv[:]:
            out.write(str(vocabulary[index]) + "\t")
            index = index + 1
            for j in i:
                out.write(str(j) + " ")
            out.write("\n")
    out.close()


def gene2vec_training(gene_pairs):
    dimension = 50
    num_workers = 8
    sg = 1
    max_iter = 10
    window_size = 1
    txtoutput = True
    export_dir = '../embedding_result/'
    for current_iter in range(max_iter):
        if current_iter == 0:
            print('gene2vec dimension ' + str(dimension) + ' iteration ' + str(current_iter) + ' start')
            model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1,
                                           workers=num_workers, iter=1, sg=sg)
            model.save(export_dir + 'gene2vec_dim_' + str(dimension) + '_iter_' + str(current_iter))
            if txtoutput:
                outputtxt(export_dir + 'gene2vec_dim_' + str(dimension) + '_iter_' + str(current_iter))
            print('gene2vec dimension ' + str(dimension) + ' iteration' + str(current_iter) + ' done')
            del model
        else:
            random.shuffle(gene_pairs)
            print('gene2vec dimension ' + str(dimension) + ' iteration ' + str(current_iter) + ' start')
            model = gensim.models.Word2Vec.load(export_dir+"gene2vec_dim_"+str(dimension)+'_iter_'+str(current_iter-1))
            model.train(gene_pairs, total_examples=model.corpus_count, epochs=model.iter)
            model.save(export_dir+'gene2vec_dim_'+str(dimension)+'_iter_'+str(current_iter))
            if txtoutput:
                outputtxt(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter))
            print("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " done")
            del model


def read_gene2vec_result():
    gene2vec_result = {}
    with open('gene2vec_dim_200_iter_9spearman0.5.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            vec = []
            for j in range(1, len(row)):
                vec.append(float(row[j]))
            gene2vec_result[row[0]] = vec
    with open('alignment_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            name_list = []
            for j in range(len(row)):
                name_list.append(row[j])
            break
    gene2vec_weight = []
    for i in range(len(name_list)):
        gene2vec_weight.append(gene2vec_result[name_list[i]])
    gene2vec_weight = np.array(gene2vec_weight)
    return gene2vec_weight


if __name__ == '__main__':
    os.chdir('../data')

