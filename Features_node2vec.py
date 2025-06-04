# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Features_node2vec.py
# Time       : 2024/7/2 11:29
# Author     : Yinyi Liu
# version    : python 3.12
# Description:
"""



import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import networkx as nx
from tqdm import tqdm
import pickle
import time
import torch
import argparse

parser = argparse.ArgumentParser(description='Features extraction')
parser.add_argument('--dataset', type=str, default='node_embeddings_node2vec.pkl') # 输入数据的路径
parser.add_argument('--output', type=str, default='forumlate_features_mean_node2vec.csv') # 输出的features文件名及存放路径。
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()



#加载embedding
embddding=pickle.load(open(args.dataset, 'rb'))
df_emb = pd.DataFrame.from_dict(embddding, orient='index')
print(df_emb.shape)






# 打印前几行内容以检查数据框结构
print(df_emb.head())


# 加载embedding和标签
data = pickle.load(open(args.dataset, 'rb'))
embedding = data['embeddings']
labels = data['labels']
df_emb = pd.DataFrame.from_dict(embedding, orient='index')
print(df_emb.shape)



df_list = []

edge_files = ['edge_PR_1', 'edge_PR_1Photo', 'edge_PR_21', 'edge_PR_21Photo', 'edge_YE_1S', 'edge_YE_1SPhoto', 'edge_YE_19S', 'edge_YE_19SPhoto']


for edge_file in edge_files:
    print(f'processing edge file: {edge_file}.csv')
    df_edge = pd.read_csv(f'{edge_file}.csv')
    print(df_edge.head())

    G_s = nx.from_pandas_edgelist(df_edge, "id_from", "id_to", create_using=nx.Graph())  # 读取每个样品中的分子网络

    rows = []
    nodes = []
    node_labels = []
    for node in tqdm(embedding.keys()):
        feas = embedding[node]
        if node not in G_s:
            continue  # 如果节点不在图中，跳过该节点
        nei_node = list(G_s.neighbors(node))
        for nnod in nei_node:
            if nnod not in embedding.keys():
                continue  # 如果邻居节点不在embedding中，跳过该邻居节点
            feas = np.vstack((feas, embedding[nnod]))

        ner_feas = np.mean(feas[1:, :], axis=0)
        result_vector = np.concatenate((feas[0, :], ner_feas))
        rows.append(result_vector)
        nodes.append(node)
        node_labels.append(labels[node])  # 添加标签

    df_tmp = pd.DataFrame(data=rows, index=nodes)
    df_tmp['label'] = node_labels  # 添加标签列
    df_list.append(df_tmp)

df_result = pd.concat(df_list, axis=0)
df_result = df_result.drop_duplicates()  # 去重操作
df_result.to_csv(args.output)
print(df_result.head())   # 打印前5个样本
print('shape of samples:', df_result.shape)



