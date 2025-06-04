# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : edge_label.py
# Time       : 2024/7/4 14:10
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""


import pandas as pd

# 读取数据
edges_df = pd.read_csv('combined_edges.csv')
nodes_df = pd.read_csv('combined_nodes.csv')

# 将id_from和id_to放到同一列名叫edge_node的列
edge_nodes_from = edges_df[['id_from']].rename(columns={'id_from': 'edge_node'})
edge_nodes_to = edges_df[['id_to']].rename(columns={'id_to': 'edge_node'})
edges_df_combined = pd.concat([edge_nodes_from, edge_nodes_to])

# 去重
# edges_df_unique = edges_df_combined.drop_duplicates(subset=['edge_node'])

# 创建字典以便快速查找标签
labels_dict = nodes_df.set_index('id2')['label'].to_dict()

# 查找每个edge_node对应的label
edges_df_combined['label'] = edges_df_combined['edge_node'].apply(lambda x: labels_dict.get(x, 'unknown'))

# 查看标签分布
label_counts = edges_df_combined['label'].value_counts()
print("Label distribution:")
print(label_counts)

# 保存结果
edges_df_combined.to_csv('edges_with_labels.csv', index=False)
