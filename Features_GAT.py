# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Features_GAT.py
# Time       : 2024/7/20 14:21
# Author     : Yinyi Liu
# version    : python 3.12
# Description:
"""


import pandas as pd
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Features extraction')
parser.add_argument('--dataset', type=str, default='merged_node_embeddings_gat.pkl', help='Embedding file containing merged network embeddings')  # 输入数据的路径
parser.add_argument('--output', type=str, default='merged_combined_features_gat.csv', help='Output file for combined features')  # 输出的features文件名及存放路径
args = parser.parse_args()

# Load embeddings and labels
print('Loading embeddings and labels...')
data = pickle.load(open(args.dataset, 'rb'))
embeddings = data['embeddings']
labels = data.get('labels', {})

# Merge features
print('Merging features...')
nodes = list(embeddings.keys())
merged_features = []

for node in nodes:
    merged_features.append(embeddings[node])

merged_features_df = pd.DataFrame(merged_features, index=nodes)

# Check the dimension of the embedding vectors
print(f"Embedding dimension: {merged_features_df.shape[1]}")

# Add labels
if labels:
    merged_features_df['label'] = merged_features_df.index.map(labels)

# Save merged features
merged_features_df.to_csv(args.output)
print(f'Combined features saved to {args.output}')
