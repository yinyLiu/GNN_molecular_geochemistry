# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : addnetwork.py
# Time       : 2024/7/12 14:10
# Author     : Yinyi Liu
# version    : python 3.12
# Description:
"""


# 合并“大网络”和“由小网络合并成的网络”

import pandas as pd
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Merge embeddings')
parser.add_argument('--combined_dataset', type=str, default='combined_node_embeddings_gat.pkl', help='Embedding file for the combined network')  # 输入“小网络合并生成的大网络”嵌入文件的路径
parser.add_argument('--big_dataset', type=str, default='big_network_node_embeddings_gat.pkl', help='Embedding file for the big network')  # 输入“大网络”嵌入文件的路径
parser.add_argument('--output', type=str, default='merged_node_embeddings_gat.pkl', help='Output file for merged embeddings')  # 输出的合并后的嵌入文件路径
args = parser.parse_args()

# Load combined network embeddings
print('Loading combined network embeddings...')
with open(args.combined_dataset, 'rb') as f:
    combined_data = pickle.load(f)
combined_embeddings = combined_data['embeddings']
combined_labels = combined_data.get('labels', {})  # 使用 get 方法来安全地获取标签

# Load big network embeddings
print('Loading big network embeddings...')
with open(args.big_dataset, 'rb') as f:
    big_data = pickle.load(f)
big_embeddings = big_data['embeddings']
big_labels = big_data.get('labels', {})  # 使用 get 方法来安全地获取标签

# Average embeddings for nodes with multiple representations
def average_embeddings(embeddings):
    averaged_embeddings = {}
    for key, value in embeddings.items():
        if key in averaged_embeddings:
            averaged_embeddings[key].append(value)
        else:
            averaged_embeddings[key] = [value]
    
    for key, value in averaged_embeddings.items():
        averaged_embeddings[key] = np.mean(value, axis=0)
    
    return averaged_embeddings

print('Averaging big network embeddings...')
big_embeddings = average_embeddings(big_embeddings)

# Merge embeddings based on the same identifier
print('Merging embeddings...')
merged_embeddings = {}
for node in combined_embeddings:
    if node in big_embeddings:
        merged_embeddings[node] = np.concatenate((combined_embeddings[node], big_embeddings[node]))
    else:
        print(f"Warning: Node {node} not found in big network embeddings. Using only combined network embedding.")
        merged_embeddings[node] = np.concatenate((combined_embeddings[node], np.zeros_like(combined_embeddings[node])))

# Ensure all nodes are considered
for node in big_embeddings:
    if node not in merged_embeddings:
        merged_embeddings[node] = np.concatenate((np.zeros_like(big_embeddings[node]), big_embeddings[node]))

# Merge labels (assuming labels are the same for both networks)
if combined_labels:
    merged_labels = combined_labels
else:
    merged_labels = {}

# Save merged embeddings
merged_data = {'embeddings': merged_embeddings, 'labels': merged_labels}
with open(args.output, 'wb') as f:
    pickle.dump(merged_data, f)

print("Merged embeddings saved successfully.")
