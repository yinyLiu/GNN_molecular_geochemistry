# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : emb_GAT_CZ.py
# Time       : 2024/7/4 14:10
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import pandas as pd
import pickle
import time
import copy
import argparse

parser = argparse.ArgumentParser(description='pytorch version of GAT')
parser.add_argument('--dataset', type=str, default='combined_edges.csv')  # 输入边数据的路径
parser.add_argument('--nodes', type=str, default='combined_nodes.csv')  # 输入节点数据的路径
parser.add_argument('--output', type=str, default='combined_node_embeddings_gat.pkl')  # 输出的node embedding路径
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=64)  # 降低嵌入维度以减少显存占用
parser.add_argument('--num_heads', type=int, default=4)  # 减少注意力头的数量
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)  # 降低 epoch 数以加快训练
parser.add_argument('--batch_size', type=int, default=64)  # 降低 batch size 以减少显存占用

args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')

# Step 1: Load your graph using NetworkX
print('read network data.....')
df_edges = pd.read_csv(args.dataset)
df_nodes = pd.read_csv(args.nodes)
G = nx.from_pandas_edgelist(df_edges, "id_from", "id_to", create_using=nx.Graph())

# 节点重编码
original_nodes = list(G.nodes())
node_size = len(original_nodes)
new_nodes = list(range(len(original_nodes)))  # 对原始节点重新编号，从0开始
node_mapping = dict(zip(original_nodes, new_nodes))  # 将新编号与原有节点名字对应，构成Key:forumla,value:id的字典
rev_node_mapping = {v: k for k, v in node_mapping.items()}

# 重新标记节点
G = nx.relabel_nodes(G, node_mapping)

# Step 2: Convert the networkx graph to PyG's Data object
print('convert to PYG Data')
edge_index = torch.tensor(list(G.edges())).t().contiguous()  # 取得图中节点边，2*n的列表，第一个列表表示头节点，第二个列表表示尾节点
data = Data(edge_index=edge_index)

# 初始化节点特征为随机向量，可以根据需要更改
data.num_nodes = len(new_nodes)
data.x = torch.randn((data.num_nodes, args.embedding_dim)).to(device)
data = data.to(device)

# Step 3: Define and train GAT model on GPU
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

print('model defining......')
model = GAT(in_channels=args.embedding_dim, out_channels=args.embedding_dim, heads=args.num_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print('start training......')
model.train()
for epoch in range(args.epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)  # 这里假设一个简单的损失函数，可以根据任务调整
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}')

# 保存节点表征
model.eval()
with torch.no_grad():
    node_embeddings = model(data).cpu().numpy()

node_ids = list(G.nodes())
embedding_dict = {rev_node_mapping[i]: node_embeddings[i] for i in node_ids}

# 构建保存的数据字典
data_to_save = {
    'embeddings': embedding_dict
}

# 仅当存在标签列时才添加标签信息
if 'label' in df_nodes.columns:
    labels = {row['id2']: row['label'] for _, row in df_nodes.iterrows()}
    data_to_save['labels'] = labels
    
    # 检查标签分布
    label_counts = pd.Series(list(labels.values())).value_counts()
    print("Label distribution:")
    print(label_counts)
    
    # 创建标签到类别的映射
    label_to_class = {label: i for i, label in enumerate(label_counts.index)}
    class_to_label = {v: k for k, v in label_to_class.items()}
    
    print("\nClass to Label mapping:")
    for class_idx in sorted(class_to_label.keys()):
        print(f'class {class_idx}: {class_to_label[class_idx]}')
    
    data_to_save['class_to_label'] = class_to_label  # 添加标签映射

# 保存数据到文件
with open(args.output, 'wb') as f:
    pickle.dump(data_to_save, f)

print("Node embeddings saved successfully.")


# 加载新数据
df_new_edges = pd.read_csv("edge_external/new_edges_20240906.csv")
df_new_nodes = pd.read_csv("node_external/external_node.csv")

# 构建新图
G_new = nx.from_pandas_edgelist(df_new_edges, "id_from", "id_to", create_using=nx.Graph())
original_new_nodes = list(G_new.nodes())

# 为新节点分配新的唯一索引，同时保留原有节点的索引
new_node_mapping = {}
current_max_index = max(node_mapping.values())

for node in original_new_nodes:
    if node in node_mapping:
        # 如果节点已经在原图中，使用原图中的索引
        new_node_mapping[node] = node_mapping[node]
    else:
        # 为新节点分配新的索引，确保其从原图最高索引的下一个开始
        current_max_index += 1
        new_node_mapping[node] = current_max_index
        rev_node_mapping[current_max_index] = node  # 更新反向映射

# 更新 new_data.num_nodes，确保它大于或等于 new_node_mapping 中的最大索引值加 1
new_data_num_nodes = max(new_node_mapping.values()) + 1

# 重新标记新图的节点，使得它们对应到新的索引
G_new = nx.relabel_nodes(G_new, new_node_mapping)

# 转换为 PyG 数据
new_edge_index = torch.tensor(list(G_new.edges())).t().contiguous()
new_data = Data(edge_index=new_edge_index)
new_data.num_nodes = new_data_num_nodes

# 初始化新节点特征
new_data.x = torch.empty((new_data.num_nodes, args.embedding_dim)).to(device)
for node, idx in new_node_mapping.items():
    if node in node_mapping:  # 如果是共享节点
        original_node_idx = node_mapping[node]
        new_data.x[idx] = data.x[original_node_idx]
    else:  # 如果是新节点
        new_data.x[idx] = torch.randn(args.embedding_dim).to(device)

# 检查 edge_index 是否有效
print(f"Edge index max value: {new_data.edge_index.max().item()}")
print(f"Edge index min value: {new_data.edge_index.min().item()}")

if new_data.edge_index.max() >= new_data.num_nodes or new_data.edge_index.min() < 0:
    raise ValueError("Invalid edge_index detected: out of bounds")

# 将数据转移到 CUDA 设备上
new_data = new_data.to(device)

# 生成新的节点嵌入
model.eval()
with torch.no_grad():
    new_node_embeddings = model(new_data).cpu().numpy()

# 反向映射回原始节点ID，确保所有节点的键都是分子式或合理的标识符
new_embedding_dict = {rev_node_mapping[i]: new_node_embeddings[j] for j, i in enumerate(G_new.nodes())}

# 保存新的节点嵌入
with open('external_embeddings_gat.pkl', 'wb') as f:
    pickle.dump({'embeddings': new_embedding_dict}, f)

print("New node embeddings saved successfully.")