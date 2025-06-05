# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : emb_GAT.py
# Time       : 2024/7/6 14:21
# Author     : Yinyi Liu
# version    : python 3.12
# Description:
"""





'''

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
parser.add_argument('--dataset', type=str, default='big_network/edges.csv')  # 输入边数据的路径
parser.add_argument('--nodes', type=str, default='big_network/nodes.csv')  # 输入节点数据的路径
parser.add_argument('--output', type=str, default='big_network_node_embeddings_gat.pkl')  # 输出的node embedding路径
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)  # GAT 的注意力头数量
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)

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

# 初始化节点特征为全1，可以根据需要更改
data.x = torch.ones((data.num_nodes, 1)).to(device)
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
model = GAT(in_channels=1, out_channels=args.embedding_dim, heads=args.num_heads).to(device)
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

'''



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
parser.add_argument('--epochs', type=int, default=200)  # 降低 epoch 数以加快训练
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
