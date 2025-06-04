# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : emb_node2vec.py
# Time       : 2024/7/8 14:21
# Author     : Yinyi Liu
# version    : python 3.12
# Description:
"""

"""


import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
import pandas as pd
import pickle
import time
import copy
import argparse

parser = argparse.ArgumentParser(description='pytorch version of node2vec')
parser.add_argument('--dataset', type=str, default='big_network/edges.csv') #输入边数据的路径
parser.add_argument('--nodes', type=str, default='big_network/nodes.csv') # 输入节点数据的路径
parser.add_argument('--output', type=str, default='big_network_node_embeddings_node2vec.pkl') #输出的node embedding路径。
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=50)
parser.add_argument('--context_size', type=int, default=5)
parser.add_argument('--walks_per_node', type=int, default=1)
parser.add_argument('--num_negative_samples', type=int, default=5)
parser.add_argument('--p', type=float, default=1)
parser.add_argument('--q', type=float, default=1)
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

# Step 3: Define and train node2vec model on GPU
print('model defining......')
model = Node2Vec(data.edge_index, embedding_dim=args.embedding_dim, walk_length=args.walk_length, context_size=args.context_size,
                 walks_per_node=args.walks_per_node, num_negative_samples=args.num_negative_samples, p=args.p, q=args.q, sparse=True).to(device)
loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

print('start training......')
model.train()
min_loss = float('inf')  # 用于记录最小验证集损失
for epoch in range(args.epochs):
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if total_loss < min_loss:
        min_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    print('Epoch:', epoch, 'Loss:', total_loss)

    if wait >= 20:
        print(f'Early stopping triggered after {epoch + 1} epochs.')
        break

# 保存节点表征
model.load_state_dict(best_model_weights)
node_embeddings = model.embedding.weight.detach().cpu().numpy()

node_ids = list(G.nodes())
embedding_dict = {rev_node_mapping[i]: node_embeddings[i] for i in node_ids}





# 加载标签数据
labels = {row['id2']: row['label'] for _, row in df_nodes.iterrows()}


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


# 保存表征向量、标签和标签映射到本地文件
data_to_save = {
    'embeddings': embedding_dict,
    'labels': labels,
    'class_to_label': class_to_label  # 添加标签映射
}





with open(args.output, 'wb') as f:
    pickle.dump(data_to_save, f)

print("Node embeddings and labels saved successfully.")


"""


import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
import pandas as pd
import pickle
import copy
import argparse

parser = argparse.ArgumentParser(description='pytorch version of node2vec')
parser.add_argument('--dataset', type=str, default='combined_edges.csv')  # 输入边数据的路径
parser.add_argument('--nodes', type=str, default='combined_nodes.csv')  # 输入节点数据的路径
parser.add_argument('--output', type=str, default='combined_big_node_embeddings_node2vec.pkl')  # 输出的node embedding路径
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=50)
parser.add_argument('--context_size', type=int, default=5)
parser.add_argument('--walks_per_node', type=int, default=1)
parser.add_argument('--num_negative_samples', type=int, default=5)
parser.add_argument('--p', type=float, default=1)
parser.add_argument('--q', type=float, default=1)
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

# Step 3: Define and train node2vec model on GPU
print('model defining......')
model = Node2Vec(data.edge_index, embedding_dim=args.embedding_dim, walk_length=args.walk_length, context_size=args.context_size,
                 walks_per_node=args.walks_per_node, num_negative_samples=args.num_negative_samples, p=args.p, q=args.q, sparse=True).to(device)
loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

print('start training......')
model.train()
min_loss = float('inf')  # 用于记录最小验证集损失
for epoch in range(args.epochs):
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if total_loss < min_loss:
        min_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    print('Epoch:', epoch, 'Loss:', total_loss)

    if wait >= 20:
        print(f'Early stopping triggered after {epoch + 1} epochs.')
        break

# 保存节点表征
model.load_state_dict(best_model_weights)
node_embeddings = model.embedding.weight.detach().cpu().numpy()

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



