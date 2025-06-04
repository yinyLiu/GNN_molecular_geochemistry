# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Classification_DNN.py
# Time       : 2024/7/18 14:19
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""

'''
class 0：labile
class 1：product
class 2：resistant
'''



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import copy
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold
import pickle  # 新增

warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--dataset', type=str, default='forumlate_features_mean_node2vec.csv')  # 输入数据的路径
parser.add_argument('--eval_output', type=str, default='res.txt')  # 输出的node embedding路径
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=3)
parser.add_argument('--hidden_size', nargs='+', type=int, default=[512, 256, 128], help='一个由逗号分隔的数值列表')
parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--num_epochs', type=int, default=89)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

# 指定使用的GPU
device = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# 打印当前参数
print("Current parameters:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelDNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        layers.append(nn.PReLU())
        layers.append(nn.BatchNorm1d(hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.PReLU())
            layers.append(nn.BatchNorm1d(hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train(data, input_size, hidden_size, output_size, learning_rate, num_epochs=1000, batch_size=500, patience=50):
    model = MultiLabelDNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 构造多标签分类损失
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-6, momentum=0.9)

    min_val_loss = float('inf')  # 用于记录最小验证集损失
    best_model_weights = copy.deepcopy(model.state_dict())  # 用于保存最佳模型权重
    wait = 0  # 用于记录当前等待轮数
    val_f1_list = []
    train_losses = []
    val_losses = []

    X_train = data['X_train']
    y_train = data['y_train']

    for epoch in tqdm(range(num_epochs)):
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_train_loss = 0.0
        model.train()
        for i in range(0, len(X_train_shuffled), batch_size):
            inputs = X_train_shuffled[i:i + batch_size]
            targets = y_train_shuffled[i:i + batch_size]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(X_train_shuffled) // batch_size
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(data['X_val']), data['y_val'])
            val_loss = v_loss.item()
            val_losses.append(val_loss)
            predicted_val = torch.sigmoid(model(data['X_val']))
            predicted_val_labels = torch.zeros_like(predicted_val)
            predicted_val_labels[range(len(predicted_val)), predicted_val.argmax(dim=1)] = 1
            val_f1 = f1_score(data['y_val'].cpu(), predicted_val_labels.cpu(), average='macro')
            val_f1_list.append(val_f1)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            wait = 0  # 重置等待轮数
        else:
            wait += 1  # 增加等待轮数

        print(f'Epoch: {epoch:03d}, Training Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # 如果达到早停条件，退出循环
        if wait >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs。')
            break

    model.load_state_dict(best_model_weights)
    return model, train_losses, val_losses, val_f1_list

def validate(model, data):
    X_val = data['X_val']
    y_val = data['y_val']

    X_test = data['X_test']
    y_test = data['y_test']
    with torch.no_grad():
        predicted_val = torch.sigmoid(model(X_val))
        predicted_test = torch.sigmoid(model(X_test))

        # 保留每行最大值对应的列，其余列设为0
        predicted_val_labels = torch.zeros_like(predicted_val)
        predicted_test_labels = torch.zeros_like(predicted_test)
        predicted_val_labels[range(len(predicted_val)), predicted_val.argmax(dim=1)] = 1
        predicted_test_labels[range(len(predicted_test)), predicted_test.argmax(dim=1)] = 1

        val_f1 = f1_score(y_val.cpu(), predicted_val_labels.cpu(), average='macro')
        test_f1 = f1_score(y_test.cpu(), predicted_test_labels.cpu(), average='macro')

    return val_f1, test_f1

def test(data, model, fold, class_to_label):  # 修改
    # 利用最优模型在测试集上进行测试，并评估
    with torch.no_grad():
        predicted_score = torch.sigmoid(model(data['X_val']))
        y_val = data['y_val'].cpu()

        # 保留每行最大值对应的列，其余列设为0
        predicted_labels = torch.zeros_like(predicted_score)
        predicted_labels[range(len(predicted_score)), predicted_score.argmax(dim=1)] = 1

        accuracy_ = accuracy_score(y_val, predicted_labels.cpu())
        precision = precision_score(y_true=y_val, y_pred=predicted_labels.cpu(), average='macro')
        recall = recall_score(y_true=y_val, y_pred=predicted_labels.cpu(), average='macro')
        f1 = f1_score(y_true=y_val, y_pred=predicted_labels.cpu(), average='macro')
        print(f'accuracy: {accuracy_:.4f}\t precision: {precision:.4f}\t recall: {recall:.4f}\t F1: {f1:.4f}')

        fpr, tpr, roc_auc = calculate_roc_curve(y_val, predicted_score.cpu())
        draw_roc_curve(fpr, tpr, roc_auc, fold, class_to_label)  # 修改
        precision_c, recall_c, average_precision = draw_precision_recall(y_val, predicted_score.cpu())

    return accuracy_, precision, recall, f1, fpr, tpr, roc_auc, precision_c, recall_c, average_precision

def draw_precision_recall(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    average_precision = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
    timestamp = int(time.time())
    plt.savefig(f'DNN_PR_node2vec_{timestamp}.png')
    return precision, recall, average_precision

def draw_roc_curve(fpr, tpr, roc_auc, fold, class_to_label):  # 修改
    plt.figure()
    lw = 2
    if isinstance(fpr, dict):
        # 多分类情况
        for i in fpr:
            plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_to_label[i], roc_auc[i]))  # 修改
    else:
        # 二分类情况
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    timestamp = int(time.time())
    plt.savefig(f'DNN_Roc_curve_node2vec_{fold}_{timestamp}.png')

def calculate_roc_curve(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

def k_fold_cross_validation(data, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, patience, k=5):
    kf = KFold(n_splits=k, shuffle=True)

    val_accuracy_, val_precision, val_recall, val_f1 = [], [], [], []
    precision_c_fold, recall_c_folds, average_precisions = [], [], []
    val_f1_lists = []
    train_loss_lists = []
    val_loss_lists = []

    fold = 0  # 记录当前折数
    for train_index, val_index in kf.split(data['X']):
        X_train_fold, X_val_fold = data['X'][train_index], data['X'][val_index]
        y_train_fold, y_val_fold = data['y'][train_index], data['y'][val_index]

        X_train_fold = torch.tensor(X_train_fold, dtype=torch.float).to(device)
        X_val_fold = torch.tensor(X_val_fold, dtype=torch.float).to(device)
        y_train_fold = torch.tensor(y_train_fold, dtype=torch.float).to(device)
        y_val_fold = torch.tensor(y_val_fold, dtype=torch.float).to(device)

        # 将每个子集组成新的训练集和验证集
        fold_data = {'X_train': X_train_fold, 'X_val': X_val_fold, 'y_train': y_train_fold, 'y_val': y_val_fold}

        # 训练模型
        model, train_losses, val_losses, val_f1_list = train(fold_data, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, patience)
        val_f1_lists.append(val_f1_list)
        train_loss_lists.append(train_losses)
        val_loss_lists.append(val_losses)

        # 验证模型
        accuracy_, precision, recall, f1, fpr, tpr, roc_auc, precision_c, recall_c, average_precision = test(fold_data, model, fold, class_to_label)  # 修改
        val_accuracy_.append(accuracy_)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)

        precision_c_fold.append(precision_c)
        recall_c_folds.append(recall_c)
        average_precisions.append(average_precision)

        fold += 1

    avg_acc = np.mean(val_accuracy_)
    avg_pre = np.mean(val_precision)
    avg_rec = np.mean(val_recall)
    avg_f1 = np.mean(val_f1)

    # 绘制F1值随训练过程变化的曲线
    plt.figure()
    for i, val_f1_list in enumerate(val_f1_lists):
        plt.plot(val_f1_list, label=f'Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 Score')
    plt.title('Validation F1 Score over Epochs')
    plt.legend()
    timestamp = int(time.time())
    plt.savefig(f'DNN_F1_Score_over_Epochs_{timestamp}.png')

    # 绘制训练和验证loss随训练过程变化的曲线
    plt.figure()
    for i in range(k):
        plt.plot(train_loss_lists[i], label=f'Fold {i+1} Train Loss')
        plt.plot(val_loss_lists[i], label=f'Fold {i+1} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()

    # 找出最低的验证损失并标记在图中
    min_val_loss_epoch = np.argmin(np.mean(val_loss_lists, axis=0))
    plt.axvline(x=min_val_loss_epoch, color='r', linestyle='--', label=f'Best Epoch: {min_val_loss_epoch+1}')
    plt.legend()
    plt.savefig(f'DNN_Loss_over_Epochs_{timestamp}.png')

    return avg_acc, avg_pre, avg_rec, avg_f1

if __name__ == '__main__':
    # 准备数据并将其移动到GPU上
    df = pd.read_csv(args.dataset, index_col=0)

    # 检查标签分布
    print(df['label'].value_counts())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 将字符串标签转换为0/1向量
    y = pd.get_dummies(df['label']).values

    # 确保数据类型为数值类型
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    all_data = {'X': X, 'y': y}

    input_size = X.shape[1]  # 自动确定输入大小
    output_size = y.shape[1]  # 根据标签的独特值确定输出大小
    hidden_size = args.hidden_size
    learning_rate = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    patience = args.patience

    # 加载标签映射
    with open('node_embeddings_node2vec.pkl', 'rb') as f:
        data = pickle.load(f)
    class_to_label = data['class_to_label']

    avg_acc, avg_pre, avg_rec, avg_f1 = k_fold_cross_validation(all_data, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, patience, k=5)
    with open(args.eval_output, 'a') as f:
        f.write(f'{avg_acc:.4f}\t {avg_pre:.4f}\t {avg_rec:.4f}\t {avg_f1:.4f}\n')

    print(f'avg_accuracy: {avg_acc:.4f}\t avg_precision: {avg_pre:.4f}\t avg_recall: {avg_rec:.4f}\t avg_F1: {avg_f1:.4f}')
