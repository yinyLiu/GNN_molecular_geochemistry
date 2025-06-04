# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Classification_DecisionTree.py
# Time       : 2024/7/16 14:23
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import time
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Classification using Decision Tree')
parser.add_argument('--dataset', type=str, default='forumlate_features_mean_node2vec.csv')  # 输入数据的路径
parser.add_argument('--eval_output', type=str, default='res.txt')  # 输出的node embedding路径
parser.add_argument('--num_epochs', type=int, default=50)  # 添加num_epochs参数
parser.add_argument('--patience', type=int, default=10)  # 添加patience参数
args = parser.parse_args()

# 打印当前参数
print("Current parameters:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

def train(data):
    model = DecisionTreeClassifier()
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    model.fit(X_train, y_train)
    
    return model

def validate(model, data):
    X_val = data['X_val']
    y_val = data['y_val']

    predicted_val = model.predict(X_val)
    val_f1 = f1_score(y_val, predicted_val, average='macro')

    return val_f1

def test(data, model, fold):
    X_val = data['X_val']
    y_val = data['y_val']

    predicted_score = model.predict_proba(X_val)
    predicted_labels = model.predict(X_val)

    accuracy_ = accuracy_score(y_val, predicted_labels)
    precision = precision_score(y_val, predicted_labels, average='macro')
    recall = recall_score(y_val, predicted_labels, average='macro')
    f1 = f1_score(y_val, predicted_labels, average='macro')
    print(f'accuracy: {accuracy_:.4f}\t precision: {precision:.4f}\t recall: {recall:.4f}\t F1: {f1:.4f}')

    if len(predicted_score.shape) == 1:
        predicted_score = predicted_score.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = pd.get_dummies(y_val).values

    fpr, tpr, roc_auc = calculate_roc_curve(y_val, predicted_score)
    draw_roc_curve(fpr, tpr, roc_auc, fold)
    precision_c, recall_c, average_precision = draw_precision_recall(y_val, predicted_score)

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
    plt.savefig(f'Deci_PR_node2vec_{timestamp}.png')
    return precision, recall, average_precision

def draw_roc_curve(fpr, tpr, roc_auc, fold):
    plt.figure()
    lw = 2
    if isinstance(fpr, dict):
        # 多分类情况
        for i in fpr:
            plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
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
    plt.savefig(f'Deci_Roc_curve_node2vec_{fold}_{timestamp}.png')

def calculate_roc_curve(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

def k_fold_cross_validation(data, num_epochs, patience, k=5):
    kf = KFold(n_splits=k, shuffle=True)

    val_accuracy_, val_precision, val_recall, val_f1 = [], [], [], []
    precision_c_fold, recall_c_folds, average_precisions = [], [], []
    f1_scores_per_epoch = [[] for _ in range(num_epochs)]

    fold = 0  # 记录当前折数
    for train_index, val_index in kf.split(data['X']):
        X_train_fold, X_val_fold = data['X'][train_index], data['X'][val_index]
        y_train_fold, y_val_fold = data['y'][train_index], data['y'][val_index]

        fold_data = {'X_train': X_train_fold, 'X_val': X_val_fold, 'y_train': y_train_fold, 'y_val': y_val_fold}

        # 训练模型
        model = train(fold_data)

        for epoch in range(num_epochs):
            f1_score_epoch = validate(model, fold_data)
            f1_scores_per_epoch[epoch].append(f1_score_epoch)

        # 验证模型
        accuracy_, precision, recall, f1, fpr, tpr, roc_auc, precision_c, recall_c, average_precision = test(fold_data, model, fold)
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

    return avg_acc, avg_pre, avg_rec, avg_f1, f1_scores_per_epoch

def plot_f1_scores(f1_scores_per_epoch):
    avg_f1_scores = np.mean(f1_scores_per_epoch, axis=1)
    epochs = range(len(f1_scores_per_epoch))
    
    plt.figure()
    for i, f1_scores in enumerate(zip(*f1_scores_per_epoch)):
        plt.plot(epochs, f1_scores, label=f'Fold {i+1}')
    plt.plot(epochs, avg_f1_scores, label='Average', linewidth=2, color='black')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 Score')
    plt.title('Validation F1 Score over Epochs')
    plt.legend()
    plt.savefig('Deci_F1_Score_over_Epochs.png')

if __name__ == '__main__':
    # 准备数据
    df = pd.read_csv(args.dataset, index_col=0)

    # 检查标签分布
    print(df['label'].value_counts())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    all_data = {'X': X, 'y': y}

    num_epochs = args.num_epochs
    patience = args.patience

    avg_acc, avg_pre, avg_rec, avg_f1, f1_scores_per_epoch = k_fold_cross_validation(all_data, num_epochs, patience, k=5)
    plot_f1_scores(f1_scores_per_epoch)
    
    with open(args.eval_output, 'a') as f:
        f.write(f'{avg_acc:.4f}\t {avg_pre:.4f}\t {avg_rec:.4f}\t {avg_f1:.4f}\n')

    print(f'avg_accuracy: {avg_acc:.4f}\t avg_precision: {avg_pre:.4f}\t avg_recall: {avg_rec:.4f}\t avg_F1: {avg_f1:.4f}')
