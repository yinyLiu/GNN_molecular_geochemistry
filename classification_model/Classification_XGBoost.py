# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Classification_XGBoost.py
# Time       : 2024/7/20 14:19
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import time
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Classification using XGBoost')
parser.add_argument('--dataset', type=str, default='forumlate_features_mean_node2vec.csv')  # 输入数据的路径
parser.add_argument('--eval_output', type=str, default='res.txt')  # 输出的node embedding路径
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=3)
parser.add_argument('--hidden_size', nargs='+', type=int, default=[512, 256, 128], help='一个由逗号分隔的数值列表')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

# 打印当前参数
print("Current parameters:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

def train(data, num_epochs=100, patience=50, learning_rate=0.1):
    model = xgb.XGBClassifier(n_estimators=num_epochs, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')
    
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
    plt.savefig(f'XGBoost_PR_node2vec_{timestamp}.png')
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
    plt.savefig(f'XGBoost_Roc_curve_node2vec_{fold}_{timestamp}.png')

def calculate_roc_curve(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

def k_fold_cross_validation(data, num_epochs, patience, learning_rate, k=5):
    kf = KFold(n_splits=k, shuffle=True)

    val_accuracy_, val_precision, val_recall, val_f1, f1_scores_per_epoch = [], [], [], [], []

    fold = 0  # 记录当前折数
    for train_index, val_index in kf.split(data['X']):
        X_train_fold, X_val_fold = data['X'][train_index], data['X'][val_index]
        y_train_fold, y_val_fold = data['y'][train_index], data['y'][val_index]

        fold_data = {'X_train': X_train_fold, 'X_val': X_val_fold, 'y_train': y_train_fold, 'y_val': y_val_fold}

        # 训练模型
        model = train(fold_data, num_epochs=num_epochs, patience=patience, learning_rate=learning_rate)

        # 验证模型
        accuracy_, precision, recall, f1, fpr, tpr, roc_auc, precision_c, recall_c, average_precision = test(fold_data, model, fold)
        val_accuracy_.append(accuracy_)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)

        f1_score_epoch = validate(model, fold_data)
        f1_scores_per_epoch.append(f1_score_epoch)

        fold += 1

    avg_acc = np.mean(val_accuracy_)
    avg_pre = np.mean(val_precision)
    avg_rec = np.mean(val_recall)
    avg_f1 = np.mean(val_f1)
    avg_f1_score_per_epoch = np.mean(f1_scores_per_epoch)

    return avg_acc, avg_pre, avg_rec, avg_f1, avg_f1_score_per_epoch, f1_scores_per_epoch

def plot_f1_scores(f1_scores_per_epoch):
    epochs = range(len(f1_scores_per_epoch[0]))
    for i, f1_scores in enumerate(f1_scores_per_epoch):
        plt.plot(epochs, f1_scores, label=f'Fold {i + 1}')
    plt.plot(epochs, np.mean(f1_scores_per_epoch, axis=0), label='Average', color='black', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 Score')
    plt.title('Validation F1 Score over Epochs')
    plt.legend()
    timestamp = int(time.time())
    plt.savefig(f'XGBoost_F1_Scores_{timestamp}.png')

if __name__ == '__main__':
    # 准备数据
    df = pd.read_csv(args.dataset, index_col=0)

    # 检查标签分布
    print(df['label'].value_counts())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 转换字符串标签为数值标签
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    # 打印标签编码的映射关系
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label encoding mapping:", label_mapping)

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    all_data = {'X': X, 'y': y}

    num_epochs = args.num_epochs
    patience = args.patience
    learning_rate = args.lr

    avg_acc, avg_pre, avg_rec, avg_f1, avg_f1_score_per_epoch, f1_scores_per_epoch = k_fold_cross_validation(all_data, num_epochs, patience, learning_rate, k=5)
    with open(args.eval_output, 'a') as f:
        f.write(f'{avg_acc:.4f}\t {avg_pre:.4f}\t {avg_rec:.4f}\t {avg_f1:.4f}\t {avg_f1_score_per_epoch:.4f}\n')

    print(f'avg_accuracy: {avg_acc:.4f}\t avg_precision: {avg_pre:.4f}\t avg_recall: {avg_rec:.4f}\t avg_F1: {avg_f1:.4f}\t avg_F1_score_per_epoch: {avg_f1_score_per_epoch:.4f}')

    plot_f1_scores(f1_scores_per_epoch)
