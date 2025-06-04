# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : compute.py
# Time       : 2024/7/3 14:10
# Author     : Yinyi Liu
# version    : python 3.12
# Description: 
"""


import pandas as pd
import numpy as np
import re

# 解析分子式，返回一个字典，表示分子式中的元素和数量
def parse_formula(formula):
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    parsed = pattern.findall(formula)
    parsed_dict = {elem: int(count) if count else 1 for elem, count in parsed}
    return parsed_dict

# 将字典转换为可哈希的元组
def dict_to_tuple(d):
    return tuple(sorted(d.items()))

# 加载数据
df = pd.read_csv('combined_nodes.csv')

# 解析分子式并加入到DataFrame
df['parsed_formula'] = df['id2'].apply(parse_formula)
df['parsed_formula'] = df['parsed_formula'].apply(dict_to_tuple)

# 分组并检查标签是否一致
grouped = df.groupby('parsed_formula')['label'].nunique()

# 统计同分异构物标签不同的记录数
isomers_with_different_labels = grouped[grouped > 1].count()

print(f'分子数相同，同分异构物向量标签不同的记录数: {isomers_with_different_labels}')

