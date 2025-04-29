import sys
import os
import pickle
from Bio import SeqIO
import pandas as pd
import numpy as np



def seq_to_one_hot(filename):
    """将序列转换为 one-hot 编码并保存为 NPZ 文件。"""
    seq_dict = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
                'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                'a': [1, 0, 0, 0], 'g': [0, 1, 0, 0],
                'c': [0, 0, 1, 0], 't': [0, 0, 0, 1]}  # 定义碱基的 one-hot 编码
    
    data = pd.read_csv(filename)  # 读取 CSV 文件
    label = np.array(data['label'])  # 提取标签
    n_sample = data.shape[0]  # 样本数量
    one_hot_list = []  # 存储 one-hot 编码
    
    # 遍历每个样本
    for i in range(0, n_sample):
        temp = []
        for c in data['resized_sequence'].iloc[i]:  # 遍历序列中的每个碱基
            temp.extend(seq_dict.get(c, [0, 0, 0, 0]))  # 转换为 one-hot 编码
        one_hot_list.append(temp)
    
    # 转换为 NumPy 数组
    sequence = np.array(one_hot_list, dtype='float32')
    
    # 保存为 NPZ 文件
    output_filename = filename.split('.')[0] + '.npz'
    np.savez(output_filename, label=label, sequence=sequence)
    print(f"One-hot encoded data saved to {output_filename}")

# 主程序
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python one_hot_encoding.py <input_file_A.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    seq_to_one_hot(input_file)  # 将序列转换为 one-hot 编码