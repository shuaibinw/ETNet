import sys
import os
import pickle
from Bio import SeqIO
import pandas as pd
import numpy as np

# def generate_sequences():
#     """从输入的 BED 文件中生成调整长度后的序列，并保存到 CSV 文件中。"""
#     filepath = sys.argv[1]  # 获取输入文件路径
#     RESIZED_LEN = 2000  # 调整后的序列长度
#     fasta_sequence_dict = SeqIO.to_dict(SeqIO.parse(open('/public/home/shenyin_wsb_2606/EnContact-master/code/hg19.fa'), 'fasta'))  # 读取 hg19 参考基因组

#     # 读取输入的 BED 文件，并指定列名
#     locations = pd.read_csv(filepath, sep='\t', 
#                             names=['chr1', 'start1', 'stop1', 'name1', 
#                                     'chr2', 'start2', 'stop2', 'name2', 'label'])
    
#     # 创建 DataFrame 存储增强子信息
#     enhancer = pd.DataFrame(index=range(0, locations.shape[0] * 2),  # 每行包含两个增强子
#                             columns=['chr', 'original_start', 'original_end',
#                                       'resized_start', 'resized_end', 'resized_sequence',
#                                       'label', 'index'])
    
#     # 遍历每一行数据
#     for i in range(0, locations.shape[0]):
#         # 处理第一个增强子
#         chromosome = locations['chr1'][i]
#         original_location = (locations['start1'][i], locations['stop1'][i])
#         label = locations['label'][i]
        
#         # 计算调整后的起始和终止位置
#         original_len = original_location[1] - original_location[0]
#         len_difference = RESIZED_LEN - original_len
#         resized_start = original_location[0] - len_difference // 2  # 使用整数除法
#         resized_end = resized_start + RESIZED_LEN
        
#         # 存储第一个增强子的信息
#         enhancer.at[i * 2, 'chr'] = chromosome
#         enhancer.at[i * 2, 'original_start'] = original_location[0]
#         enhancer.at[i * 2, 'original_end'] = original_location[1]
#         enhancer.at[i * 2, 'resized_start'] = resized_start
#         enhancer.at[i * 2, 'resized_end'] = resized_end
#         enhancer.at[i * 2, 'resized_sequence'] = str(fasta_sequence_dict[chromosome].seq[resized_start:resized_end])
#         enhancer.at[i * 2, 'label'] = label
#         enhancer.at[i * 2, 'index'] = i * 2
        
#         # 处理第二个增强子
#         chromosome = locations['chr2'][i]
#         original_location = (locations['start2'][i], locations['stop2'][i])
        
#         # 计算调整后的起始和终止位置
#         original_len = original_location[1] - original_location[0]
#         len_difference = RESIZED_LEN - original_len
#         resized_start = original_location[0] - len_difference // 2  # 使用整数除法
#         resized_end = resized_start + RESIZED_LEN
        
#         # 存储第二个增强子的信息
#         enhancer.at[i * 2 + 1, 'chr'] = chromosome
#         enhancer.at[i * 2 + 1, 'original_start'] = original_location[0]
#         enhancer.at[i * 2 + 1, 'original_end'] = original_location[1]
#         enhancer.at[i * 2 + 1, 'resized_start'] = resized_start
#         enhancer.at[i * 2 + 1, 'resized_end'] = resized_end
#         enhancer.at[i * 2 + 1, 'resized_sequence'] = str(fasta_sequence_dict[chromosome].seq[resized_start:resized_end])
#         enhancer.at[i * 2 + 1, 'label'] = label
#         enhancer.at[i * 2 + 1, 'index'] = i * 2 + 1
    
#     # 保存结果到 CSV 文件
#     enhancer.to_csv(filepath[:-4] + '_A.csv', index=False)
#     return

def split_data():
    """将数据划分为训练集和测试集。"""
    FOLD = 10  # 划分比例
    # cell = sys.argv[1]  # 获取细胞类型参数
    enhancer1 = pd.read_csv('enhancer1_A.csv')
    enhancer2 = pd.read_csv('enhancer2_A.csv')
    
    n_sample = enhancer1.shape[0]  # 样本数量
    rand_index = list(range(0, n_sample))  # 生成索引列表
    np.random.seed(n_sample)  # 设置随机种子
    np.random.shuffle(rand_index)  # 打乱索引
    
    # 划分训练集和测试集
    n_sample_B = n_sample - n_sample // FOLD
    enhancer1_B = enhancer1.iloc[rand_index[:n_sample_B]]
    enhancer1_C = enhancer1.iloc[rand_index[n_sample_B:]]
    enhancer2_B = enhancer2.iloc[rand_index[:n_sample_B]]
    enhancer2_C = enhancer2.iloc[rand_index[n_sample_B:]]
    
    # 保存随机索引
    with open('rand_index.pkl', 'wb') as f:
        pickle.dump(rand_index, f)
    
    # 保存划分后的数据
    enhancer1_B.to_csv('enhancer1_B.csv', index=False)
    enhancer1_C.to_csv('enhancer1_C.csv', index=False)
    enhancer2_B.to_csv('enhancer2_B.csv', index=False)
    enhancer2_C.to_csv('enhancer2_C.csv', index=False)
    return

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
    filename = filename.split('.')[0] + '.npz'
    np.savez(filename, label=label, sequence=sequence)
    return

# 主程序
if __name__ == "__main__":
    # generate_sequences()  # 生成调整长度后的序列
    split_data()  # 划分数据集
    # seq_to_one_hot(sys.argv[1][:-4] + '_A.csv')  # 将序列转换为 one-hot 编码