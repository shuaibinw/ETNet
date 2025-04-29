import sys
import pandas as pd
from Bio import SeqIO
import  os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_sequences():
    """从输入的 BED 文件中生成调整长度后的序列，并保存到两个 CSV 文件中。"""
    filepath = sys.argv[1]  # 获取输入文件路径
    RESIZED_LEN = 2000  # 调整后的序列长度
    fasta_sequence_dict = SeqIO.to_dict(SeqIO.parse(open('/public/home/shenyin_wsb_2606/EnContact-master/code/hg19.fa'), 'fasta'))  # 读取 hg19 参考基因组

    # 读取输入的 BED 文件，并指定列名
    locations = pd.read_csv(filepath, sep='\t', 
                            names=['chr1', 'start1', 'stop1', 'name1', 
                                   'chr2', 'start2', 'stop2', 'name2', 'label'])
    
    # 创建两个 DataFrame 分别存储第一个和第二个增强子
    enhancer1 = pd.DataFrame(columns=['chr', 'original_start', 'original_end',
                                      'resized_start', 'resized_end', 'resized_sequence',
                                      'label', 'index'])
    enhancer2 = pd.DataFrame(columns=['chr', 'original_start', 'original_end',
                                      'resized_start', 'resized_end', 'resized_sequence',
                                      'label', 'index'])
    
    # 遍历每一行数据
    for i in range(locations.shape[0]):
        # 处理第一个增强子
        chromosome = locations['chr1'][i]
        original_location = (locations['start1'][i], locations['stop1'][i])
        label = locations['label'][i]
        
        # 计算调整后的起始和终止位置
        original_len = original_location[1] - original_location[0]
        len_difference = RESIZED_LEN - original_len
        resized_start = original_location[0] - len_difference // 2  # 使用整数除法
        resized_end = resized_start + RESIZED_LEN
        
        # 提取调整后的序列
        resized_sequence = str(fasta_sequence_dict[chromosome].seq[resized_start:resized_end])
        
        # 存储第一个增强子的信息
        enhancer1.loc[i] = [
            chromosome,
            original_location[0],
            original_location[1],
            resized_start,
            resized_end,
            resized_sequence,
            label,
            i  # 使用行索引作为 index
        ]
        
        # 处理第二个增强子
        chromosome = locations['chr2'][i]
        original_location = (locations['start2'][i], locations['stop2'][i])
        
        # 计算调整后的起始和终止位置
        original_len = original_location[1] - original_location[0]
        len_difference = RESIZED_LEN - original_len
        resized_start = original_location[0] - len_difference // 2  # 使用整数除法
        resized_end = resized_start + RESIZED_LEN
        
        # 提取调整后的序列
        resized_sequence = str(fasta_sequence_dict[chromosome].seq[resized_start:resized_end])
        
        # 存储第二个增强子的信息
        enhancer2.loc[i] = [
            chromosome,
            original_location[0],
            original_location[1],
            resized_start,
            resized_end,
            resized_sequence,
            label,
            i  # 使用行索引作为 index
        ]
    
    # 保存结果到 CSV 文件
    enhancer1.to_csv('enhancer1_A.csv', index=False)  # 保存为 CSV 文件
    enhancer2.to_csv('enhancer2_A.csv', index=False)  # 保存为 CSV 文件
    return

# 主程序
if __name__ == "__main__":
    generate_sequences()  # 生成两个增强子文件