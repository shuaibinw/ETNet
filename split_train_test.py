import pandas as pd
from sklearn.model_selection import train_test_split

# 读取文件
bed_file = "GM.bed"
df = pd.read_csv(bed_file, sep="\t", header=None)

# 假设最后一列是label列，获取列数
label_col = df.shape[1] - 1

# 分离正负样本
positive_samples = df[df[label_col] == 1]
negative_samples = df[df[label_col] == 0]

# 随机划分正样本 (80% train, 20% test)
train_positive, test_positive = train_test_split(
    positive_samples, test_size=0.2, random_state=42, stratify=None
)

# 随机划分负样本 (80% train, 20% test)
train_negative, test_negative = train_test_split(
    negative_samples, test_size=0.2, random_state=42, stratify=None
)

# 合并正负样本
test_df = pd.concat([test_positive, test_negative], ignore_index=True)
train_df = pd.concat([train_positive, train_negative], ignore_index=True)

# 保存
test_df.to_csv("test.bed", sep="\t", header=False, index=False)
train_df.to_csv("train.bed", sep="\t", header=False, index=False)

print(f"随机划分完成：")
print(f"train.bed {train_df.shape[0]} 行 (正样本: {len(train_positive)}, 负样本: {len(train_negative)})")
print(f"test.bed {test_df.shape[0]} 行 (正样本: {len(test_positive)}, 负样本: {len(test_negative)})")