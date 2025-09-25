import pandas as pd

# 读取文件
train_file = "train.bed"
test_file = "test.bed"

cols = [
    "chr1", "start1", "end1", "id1",
    "chr2", "start2", "end2", "id2",
    "label"
]
train_df = pd.read_csv(train_file, sep="\t", header=None, names=cols)
test_df = pd.read_csv(test_file, sep="\t", header=None, names=cols)

# 收集 train 所有 anchor
anchors_train_left = train_df[["chr1", "start1", "end1"]].drop_duplicates()
anchors_train_right = train_df[["chr2", "start2", "end2"]].drop_duplicates()
anchors_train = pd.concat([anchors_train_left, anchors_train_right], ignore_index=True)
anchors_train_set = set([tuple(x) for x in anchors_train.values])

# 初始化用于保存唯一的 test 样本
test_filtered_rows = []
anchors_test_seen = set()  # 保存 test 中已经出现的 anchor

for _, row in test_df.iterrows():
    left = (row["chr1"], row["start1"], row["end1"])
    right = (row["chr2"], row["start2"], row["end2"])
    
    # 条件1: 左右 anchor 都不在 train 中
    # 条件2: 左右 anchor 都不在 test 已经选中过的 anchor 中
    if (left not in anchors_train_set and right not in anchors_train_set and
        left not in anchors_test_seen and right not in anchors_test_seen):
        
        test_filtered_rows.append(row)
        anchors_test_seen.add(left)
        anchors_test_seen.add(right)

# 构建新的 DataFrame
test_filtered = pd.DataFrame(test_filtered_rows).reset_index(drop=True)

# 保存
test_filtered.to_csv("test_filtered_unique.bed", sep="\t", header=False, index=False)

print(f"原始 test 数量: {len(test_df)}")
print(f"过滤后 test 数量（train 与 test 内部 anchor 全部唯一）: {len(test_filtered)}")
