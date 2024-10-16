import pandas as pd
import numpy as np

# 假设 CSV 文件名为 'data.csv'，其中包含一列表示序列数据的列名为 'sequence'

csv_file = 'data/train_last.csv'
data = pd.read_csv(csv_file)

# 提取序列数据并计算每个序列的长度
train_lengths = data['Sequence'].apply(len).tolist()

# 计算均值和标准差
mean_len = np.mean(train_lengths)
std_len = np.std(train_lengths)

# # 从高斯分布中采样一个长度
# seq_len = int(np.random.normal(loc=mean_len, scale=std_len))
#
# # 确保采样的长度在合理范围内
# seq_len = max(1, seq_len)  # 确保长度不小于1

# 输出均值和标准差
print(f"Mean length: {mean_len}")
print(f"Standard deviation of length: {std_len}")
# Mean length: 30.398994974874373
# Standard deviation of length: 22.709504772894665
