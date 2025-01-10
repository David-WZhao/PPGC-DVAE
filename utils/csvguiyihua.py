import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
df = pd.read_csv('generated_sequences3.csv')

# 选择需要归一化的列
columns_to_normalize = [
    'Molecular Weight',
    'Isoelectric Point (pI)',
    'GRAVY',
    'Aromaticity',
    'Instability Index',
    'Disulfide Bonds',
    'Molecular Volume',
    'Secondary Structure Fraction 1',
    'Secondary Structure Fraction 2',
    'Secondary Structure Fraction 3'
]

# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 对选定的列进行归一化
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# 将归一化后的数据保存到新的CSV文件
df.to_csv('generated_sequences3guiyi.csv', index=False)


