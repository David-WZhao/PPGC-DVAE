import pandas as pd
import numpy as np

def find_min_max(csv_file, columns):
    """
    从CSV文件中找出指定列的最大值和最小值。

    参数:
    - csv_file: CSV文件的路径
    - columns: 需要计算最大值和最小值的列名列表

    返回:
    - min_values: 每列的最小值，形状为 (len(columns),)
    - max_values: 每列的最大值，形状为 (len(columns),)
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 计算每列的最小值和最大值
    min_values = data[columns].min().values
    max_values = data[columns].max().values

    return min_values, max_values

# 示例使用
if __name__ == "__main__":
    # 假设你的CSV文件名为'data.csv'
    csv_file = 'data/train_last.csv'

    # 指定需要计算最大值和最小值的列名
    columns = ['Molecular Weight', 'Isoelectric Point (pI)', 'GRAVY', 'Aromaticity',
               'Instability Index', 'Disulfide Bonds', 'Molecular Volume',
               'Alpha Helix', 'Beta Sheet', 'Random Coil']

    # 调用函数找出每列的最大值和最小值
    min_values, max_values = find_min_max(csv_file, columns)

    # 打印结果
    print("Min values:", min_values)
    print("Max values:", max_values)
# Min values: [172.1818       4.05002842  -4.5          0.         -89.84
#    0.         172.8          0.           0.           0.        ]
# Max values: [1.98424843e+04 1.19999678e+01 4.20000000e+00 1.00000000e+00
#  5.41171429e+02 4.95000000e+04 2.45184000e+04 1.00000000e+00
#  1.00000000e+00 1.00000000e+00]
