import torch

import torch.nn as nn
import numpy as np
from model import VAE  # 假设模型代码块保存在 model.py 文件中
from utils.predict1 import calculate_properties,normalize_matrix
import copy
import csv
from tqdm import tqdm
# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE()
# 5961版本是可以生成不同的序列
model.load_state_dict(torch.load('weight/model_5961_new_5.pth'))
model.to(device)
model.eval()

# 氨基酸字母表
idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y',21:'<eos>'}
properties = np.array([[0.19276148,0.19115417,0.57944557,0.05882353,0.1707089,0.03010101,
 0.19777205,0.41176471,0.14705882,0.35294118]], dtype=np.float32)
#Tachyplesin

# properties = np.array([[0.10658847,0.73974641,0.45774172,0.23529412,0.16337507,0.17636364,
#  0.10139576, 0.35294118 ,0.05882353 ,0.        ]], dtype=np.float32)
#002
# properties = np.array([[0.36279801, 0.15895403, 0.39552685, 0.14462616, 0.27889969,
#                         0.23950427, 0.35638272, 0.29547253, 0.20756271, 0.1304345]], dtype=np.float32)
#抗结核杆菌
# properties = np.array([[0.0589465,0.80649348,0.48754789,0.08333333,0.11726164,0.03010101,
#  0.06245174 ,0.41666667 ,0.25      , 0.25      ]], dtype=np.float32)
def generate_sequences(model, num_sequences, max_len=50):
    model.train()
    sequences = []

    # 生成随机特征属性，所有序列使用相同的属性
    # properties = np.random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                size=(1, 10)).astype(np.float32)
    # Nisin
    properties = np.array([[0.19276148, 0.19115417, 0.57944557, 0.05882353, 0.1707089, 0.03010101,
                            0.19777205, 0.41176471, 0.14705882, 0.35294118]], dtype=np.float32)
    # Tachyplesin KWCFRVCYRGICYRRCR
 #    properties = np.array([[0.10658847,0.73974641,0.45774172,0.23529412,0.16337507,0.17636364,
 # 0.10139576, 0.35294118 ,0.05882353 ,0.        ]], dtype=np.float32)
    # 002
    # properties = np.array([[0.36279801, 0.15895403, 0.39552685, 0.14462616, 0.27889969,
    #                     0.23950427, 0.35638272, 0.29547253, 0.20756271, 0.1304345]], dtype=np.float32)
    # properties = np.array([[0.0589465, 0.80649348, 0.48754789, 0.08333333, 0.11726164, 0.03010101,
    #                         0.06245174, 0.41666667, 0.25, 0.25]], dtype=np.float32)
    print(properties)
    properties = torch.tensor(properties).to(device)

    for _ in tqdm(range(num_sequences), desc="Generating sequences"):

        while True:
            with torch.no_grad():
                z = torch.randn(1, model.d_model).to(device)

                # tar_sequence_idx = []
                #
                # aa_to_idx = {aa: idx for idx, aa in idx_to_aa.items()}
                #
                #
                #
                # tar_sequence_idx = [aa_to_idx[aa] for aa in target]
                # tar_sequence_idx.append(21)
                # tar_sequence_idx = torch.tensor(tar_sequence_idx, dtype=torch.long)
                #
                # padding = torch.zeros(50-(len(target)+1), dtype=torch.long, device=device)
                #
                # tar_sequence_idx = torch.cat([tar_sequence_idx.to(device), padding])
                # tar_sequence_idx = tar_sequence_idx.unsqueeze(0)
                tar = torch.zeros(1,1, model.d_model).to(device)
                # tar_indices = torch.randint(1, 21, (1, 1)).to(device)  # 形状为 (1, 1)
                # tar = model.embedding1(tar_indices)
                # print(z.shape,tar.shape)
                # print(tar_sequence_idx)
                # print(std_len)

                generated_seq = model.generate(tar, z, properties)
                # print(generated_seq.shape)

            # 只取前 seq_len 个氨基酸
            # generated_seq = generated_seq[:seq_len]
            # print(generated_seq)
            # 将索引转换为氨基酸字符
            # print(generated_seq)

            # print(generated_seq.shape)
            sequence = ''.join([idx_to_aa.get(idx, 'X') for idx in generated_seq])
            # sequence = sequence[5:]
            # print(sequence)
            # 查找结束标记 <eos> 的索引并截断序列
            if '<eos>' in sequence:
                eos_position = sequence.index('<eos>')
                sequence = sequence[:eos_position]  # 截取 <eos> 之前的内容
            print(sequence)
            if sequence not in sequences:
                sequences.append(sequence)
                # print(sequence)
                break  # 跳出 while 循环，进入下一个生成

    return sequences


# 生成 1000 个序列
num_sequences = 100
num = 0
# target = 'ISILEKAILMLNPIMEKLFVTELVMKTEYSIHTN'
# target = 'KWCFRVCYRGICYRRCR'
sequences = generate_sequences(model, num_sequences)

for seq in sequences:
    #print(1)
    #print(seq)
    #print('标准：')
    if not seq:
        continue
    #print(seq)
    a = calculate_properties(seq)
    properties_array = []

    if a is None:
        print(f"Warning: properties_dict is None for sequence {seq}. Using default zero array.")
        properties_array = [0.0] * 10  # 使用包含10个零的数组
    else:
        for value in a.values():

            if isinstance(value, tuple):
                properties_array.extend(value)  # 如果是元组，展开并添加到列表中
            else:
                properties_array.append(value)  # 否则，直接添加到列表中
    normal = normalize_matrix(properties_array)
    #print(normal)
    differences = np.abs(properties - normal)
    # 选取后九位元素
    last_nine = differences[-10:]

    # 计算后九位元素的和
    total_sum = np.sum(last_nine)

    #print(normal)
    #print(properties_array)
    # if np.all(last_nine < 0.15) and all(x > 0 for x in properties_array):
    # if np.all(last_nine < 0.1) :
    if  np.all(last_nine < 0.2):
        num +=1
        print(seq)
        print(normal)
        #print(properties_array)
print(num)
# with open('generated_sequences_001.csv', 'w') as f:
#     writer = csv.writer(f)
#
#     # 写入CSV文件的表头，包含10个Normalized Properties列
#     writer.writerow(['Sequence ID', 'Sequence'] + [f'Normalized Property {i + 1}' for i in range(10)])
#     for i, seq in enumerate(sequences):
#         a = calculate_properties(seq)
#         properties_array = []
#         if a is None:
#             print(f"Warning: properties_dict is None for sequence {seq}. Using default zero array.")
#             properties_array = [0.0] * 10  # 使用包含10个零的数组
#         else:
#             for value in a.values():
#
#                 if isinstance(value, tuple):
#                     properties_array.extend(value)  # 如果是元组，展开并添加到列表中
#                 else:
#                     properties_array.append(value)  # 否则，直接添加到列表中
#         normal = normalize_matrix(properties_array)
#         #print(normal)
#         differences = np.abs(properties - normal)
#         # 选取后九位元素
#         last_nine = differences[-10:]
#
#         # 计算后九位元素的和
#         total_sum = np.sum(last_nine)
#         #if np.all(last_nine < 0.2):
#         writer.writerow([f"Sequence_{i + 1}", seq] + normal.tolist())





