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

                
                tar = torch.zeros(1,1, model.d_model).to(device)
                

                generated_seq = model.generate(tar, z, properties)
                # print(generated_seq.shape)

            

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

sequences = generate_sequences(model, num_sequences)

for seq in sequences:
    #print(1)
    print(seq)
    





