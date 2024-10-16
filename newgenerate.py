import torch

import torch.nn as nn
import numpy as np
from model import VAE  # 假设模型代码块保存在 model.py 文件中
from utils.predict1 import calculate_properties,normalize_matrix
import copy
import csv
# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE()
model.load_state_dict(torch.load('best_model1.pth'))
model.to(device)
model.eval()

# 氨基酸字母表
idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
properties = np.array([[0.19276148,0.19115417,0.57944557,0.05882353,0.1707089,0.03010101,
 0.19777205,0.41176471,0.14705882,0.35294118]], dtype=np.float32)

def generate_sequences(model, num_sequences, max_len=256):
    model.train()
    sequences = []

    # 生成随机特征属性，所有序列使用相同的属性
    # properties = np.random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                size=(1, 10)).astype(np.float32)
    # Nisin
    properties = np.array([[0.19276148, 0.19115417, 0.57944557, 0.05882353, 0.1707089, 0.03010101,
                            0.19777205, 0.41176471, 0.14705882, 0.35294118]], dtype=np.float32)

    print(properties)
    properties = torch.tensor(properties).to(device)

    for _ in range(num_sequences):
        # 随机序列长度
        #seq_len = np.random.randint(10, max_len + 1)
        #seq_len = 10
        mean_len = 30.398994974874373
        std_len =  22.709504772894665
        seq_len = int(np.random.normal(loc=mean_len, scale=std_len))
        if seq_len > 50:
            seq_len = 50
        if seq_len < 12:
            seq_len = 12
        # 生成潜在向量 z
        #std_len = 2
        #seq_len = 30
        with torch.no_grad():
            z = torch.randn(1, model.d_model).to(device)

            tar = torch.zeros(1, 256, dtype=torch.long).to(device)
            #print(std_len)
            mask = torch.zeros(256, dtype=torch.float32)
            #std_len = int(std_len)
            mask[:seq_len] = 1
            mask = mask.unsqueeze(0)
            mask.clone().detach().to(device)

            generated_seq = model.decode(tar,z, properties,mask).squeeze(0)

            

        # 只取前 seq_len 个氨基酸
        generated_seq = generated_seq[:seq_len]
        #print(generated_seq)
        # 将索引转换为氨基酸字符
        generated_seq = torch.argmax(generated_seq, dim=1).cpu().numpy()
        sequence = ''.join([idx_to_aa.get(idx, 'X') for idx in generated_seq])
        sequences.append(sequence)

    return sequences

# 生成 1000 个序列
num_sequences = 10000
num = 0
sequences = generate_sequences(model, num_sequences)
for seq in sequences:
    #print(seq)
    #print('标准：')
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
    # if np.all(last_nine < 0.15) and all(x > 0 for x in properties_array):
    if np.all(last_nine < 0.15) :
        num +=1
        print(seq)
        #print(normal)
        print(properties_array)
print(num)






