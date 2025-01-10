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
# model.load_state_dict(torch.load('weight/pretrained_model_4414_7000000.pth'))
model.load_state_dict(torch.load('weight/model_8252.pth'))
model.to(device)


# 氨基酸字母表
idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y',21:'<eos>'}

# properties = np.array([[0.62383954,0.19115417,0.57944557,0.05882353,0.1707089,0.03010101,
#  0.65505701,0.41176471,0.14705882,0.35294118]], dtype=np.float32)
#Tachyplesin 

# properties = np.array([[0.34495535,0.73974641,0.45774172,0.23529412,0.16337507,0.17636364,
#  0.3358412, 0.35294118 ,0.05882353 ,0.        ]], dtype=np.float32)

#抗结核杆菌 GGLYRLKKVLGK 
properties = np.array([[0.19077027,0.80649348,0.48754789,0.08333333,0.11726164,0.03010101,
 0.20685151 ,0.41666667 ,0.25      , 0.25      ]], dtype=np.float32)
def generate_sequences(model, num_sequences, max_len=51):
    model.train()
    sequences = []
    #print(model.transformer_decoder)
    # 生成随机特征属性，所有序列使用相同的属性
    # properties = np.random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                size=(1, 10)).astype(np.float32)
    # Nisin 
    # properties = np.array([[0.62383954, 0.19115417, 0.57944557, 0.05882353, 0.1707089, 0.03010101,
    #                         0.65505701, 0.41176471, 0.14705882, 0.35294118]], dtype=np.float32)
    # Tachyplesin KWCFRVCYRGICYRRCR  
 #    properties = np.array([[0.34495535,0.73974641,0.45774172,0.23529412,0.16337507,0.17636364,
 # 0.3358412, 0.35294118 ,0.05882353 ,0.        ]], dtype=np.float32)
    
    # 100（）
    properties = np.array([[0.19077027, 0.80649348, 0.48754789, 0.08333333, 0.11726164, 0.03010101,
                            0.20685151, 0.41666667, 0.25, 0.25]], dtype=np.float32)
    print(properties)
    properties = torch.tensor(properties).to(device)

    for _ in tqdm(range(num_sequences), desc="Generating sequences"):

        while True:
            with torch.no_grad():
                z = torch.randn(1, model.d_model).to(device)


                # tar = torch.zeros(1,1, model.d_model).to(device)
                # 假设起始符的索引是 start_token_index
                start_token_index = 22  # 根据你的词汇表定义的索引

                # 创建一个张量表示起始符索引
                start_token_tensor = torch.tensor([[start_token_index]]).to(device)  # shape: (1, 1)

                # 通过嵌入层生成嵌入向量
                start_token_embedding = model.embedding1(start_token_tensor)  # shape: (1, 1, d_model)

                # 将嵌入向量赋值给 tar
                tar = start_token_embedding
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
            #print(sequence)
            sequence = sequence.replace('X', '')
            if sequence not in sequences and len(sequence) < 51 :
                sequences.append(sequence)
                # print(sequence)
                break  # 跳出 while 循环，进入下一个生成

    return sequences


# 生成 1000 个序列
num_sequences = 1000
num = 0

sequences = generate_sequences(model, num_sequences)

for seq in sequences:
    #print(1)
    print(seq)
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
    print(normal)
    
    mse = np.mean((properties - normal) ** 2)

    if mse < 0.0625 :
        num +=1
        # print(seq)
        # print(normal)
        #print(properties_array)
print(num)
with open('generated_sequences_jiehe.csv', 'w') as f:
    writer = csv.writer(f)

    # 写入CSV文件的表头，包含10个Normalized Properties列
    writer.writerow(['Sequence ID', 'Sequence'] + [f'Normalized Property {i + 1}' for i in range(10)])
    for i, seq in enumerate(sequences):
        if not seq:
            continue
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
        # print(normal)

        

        writer.writerow([f"Sequence_{i + 1}", seq] + normal.tolist())




