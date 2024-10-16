import torch

import torch.nn as nn
import numpy as np
from model import VAE  # 假设模型代码块保存在 model.py 文件中
from utils.predict1 import calculate_properties,normalize_matrix
import copy
# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE()
model.load_state_dict(torch.load('best_model1.pth'))
model.to(device)
model.eval()

# 氨基酸字母表
idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
#properties = np.array([[0.2, 1.0, 0.48, 0.14, 0.19, 0.03, 0.15, 0.40, 0.22, 0.07]], dtype=np.float32)
def generate_sequences(model, num_sequences, max_len=256):
    model.train()
    sequences = []

    

    for _ in range(num_sequences):
        # 随机序列长度
        # 生成随机特征属性，所有序列使用相同的属性
        #properties = np.array([[0.2, 1.0, 0.48, 0.14, 0.19, 0.03, 0.15, 0.40, 0.22, 0.07]], dtype=np.float32)
        # 生成随机特征属性，所有序列使用相同的属性
        properties = np.random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       size=(1, 10)).astype(np.float32)
        properties = torch.tensor(properties).to(device)

        mean_len = 30.398994974874373
        std_len =  22.709504772894665
        seq_len = int(np.random.normal(loc=mean_len, scale=std_len))
        if seq_len > 256:
            seq_len = 256
        if seq_len < 12:
            seq_len = 12
        # 生成潜在向量 z

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

sequences = generate_sequences(model, num_sequences)



# 将生成的序列保存到 txt 文件
# with open('generated_sequences3.txt', 'w') as f:
#     for i, seq in enumerate(sequences):
#         f.write(seq + '\n')
#
# print(f"{num_sequences} sequences have been generated and saved to 'generated_sequences3.txt'.")
# 将生成的序列保存为 FASTA 文件
with open('generated_sequences.fasta', 'w') as f:
    for i, seq in enumerate(sequences):
        f.write(f">Sequence_{i+1}\n")
        f.write(seq + '\n')

print(f"{num_sequences} sequences have been generated and saved to 'generated_sequences.fasta'.")

