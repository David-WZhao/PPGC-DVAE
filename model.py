import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
import numpy as np
from utils.predict1 import calculate_properties, normalize_matrix
import copy

# 设置随机种子
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoisyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, noise_std=1):
        super(NoisyEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.noise_std = noise_std  # 噪声标准差，控制噪声的强度

    def forward(self, input):
        # 获取embedding向量
        embedding_output = self.embedding(input)

        # 在训练时添加噪声，测试时不加噪声
        if self.training:
            noise = torch.randn_like(embedding_output) * self.noise_std
            embedding_output = embedding_output + noise

        return embedding_output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VAE(nn.Module):
    def __init__(self, seq_len=256, feature_dim=10, d_model=128, nhead=8, num_layers=12):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model

        #self.embedding = nn.Embedding(21, d_model)  # 21种氨基酸，包括填充值
        self.embedding = NoisyEmbedding(21, d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.feature_fc1 = nn.Linear(feature_dim, d_model)
        self.feature_fc2 = nn.Linear(d_model, d_model)  # 增加一个全连接层处理特征
        self.feature_fc3 = nn.Linear(d_model + feature_dim, d_model)  # 映射回去
        # 将线性变换层转化为带激活函数的
        # self.fc_mu = nn.Linear(d_model, d_model)
        # self.fc_logvar = nn.Linear(d_model, d_model)
        self.fc_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_decode = nn.Linear(d_model, d_model)
        # Decoder 解码器部分
        self.transformer_decoder1 = TransformerEncoder(encoder_layers, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_fc1 = nn.Linear(d_model, d_model)  # 增加一个全连接层解码
        self.decoder_fc2 = nn.Linear(d_model, 21)
        self.apply(self._init_weights)  # 调用初始化函数

        # 添加RNN层
        self.rnn = nn.LSTM(21, d_model, batch_first=True)
        self.rnn_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 10),
            nn.Sigmoid()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=1)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x, features):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        #print(x[0])
        # 64 256 128
        x = self.pos_encoder(x)

        # features = self.feature_fc1(features)
        # features = F.relu(self.feature_fc2(features)).unsqueeze(1).repeat(1, self.seq_len, 1)
        # 更改为不映射到高维
        features = features.unsqueeze(1).repeat(1, self.seq_len, 1)
        # x = x + features
        x = torch.cat((x, features), dim=-1)
        x = self.feature_fc3(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    # 重参数化：从标准正态分布中采样的噪声转换为符合给定均值和方差的分布的样本
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, tar, z, features,mask):
        # print(mask.shape)训练时mask 64 256
        tar = self.embedding(tar) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # -----------
        # 64 256 128
        tar = self.pos_encoder(tar)  # -----------------------
        # print('z shape')
        # print(z.shape)64 128

        z = self.fc_decode(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # --------使序列每个位置共享一个z   理论上可行的


        # print('生成时的mask')1 256
        #print(mask.shape)
        #new_mask = mask.sum(dim=1).to(device)
        #print(new_mask)
        #print(z.shape) #1/64 256 128
        z = self.pos_encoder(z)# --------------------原有的7.25




        # features = self.feature_fc1(features)
        # features = F.relu(self.feature_fc2(features)).unsqueeze(1).repeat(1, self.seq_len, 1)
        features = features.unsqueeze(1).repeat(1, self.seq_len, 1)

        if mask.shape[0] == 1:
            mask = mask.unsqueeze(2)  # 形状变为 [1, 256, 1]
            m3 =copy.deepcopy(mask)
            mm = copy.deepcopy(mask)
            mask = mask.repeat(z.shape[0], 1, z.shape[2]).to(device)  # 形状变为 [1, 256, 128]
            #print(mask)
            z = z * mask
            mask = mm.repeat(features.shape[0],1,features.shape[2]).to(device)
            features = features * mask
            x = torch.cat((z, features), dim=-1)
            # print('before')
            # print(x)
            x = self.feature_fc3(x)
            m3 = m3.repeat(x.shape[0],1,x.shape[2]).to(device)
            x = x*m3
            # print('after')
            # print(x)
        else:
            mask = mask.unsqueeze(2)  # 形状变为 [64, 256, 1]
            m3 = copy.deepcopy(mask)
            mm = copy.deepcopy(mask)
            mask = mask.repeat(1, 1, z.shape[2]).to(device)  # 形状变为 [64, 256, 128]
            #print(mask.shape)
            z = z * mask
            mask = mm.repeat(1, 1, features.shape[2]).to(device)
            features = features * mask
            x = torch.cat((z, features), dim=-1)
            # print('before')
            # print(x)
            x = self.feature_fc3(x)
            m3 = m3.repeat(1, 1, x.shape[2]).to(device)
            x = x * m3
            # print('after')
            # print(x)


        #print(features.shape) 1 256 10
        # x = z + features

        x1 = self.transformer_decoder1(x)
        # print('encoder')
        # print(x1)
        x = self.transformer_decoder2(tar, x1)
        # print('decoder')
        # print(x)
        x = F.relu(self.decoder_fc1(x))
        # print('after activate')
        # print(x)
        # print(x.shape)
        return self.decoder_fc2(x)

    # 补充长度
    def forward(self, x, features, tar,mask):
        mu, logvar = self.encode(x, features)
        z = self.reparameterize(mu, logvar)
        # print(z.shape) 64 128
        # print(z)
        # 测试部分---------------------------------------------------------------------
        # print(x.shape) 64 256
        # 将索引转换为氨基酸字符
        # idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
        #              12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
        #
        # properties = np.random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                                size=(1, 10)).astype(np.float32)
        # properties = torch.tensor(properties).to(device)
        # mean_len = 30.398994974874373
        # std_len = 22.709504772894665
        # seq_len = int(np.random.normal(loc=mean_len, scale=std_len))
        # if seq_len > 256:
        #     seq_len = 256
        # if seq_len < 10:
        #     seq_len = 10
        # with torch.no_grad():
        #     #z = torch.randn(1, model.d_model).to(device)
        #     zz = z[0].unsqueeze(0)
        #     tar1 = torch.zeros(1, 256, dtype=torch.long).to(device)
        #     #print(z)
        #     generated_seq = self.decode(tar1,zz, properties).squeeze(0)
        #
        # # 只取前 seq_len 个氨基酸
        # generated_seq = generated_seq[:seq_len]
        # #print(generated_seq)
        #
        #
        # generated_seq = torch.argmax(generated_seq, dim=1).cpu().numpy()
        # sequence = ''.join([idx_to_aa.get(idx, 'X') for idx in generated_seq])
        #
        # print(sequence)
        # ------------------------------------------------------------------------------
        recon_x = self.decode(tar, z, features,mask)
        
        batch_size = 64
        max_len = 256
        lengths = mask.sum(dim=1).to(device)  # 将 lengths 保持在 GPU 上
        # 通过RNN层处理64 x l x 21的矩阵
        packed_input = nn.utils.rnn.pack_padded_sequence(recon_x, mask.sum(dim=1).cpu(), batch_first=True,
                                                         enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)

        idx = (lengths - 1).view(-1, 1).expand(len(lengths), rnn_output.size(2)).unsqueeze(1).to(torch.int64)
        rnn_features = rnn_output.gather(1, idx).squeeze(1)

        # 通过全连接层将RNN输出转换为特征
        rnn_features = self.rnn_fc(rnn_features)
        return recon_x, mu, logvar,rnn_features


def loss_function(recon_x, x, mu, logvar, mask, properties,rnn_features):
    # print(recon_x.shape) 64 256 21
    # print(recon_x[0][0])
    # print(x[0])  x 是64 256 编码后的
    # torch.argmax(generated_seq, dim=1).cpu().numpy()
    recon_x_max_idx = torch.argmax(recon_x, dim=-1)  # 现在 recon_x_max_idx 的形状是 (64, 256)
    # print(recon_x_max_idx[0])
    # 在这儿将x 和 重构x 输出对比
    # x_idx = x[0]
    # recon_idx = recon_x_max_idx[0]
    # mm = mask[0]
    # print(x_idx)
    # print(recon_idx*mm)
    # print(mask.shape) 64 256
    idx_to_aa = {0: 'X', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
                 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
    # sequence_recon = ''.join([idx_to_aa.get(idx, 'X') for idx in recon_idx])
    # sequence_x = ''.join([idx_to_aa.get(idx, 'X') for idx in x_idx])
    # print(sequence_x,sequence_recon)
    # -------------------------------------------------
    # 增加一个性质间的mapping loss
    # print(properties.shape) # 64 10
    # 输入的性质矩阵和重构后的性质矩阵如下
    input_properties = properties  # 输入的性质矩阵
    # 现在的问题是怎么构造重构的性质矩阵
    reconstructed_properties = []  # 重构后的性质矩阵
    # 计算每行中1的个数
    count_ones = torch.sum(mask == 1, dim=1)

    #print(count_ones)

    sequences2 = []
    #print('----------------------------原始x----------------------------------')
    for i in range(x.size(0)):
        row = x[i, :count_ones[i]].cpu().numpy()  # 获取有效值部分并转为numpy数组
        sequence = ''.join([idx_to_aa[idx] for idx in row])  # 转换为字母序列
        sequences2.append(sequence)
    # for seq in sequences2:
    #     #print(seq)
    sequences1 = []
    #print('----------------------------重构x----------------------------------')
    for i in range(recon_x_max_idx.size(0)):
        row = recon_x_max_idx[i, :count_ones[i]].cpu().numpy()  # 获取有效值部分并转为numpy数组
        sequence = ''.join([idx_to_aa[idx] for idx in row])  # 转换为字母序列
        sequences1.append(sequence)
    
    #reconstructed_properties = rnn_features
    for seq in sequences1:
        #print(seq)
        properties_dict = calculate_properties(seq)
        properties_array = []
        if properties_dict is None:
            print(f"Warning: properties_dict is None for sequence {seq}. Using default zero array.")
            properties_array = [0.0] * 10  # 使用包含10个零的数组
        else:
            for value in properties_dict.values():

                if isinstance(value, tuple):
                    properties_array.extend(value)  # 如果是元组，展开并添加到列表中
                else:
                    properties_array.append(value)  # 否则，直接添加到列表中

        reconstructed_properties.append(properties_array)
    # 在这儿将其归一化一下

    reconstructed_properties = normalize_matrix(reconstructed_properties)


    # 将属性数组列表转换为一个64x10的矩阵
    reconstructed_properties = torch.tensor(reconstructed_properties, device=device)
    #reconstructed_properties.clone().detach().to(device)  预测器时会用到-----

    # 定义均方误差损失函数
    mse_loss = nn.MSELoss()
    # 计算损失
    mapping_loss = mse_loss(reconstructed_properties, input_properties)
    #print(reconstructed_properties[0], input_properties[0])
    #print(mapping_loss)
    # -------------------------------------------------
    recon_x = recon_x.view(-1, 21)

    x = x.view(-1)
    mask = mask.view(-1)
    # print(recon_x.shape,x.shape)
    BCE = F.cross_entropy(recon_x, x, reduction='none')
    BCE = (BCE * mask).sum()  # 仅计算非填充值的损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print(1000 * mapping_loss, BCE, 10 * KLD)
    #print(BCE)
    return 1000 * mapping_loss + BCE + 10 * KLD
