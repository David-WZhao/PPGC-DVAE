import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
import numpy as np
from utils.predict1 import calculate_properties, normalize_matrix
import copy

# 设置随机种子
# seed = random.randint(0, 10000)  # 生成 0 到 100 之间的随机整数
seed = 15
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_causal_mask(seq_len):
    """
    生成因果掩码，用于防止窥视未来标记
    seq_len: 序列长度
    返回: 上三角掩码张量 [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 上三角矩阵
    mask = mask.masked_fill(mask == 1, float('-inf'))  # 将上三角部分填充为 -inf
    return mask.to(device)


def generate_causal_mask1(seq_len, batch_size):
    """
    生成因果掩码，用于防止窥视未来标记
    seq_len: 序列长度
    batch_size: 批次大小
    返回: 上三角掩码张量 [batch_size, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 上三角矩阵
    mask = mask.masked_fill(mask == 1, float('-inf'))  # 将上三角部分填充为 -inf
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # 扩展为 [batch_size, seq_len, seq_len]
    return mask.to(device)


class NoisyEmbedding(nn.Module):
    def __init__(self, embedding_layer, noise_std=1):
        super(NoisyEmbedding, self).__init__()
        self.embedding = embedding_layer
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
    def __init__(self, d_model, max_len=50):
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
    def __init__(self, seq_len=50, feature_dim=10, d_model=128, nhead=6, num_layers=12):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        # VAE 中的原始 Embedding
        self.embedding1 = nn.Embedding(22, d_model)  # 这里是原始的 nn.Embedding
        self.embedding2 = nn.Linear(10, d_model)
        # 使用 NoisyEmbedding 并将 VAE 的 embedding 层传入
        self.embedding = NoisyEmbedding(embedding_layer=self.embedding1, noise_std=1)

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layers = TransformerEncoderLayer(d_model+10, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.feature_fc1 = nn.Linear(feature_dim, d_model)
        self.feature_fc2 = nn.Linear(d_model+10, d_model)  # 增加一个全连接层处理特征
        self.feature_fc3 = nn.Linear(d_model + feature_dim, d_model)  # 映射回去
        # 将线性变换层转化为带激活函数的
        # self.fc_mu = nn.Linear(d_model, d_model)
        # self.fc_logvar = nn.Linear(d_model, d_model)
        self.fc_mu = nn.Sequential(
            nn.Linear(d_model+10, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(d_model+10, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_decode = nn.Linear(d_model, d_model)
        # Decoder 解码器部分
        encoder_layers1 = TransformerEncoderLayer(d_model+10, nhead)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model+10, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_fc1 = nn.Linear(d_model, d_model)  # 增加一个全连接层解码
        # self.decoder_fc2 = nn.Linear(d_model+10, 22)
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(d_model+10, d_model+10),
            nn.PReLU(),
            nn.LayerNorm(d_model+10),
            nn.Linear(d_model+10, 22)
        )
        self.apply(self._init_weights)  # 调用初始化函数

        # 添加RNN层
        self.rnn = nn.LSTM(22, d_model, batch_first=True)
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

    # 编码器部分
    def encode(self, x, features, mask):
        # x: [batch_size, seq_len]
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=device))
        #print(1,x.shape)
        # x: [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2).contiguous()   # 转换为 [seq_len, batch_size, d_model]
        #print(x.shape)
        x = self.pos_encoder(x)  # 添加位置编码
        #print(x.shape)
        # features: [batch_size, feature_dim]
        features = features.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, feature_dim]
        x = torch.cat((x.permute(1, 0, 2), features), dim=-1)  # [batch_size, seq_len, d_model + feature_dim]
        #print(x.shape)
        #x = self.feature_fc3(x)  # [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2).contiguous()   # [seq_len, batch_size, d_model]
        #print(1,x.shape)
        #print(mask)
        # 通过 Transformer 编码器
        x = self.transformer_encoder(x,src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2).contiguous() # [batch_size, seq_len, d_model]
        #print(x.shape)
        x = torch.mean(x, dim=1)  # [batch_size, d_model]

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    # 重参数化：从标准正态分布中采样的噪声转换为符合给定均值和方差的分布的样本
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码器部分
    def decode(self, tar, z, features, mask):
        # z: [batch_size, d_model]
        z = self.fc_decode(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, d_model]
        z = z.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]
        z = self.pos_encoder(z)  # 添加位置编码

        # features: [batch_size, feature_dim]
        features = features.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, feature_dim]
        x = torch.cat((z.permute(1, 0, 2), features), dim=-1)  # [batch_size, seq_len, d_model + feature_dim]
        #x = self.feature_fc3(x)  # [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]

        # tar: [batch_size, seq_len]
        tar_embed = self.embedding1(tar) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=device))
        tar_embed = tar_embed.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]
        tar_embed = self.pos_encoder(tar_embed)  # 添加位置编码

        # 生成因果掩码
        causal_mask = generate_causal_mask(self.seq_len)  # [seq_len, seq_len]

        # 通过 Transformer 解码器
        mm = self.transformer_encoder1(x,src_key_padding_mask=mask)
        #print(tar_embed.shape,features.shape)
        #print(tar_embed.shape)
        tar_embed = torch.cat((tar_embed.permute(1, 0, 2),features),dim=-1)
        #print(tar_embed.shape)
        tar_embed = tar_embed.permute(1, 0, 2).contiguous()#[seq_len, batch_size, d_model]
        output = self.transformer_decoder(tar_embed, mm, tgt_mask=causal_mask)  # [seq_len, batch_size, d_model]
        output = output.permute(1, 0, 2).contiguous()  # [batch_size, seq_len, d_model]
        output = self.decoder_fc2(output)  # [batch_size, seq_len, 22]

        return output

    def generate(self, tar, z, features):
        # z: [batch_size, d_model]
        #下一步计划  打算修改为每次seqlen加一的版本-----
        # z = self.fc_decode(z).unsqueeze(1).repeat(1, self.seq_len, 1) # [batch_size, seq_len, d_model]
        # z = z.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        # z = self.pos_encoder(z)  # 添加位置编码
        #
        # # features: [batch_size, feature_dim]
        # features = features.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, feature_dim]
        # zz = torch.cat((z.permute(1, 0, 2), features), dim=-1)  # [batch_size, seq_len, d_model + feature_dim]
        # zzz = self.feature_fc3(zz)  # [batch_size, seq_len, d_model]
        # zzz = zzz.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        #causal_mask = generate_causal_mask(self.seq_len)  # [seq_len, seq_len]
        sequence = []
        one = []
        old_z = z.clone()
        old_z = old_z.unsqueeze(1)
        old_feature = features.clone
        old_feature = features.unsqueeze(1)
        z = self.fc_decode(z).unsqueeze(1)
        features = features.unsqueeze(1)
        #print(z.shape, features.shape)
        for t in range(self.seq_len):  # 最大生成长度

            #print(z.shape, features.shape)
            z = z.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]
            z = self.pos_encoder(z)  # 添加位置编码

            # features: [batch_size, feature_dim]


            zz = torch.cat((z.permute(1, 0, 2), features), dim=-1)  # [batch_size, seq_len, d_model + feature_dim]
            #zzz = self.feature_fc3(zz)  # [batch_size, seq_len, d_model]
            zzz = zz.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]
            mm = self.transformer_encoder1(zzz)
            # tar: [batch_size, seq_len]
            causal_mask = generate_causal_mask(t+1)
            tar_embed = tar.permute(1, 0, 2).contiguous()  # [seq_len, batch_size, d_model]
            tar_embed = self.pos_encoder(tar_embed)  # 添加位置编码
            tar_embed = tar_embed.permute(1, 0, 2).contiguous()
            # 通过 Transformer 解码器
            #print(tar_embed.shape)
            ff = old_feature.repeat(1,tar_embed.shape[1],1)
            #print(ff.shape,tar_embed.shape)
            tar_embed = torch.cat((tar_embed, ff), dim=-1)
            #print()
            tar_embed = tar_embed.permute(1, 0, 2).contiguous()
            #print(mm.shape,tar_embed.shape)
            output = self.transformer_decoder(mm, tar_embed, tgt_mask=causal_mask)
            output = output.permute(1, 0, 2).contiguous()  # [batch_size, seq_len, d_model]
            tar_embed = tar.permute(1, 0, 2).contiguous()
            logits = self.decoder_fc2(output)  # [batch_size, seq_len, vocab_size]
            # print(output.shape)
            # 只取最后一个时间步的 logits
            logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 使用 softmax 直接生成概率分布
            probs = torch.softmax(logits, dim=-1)

            # 贪心解码，选择概率最大的标记
            next_token = torch.argmax(probs, dim=-1)  # 按概率选择最大值
            if next_token == 21:
                break
            z = z.permute(1, 0, 2).contiguous() 
            #print(z.shape,old_z.shape)
            z = torch.cat((z,old_z),dim=1)
            features = torch.cat((features,old_feature),dim=1)
            sequence.append(next_token.item())
            #print(sequence)
            one.append(next_token.item())
            #print(one)
            # print(sequence)
            # 将生成的标记添加到目标序列中
            # tar = torch.cat((generated_tar, next_token.unsqueeze(1)), dim=1)
            # tar =
            #print(one)
            #two = np.pad(one, (0, self.seq_len - len(one)), 'constant', constant_values=(0, 0))
            # print(two)
            two_tensor = torch.tensor(one).to(device)
            tar = self.embedding1(two_tensor).unsqueeze(0)
            #print(tar)
            #print(tar.shape)
            # print(tar)
            # 如果生成了结束标记，可以提前停止生成


        return sequence  # 返回生成的目标序列

    # 补充长度
    def forward(self, x, features, tar, mask):
        mu, logvar = self.encode(x, features, mask)
        z = self.reparameterize(mu, logvar)

        recon_x = self.decode(tar, z, features, mask)

        # batch_size = 50
        # max_len = 55
        # lengths = mask.sum(dim=1).to(device)  # 将 lengths 保持在 GPU 上
        # # 通过RNN层处理64 x l x 21的矩阵
        # packed_input = nn.utils.rnn.pack_padded_sequence(recon_x, mask.sum(dim=1).cpu(), batch_first=True,
        #                                                  enforce_sorted=False)
        # packed_output, _ = self.rnn(packed_input)
        # rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        # 
        # idx = (lengths - 1).view(-1, 1).expand(len(lengths), rnn_output.size(2)).unsqueeze(1).to(torch.int64)
        # rnn_features = rnn_output.gather(1, idx).squeeze(1)
        # 
        # # 通过全连接层将RNN输出转换为特征
        # rnn_features = self.rnn_fc(rnn_features)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, mask, properties,n_batch):
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
                 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y', 21: '<eos>'}
    #print(recon_x.shape,recon_x_max_idx.shape)
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
    # count_ones = torch.sum(mask == 1, dim=1)
    count_ones = torch.full((n_batch,), 50, dtype=torch.int64)   #100是noise_batch

    # print(count_ones)

    sequences2 = []
    # print('----------------------------原始x----------------------------------')
    # for i in range(x.size(0)):
    #     row = x[i, :count_ones[i]].cpu().numpy()  # 获取有效值部分并转为numpy数组
    #     sequence = ''.join([idx_to_aa[idx] for idx in row])  # 转换为字母序列
    #     # if sequence[0] == '<sos>':
    #     # print(sequence)
    #
    #         # 查找结束标记 <eos> 的索引并截断序列
    #     if '<eos>' in sequence:
    #         eos_position = sequence.index('<eos>')
    #         sequence = sequence[:eos_position]  # 截取 <eos> 之前的内容
    #     sequences2.append(sequence)
    # for seq in sequences2:
    #     print(seq)
    sequences1 = []
    # print('----------------------------重构x----------------------------------')
    for i in range(recon_x_max_idx.size(0)):
        row = recon_x_max_idx[i, :count_ones[i]].cpu().numpy()  # 获取有效值部分并转为numpy数组
        sequence = ''.join([idx_to_aa[idx] for idx in row])  # 转换为字母序列
        # print(sequence)

        # print(sequence)
        # print(sequence)
        # 查找结束标记 <eos> 的索引并截断序列
        if '<eos>' in sequence:
            eos_position = sequence.index('<eos>')
            sequence = sequence[:eos_position]  # 截取 <eos> 之前的内容
        sequences1.append(sequence)
    # for seq in sequences1:
    #     print(seq)
    print(sequences1[0])
    # reconstructed_properties = rnn_features
    for seq in sequences1:
        # print(seq)
        if not seq:
            properties_dict = None
        else:
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
    # reconstructed_properties.clone().detach().to(device)  预测器时会用到-----

    # 定义均方误差损失函数
    mse_loss = nn.MSELoss()
    # 计算损失
    mapping_loss = mse_loss(reconstructed_properties, input_properties)
    # print(reconstructed_properties[0], input_properties[0])
    # print(mapping_loss)
    # -------------------------------------------------
    recon_x = recon_x.view(-1, 22)

    x = x.view(-1)
    mask = mask.view(-1)
    # 定义填充索引（根据你的数据集设定）
    padding_idx = 0

    # 使用 CrossEntropyLoss，并设置 ignore_index
    BCE = F.cross_entropy(recon_x, x, ignore_index=padding_idx, reduction='sum')  # 仅计算非填充位置的损失
    BCE = (BCE * mask).sum()  # 仅计算非填充值的损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(10000 * mapping_loss,  BCE, KLD)
    # print(BCE)
    return 10000 * mapping_loss +  BCE + KLD
