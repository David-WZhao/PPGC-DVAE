import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from model import VAE, loss_function  # 假设模型代码块保存在 model.py 文件中
import random
#增加一个tokenizer
# from transformers import EsmTokenizer, EsmForSequenceClassification
# 设置随机种子
# seed  = random.randint(0, 10000)  # 生成 0 到 100 之间的随机整数
seed = 8252
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
class SequenceDataset(Dataset):
    def __init__(self, csv_file, max_len=51):
        self.data = pd.read_csv(csv_file)
        # 选择需要归一化的特征列
        features_columns = ['Molecular Weight', 'Isoelectric Point (pI)', 'GRAVY', 'Aromaticity',
                            'Instability Index', 'Disulfide Bonds', 'Molecular Volume',
                            'Alpha Helix', 'Beta Sheet', 'Random Coil']

        # 提取特征数据
        features_data = self.data[features_columns].values

        # 初始化MinMaxScaler
        scaler = MinMaxScaler()

        # 对特征数据进行归一化
        normalized_features = scaler.fit_transform(features_data)

        # 将归一化后的数据替换回原数据框
        self.data[features_columns] = normalized_features

        self.max_len = max_len
        self.aa_to_idx = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY', start=1)}  # 20种氨基酸从1开始编码
        
        self.aa_to_idx['<eos>'] = 21
        self.aa_to_idx['<sos>'] = 22

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Sequence']
        # 添加 <sos> 到输入序列的开头
        sequence_idx = [self.aa_to_idx['<sos>']] + [self.aa_to_idx.get(aa, 0) for aa in sequence.upper()]
        eos_token_index = self.aa_to_idx['<eos>']
        target_seq = sequence_idx[1:] + [self.aa_to_idx['<eos>']]

        # 截断或填充序列以满足 max_len 的长度要求

        if len(sequence_idx) > self.max_len:
            sequence_idx = sequence_idx[:self.max_len]
            target_seq = target_seq[:self.max_len]
        else:
            sequence_idx = np.pad(sequence_idx, (0, self.max_len - len(sequence_idx)), 'constant',
                                  constant_values=(0, 0))
            target_seq = np.pad(target_seq, (0, self.max_len - len(target_seq)), 'constant', constant_values=(0, 0))

        sequence_idx = np.array(sequence_idx, dtype=np.int64)
        target_seq = np.array(target_seq, dtype=np.int64)
        properties = self.data.iloc[idx][['Molecular Weight', 'Isoelectric Point (pI)', 'GRAVY', 'Aromaticity', 
                                          'Instability Index', 'Disulfide Bonds', 'Molecular Volume', 
                                          'Alpha Helix', 'Beta Sheet', 'Random Coil']].values
        properties = properties.astype(np.float32)

        

        return torch.tensor(sequence_idx, dtype=torch.long), torch.tensor(properties, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.long)

def create_mask(x, pad_idx=0):
    return (x == pad_idx)

def train(model, train_loader, optimizer, epoch , batch_size,n_batch):
    model.train()
    train_loss = 0
    for batch_idx, (data, properties,tar) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
        mask = create_mask(data).to(device)
        mask = mask.repeat(n_batch, 1)
        # 修改的地方
        data = data.repeat(n_batch, 1)  # 扩展序列数据
        tar = tar.repeat(n_batch, 1)  # 扩展目标数据
        properties = properties.repeat(n_batch, 1)  # 扩展属性数据
        optimizer.zero_grad()
        recon_batch, mu, logvar, pred_features = model(data, properties,tar,mask)
        loss = loss_function(recon_batch, tar, mu, logvar, mask,properties,batch_size, pred_features)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / (len(train_loader.dataset)):.4f}')

def validate(model, val_loader,batch_size):
    model.train()
    val_loss = 0
    with torch.no_grad():
        for data, properties,tar in tqdm(val_loader, desc="Validation"):
            data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
            mask = create_mask(data).to(device)
            recon_batch, mu, logvar, pred_features = model(data, properties,tar,mask)
            loss = loss_function(recon_batch, data, mu, logvar, mask,properties,batch_size, pred_features)
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)
    print(f'====> Validation set loss: {val_loss:.4f}')
    return val_loss

def test(model, test_loader,batch_size):
    model.eval()
    test_loss = 0   
    with torch.no_grad():
        for data, properties,tar in tqdm(test_loader, desc="Testing"):
            data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
            mask = create_mask(data).to(device)
            recon_batch, mu, logvar, pred_features = model(data, properties,tar,mask)
            loss = loss_function(recon_batch, data, mu, logvar, mask,properties,batch_size, pred_features)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

if __name__ == '__main__':
    batch_size = 10
    batch_size1 = 50
    noise_batch = 30
    epochs = 5

    lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #准备修改这里
    train_dataset = SequenceDataset('data/train_last_new.csv', max_len=51)
    val_dataset = SequenceDataset('data/valid_last_new.csv', max_len=51)
    test_dataset = SequenceDataset('data/test_last_new.csv', max_len=51)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size1, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch,batch_size,noise_batch)
        val_loss = validate(model, val_loader,batch_size)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        torch.save(model.state_dict(), f'weight/model_{seed}_gai.pth')

    # print("Testing the best model on the test set:")
    # model.load_state_dict(torch.load(f'weight/model_{seed}.pth'))
    test(model, test_loader, batch_size)
