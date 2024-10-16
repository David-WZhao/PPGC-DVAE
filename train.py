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
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
class SequenceDataset(Dataset):
    def __init__(self, csv_file, max_len=256):
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
        self.aa_to_idx['X'] = 0  # 填充值0对应无效或未知氨基酸

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Sequence']
        sequence_idx = [self.aa_to_idx.get(aa, 0) for aa in sequence]  # 用0填充未知氨基酸
        if len(sequence_idx) > self.max_len:
            sequence_idx = sequence_idx[:self.max_len]  # 截断过长的序列
        else:
            sequence_idx = np.pad(sequence_idx, (0, self.max_len - len(sequence_idx)), 'constant', constant_values=(0, 0))
        sequence_idx = np.array(sequence_idx, dtype=np.int64)
        properties = self.data.iloc[idx][['Molecular Weight', 'Isoelectric Point (pI)', 'GRAVY', 'Aromaticity', 'Instability Index', 'Disulfide Bonds', 'Molecular Volume', 'Alpha Helix', 'Beta Sheet', 'Random Coil']].values
        properties = properties.astype(np.float32)

        #print(properties)
        #还需要一个target seq 需要构造 假使target一致
        tar_sequence_idx = sequence_idx
        # 随机遮盖sequence_idx中的某些位置
        # mask_prob = 0.2
        # for i in range(len(sequence_idx)):
        #     if random.random() < mask_prob:
        #         sequence_idx[i] = 0
        return torch.tensor(sequence_idx, dtype=torch.long), torch.tensor(properties, dtype=torch.float32),torch.tensor(tar_sequence_idx, dtype=torch.long)

def create_mask(x, pad_idx=0):
    return (x != pad_idx).float()

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, properties,tar) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
        mask = create_mask(data).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar,rnn_features = model(data, properties,tar,mask)
        loss = loss_function(recon_batch, data, mu, logvar, mask,properties,rnn_features)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def validate(model, val_loader):
    model.train()
    val_loss = 0
    with torch.no_grad():
        for data, properties,tar in tqdm(val_loader, desc="Validation"):
            data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
            mask = create_mask(data).to(device)
            recon_batch, mu, logvar,rnn_features = model(data, properties,tar,mask)
            loss = loss_function(recon_batch, data, mu, logvar, mask,properties,rnn_features)
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)
    print(f'====> Validation set loss: {val_loss:.4f}')
    return val_loss

def test(model, test_loader):
    model.train()
    test_loss = 0
    with torch.no_grad():
        for data, properties,tar in tqdm(test_loader, desc="Testing"):
            data, properties ,tar= data.to(device), properties.to(device),tar.to(device)
            mask = create_mask(data).to(device)
            recon_batch, mu, logvar ,rnn_features= model(data, properties,tar,mask)
            loss = loss_function(recon_batch, data, mu, logvar, mask,properties,rnn_features)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

if __name__ == '__main__':
    batch_size = 64

    epochs = 75
    lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #准备修改这里
    train_dataset = SequenceDataset('data/train_last.csv', max_len=256)
    val_dataset = SequenceDataset('data/valid_last.csv', max_len=256)
    test_dataset = SequenceDataset('data/test_last.csv', max_len=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        val_loss = validate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model1.pth')

    print("Testing the best model on the test set:")
    model.load_state_dict(torch.load('best_model1.pth'))
    test(model, test_loader)
