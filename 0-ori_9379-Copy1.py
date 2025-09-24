#!/usr/bin/env python
# coding: utf-8

# 基于 9319，参数改大点

# In[1]:

import sys
from kaitoupao_wsl import *

# In[2]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# In[3]:

from sklearn.model_selection import train_test_split

# In[4]:

device = torch.device("cuda")

# # 加载训练集和测试集，将全量字符列表给它弄出来

# In[5]:

data_train = pd.read_csv(create_originalData_path("train_set.csv"), sep="\t")#.sample(1000) , nrows=1000
data_test = pd.read_csv(create_originalData_path("test_a.csv"), sep="\t")#.sample(1000)

# In[6]:

data_train.shape, data_test.shape

# In[7]:
type_of_class = data_train.label.nunique()

# In[8]:
train_data, valid_data = train_test_split(data_train, test_size=0.3, random_state=42)

# In[9]:
train_labels = torch.tensor(train_data.label.to_list(), dtype=torch.long)
valid_labels = torch.tensor(valid_data.label.to_list(), dtype=torch.long)
test_labels = torch.tensor([-1 for x in range(data_test.shape[0])], dtype=torch.long) ## test_labels的label是假的。

# In[ ]:



# ## 使用TF-IDF提取特征

# In[10]:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_train_features = tfidf_vectorizer.fit_transform(train_data['text'])
tfidf_valid_features = tfidf_vectorizer.transform(valid_data['text'])
tfidf_test_features = tfidf_vectorizer.transform(data_test['text'])

# In[11]:
train_features = torch.tensor(tfidf_train_features.toarray(), dtype=torch.float32)
valid_features = torch.tensor(tfidf_valid_features.toarray(), dtype=torch.float32)
test_features = torch.tensor(tfidf_test_features.toarray(), dtype=torch.float32)

# In[12]:
sc_input_dim = train_features.shape[1]
sc_input_dim

# ## 创建适合于语言序列的数据

# In[13]:
# 下载数据并进行预处理
vocab_size = 8000  # 只考虑前 20k 词汇
maxlen = 800  # 只考虑每条评论的前 200 个词

# In[14]:
def preprocess_seq_str_2_int(seq, len_lim = maxlen):
    rst = [int(wd) for idx, wd in enumerate(seq.strip().split()) if idx < len_lim]
    return rst

# In[15]:
x_train = [torch.tensor(preprocess_seq_str_2_int(seq), dtype=torch.long) for seq in train_data.text]
x_valid = [torch.tensor(preprocess_seq_str_2_int(seq), dtype=torch.long) for seq in valid_data.text]
x_test = [torch.tensor(preprocess_seq_str_2_int(seq), dtype=torch.long) for seq in data_test.text]

# In[16]:
x_train = pad_sequence(x_train, batch_first=True, padding_value=0)
x_valid = pad_sequence(x_valid, batch_first=True, padding_value=0)
x_test = pad_sequence(x_test, batch_first=True, padding_value=0)

# In[ ]:



# ## 准备数据集

# In[17]:
batchsize = 32

# In[18]:
class MyData(Dataset):
    def __init__(
        self, 
        ori_data, tfidf_feats, label,
    ):
        self.ori_data = ori_data
        self.tfidf_feats = tfidf_feats
        self.label = label
 
    def __len__(self):
        return len(self.ori_data)
 
    def __getitem__(self, idx):
        tuple_ = (
            self.ori_data[idx], 
            self.tfidf_feats[idx], 
            self.label[idx]
        )
        return tuple_

# In[19]:
train_loader = DataLoader(MyData(x_train, train_features, train_labels), batch_size=batchsize, shuffle=True,)
val_loader = DataLoader(MyData(x_valid, valid_features, valid_labels), batch_size=batchsize, shuffle=True,)
test_loader = DataLoader(MyData(x_test, test_features, test_labels), batch_size=batchsize, shuffle=False,) ## 这个不要shuffle，否则传到oj上面去就GG了。

# In[ ]:



# # 定义网络结构
# 
# 并行使用CNN+BiLSTM+Transformer：
# * CNN捕捉局部n-gram特征（kernel_size=3,5,7）
# * BiLSTM捕获长距离时序依赖
# * Transformer处理全局关系

# In[ ]:



# In[20]:
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        maxlen = x.size(1)
        positions = torch.arange(0, maxlen, device=x.device).unsqueeze(0).expand(x.size(0), x.size(1)) # torch.arange(0, maxlen, device=x.device).unsqueeze(0).expand_as(x)
        # print(positions.shape)
        return x + self.pos_emb(positions)

class TransformerModel(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim):
        super(TransformerModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.embedding_layer(x).transpose(0, 1)  # Transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer_block(x)
        x = x.transpose(0, 1)
        x = self.global_avg_pool(x.permute(0, 2, 1)).squeeze(-1)
        return x

# 加载预训练词向量的函数
import numpy as np
import gensim.downloader as api

def load_pretrained_embeddings(embedding_type='glove-wiki-gigaword-300', vocab_size=8000, embed_dim=128):
    """
    加载预训练词向量并创建嵌入矩阵
    
    参数:
    - embedding_type: 预训练词向量类型
    - vocab_size: 词汇表大小
    - embed_dim: 嵌入维度
    
    返回:
    - embedding_matrix: 嵌入矩阵，形状为 (vocab_size, embed_dim)
    """
    print(f"正在加载预训练词向量: {embedding_type}...")
    
    # 加载预训练词向量
    try:
        # 尝试直接加载gensim内置的预训练词向量
        word_vectors = api.load(embedding_type)
    except:
        # 如果失败，可能需要用户自己下载词向量文件并加载
        print("无法直接加载预训练词向量，请确保已下载并放置在正确位置。")
        print("使用随机初始化的嵌入矩阵。")
        return None
    
    # 创建嵌入矩阵
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    
    # 注意：这里假设词汇表中的索引与预训练词向量的词汇存在对应关系
    # 由于我们的数据中的token是整数编码，所以这里简化处理
    # 实际应用中可能需要构建词汇表映射
    
    # 对于在预训练词向量中找到的词，使用其预训练向量
    # 由于我们的数据中token是数字编码，这里我们将数字转换为字符串来匹配
    # 这是一个简化的处理方式，实际应用中可能需要更复杂的映射
    for i in range(1, min(vocab_size, len(word_vectors))):  # 跳过0（padding）
        word = str(i)
        if word in word_vectors:
            # 如果词向量维度与目标维度不同，进行截断或填充
            if len(word_vectors[word]) == embed_dim:
                embedding_matrix[i] = word_vectors[word]
            elif len(word_vectors[word]) > embed_dim:
                embedding_matrix[i] = word_vectors[word][:embed_dim]  # 截断
            else:
                # 填充
                embedding_matrix[i, :len(word_vectors[word])] = word_vectors[word]
    
    print(f"预训练词向量加载完成，词汇覆盖率: {(np.sum(np.abs(embedding_matrix).sum(axis=1) > 0.5) / vocab_size):.2%}")
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# In[21]:
## TextCNN：捕捉短时间的关系

# https://blog.51cto.com/u_15764210/6844118
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)
        
class TextCNN(nn.Module):
    def __init__(
        self, 
        embedding_dim=128, 
        kernel_sizes=[3, 4, 5, 6], num_channels=[256, 256, 256, 256], 
    ):
        '''
        ：param num_classes：输出维度（类别数num_Classes）
        ：param num_embeddings: size of the dictionary of embeddings，词典的大小（vocab_size），当num_embeddings<O，模型会去除embedding层
        ：param embedding_dim: the size of each embedding vector，词向量特征长度
        ：param kernel_sizes: CNN层卷积核大小
        ：param num_channels: CNN层卷积核通道数
        : return:
        '''
        assert len(kernel_sizes) == len(num_channels), "len(kernel_sizes) should be equal to len(num_channels)"
        super(TextCNN, self).__init__()
        # self.num_classes = num_classes
    
        # 卷积层
        self.cnn_layers = nn.ModuleList() # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=c, 
                    kernel_size=k
                ),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        # 最大池化层
        self.pool = GlobalMaxPool1d()
            
    def forward(self, input_):
        '''
        :param input: (batch_size, context_size, embedding_size(in_channels))
        :return:
        '''
        input_ = input_.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input_)
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        return y

# In[22]:
# BiLSTM 
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_dir = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers *(2 if self.bi_dir else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers *(2 if self.bi_dir else 1), x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]

# In[23]:
# 多层神经网络：
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(32, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(32, output_dim)
 
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)        
        return x

# # 单层神经网络：
# class SentimentClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#  
#     def forward(self, x):
#         return self.fc(x)

# In[24]:
# 定义模型
class FinalModel(nn.Module):
    def __init__(self, 
                 maxlen, vocab_size, embed_dim, num_heads, ff_dim,
                tcnn_ks = [3,5,7,10], tcnn_nc = [32,64,64,64],
                 lstm_hs = 128, lstm_nlyr = 4, lstm_bd = True,
                 use_pretrained_embeddings=False, embedding_type='glove-wiki-gigaword-300',
                 freeze_embeddings=False
                ):
        super(FinalModel, self).__init__()
        ## emb层
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # 使用预训练词向量初始化
        if use_pretrained_embeddings:
            embedding_matrix = load_pretrained_embeddings(
                embedding_type=embedding_type,
                vocab_size=vocab_size,
                embed_dim=embed_dim
            )
            if embedding_matrix is not None:
                self.token_emb.weight.data.copy_(embedding_matrix)
                # 是否冻结词向量，不进行更新
                if freeze_embeddings:
                    self.token_emb.weight.requires_grad = False
                    print("预训练词向量已冻结，不会在训练中更新")
                else:
                    print("预训练词向量未冻结，会在训练中更新")
            else:
                print("无法加载预训练词向量，使用随机初始化")
        
        ## Tsfm部分：
        self.tsfm = TransformerModel(
            maxlen, 
            vocab_size, 
            embed_dim, 
            num_heads, 
            ff_dim
        ) # embed_dim
        ## TextCNN部分：
        self.textcnn = TextCNN(
            embedding_dim = embed_dim, 
            kernel_sizes = tcnn_ks,
            num_channels = tcnn_nc
        ) # sum(tcnn_nc)
        ## BiLSTM:
        self.lstm = LSTMClassifier(
            input_size = embed_dim, 
            hidden_size = lstm_hs, 
            num_layers = lstm_nlyr,  
            bidirectional = lstm_bd
        ) # lstm_hs * (2 if lstm_bd else 1)
        self.mix = nn.Sequential(
            ## 基于这个数字 embed_dim + sum(tcnn_nc) + lstm_hs * (2 if lstm_bd else 1) ，做一个全连接神经网络吧。
            nn.Linear(embed_dim + sum(tcnn_nc) + lstm_hs * (2 if lstm_bd else 1), ff_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, type_of_class)
        )
        self.sc_net = SentimentClassifier(sc_input_dim, type_of_class)
        
    def forward(self, x, x_tfidf):
        x_ori = x
        x_emb = self.token_emb(x)#.to(device)
        ## Tsfm部分：
        x_tsfm = self.tsfm(x_emb) 
        ## TextCNN部分：
        x_tcnn = self.textcnn(x_emb)
        ## BiLSTM部分:
        x_lstm = self.lstm(x_emb)
        ## 综合：
        x_cat = torch.cat(
            [x_tsfm, x_tcnn, x_lstm], axis=1
        )        
        return F.log_softmax(self.mix(x_cat) + self.sc_net(x_tfidf), dim=-1)

# In[ ]:




fnl = FinalModel(maxlen, vocab_size, embed_dim=64, num_heads=8, ff_dim=64)
# In[ ]:



# # 构建模型以及训练

# In[25]:

# 使用GloVe预训练词向量初始化token_emb
model = FinalModel(
    maxlen, vocab_size, embed_dim=128, num_heads=8, ff_dim=128,
    use_pretrained_embeddings=True,  # 启用预训练词向量
    embedding_type='glove-wiki-gigaword-300',  # 使用GloVe词向量
    freeze_embeddings=False  # 不冻结词向量，允许在训练中更新
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# In[26]:
model

# In[27]:
# 训练和评估模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, input_tfidfs, targets in tqdm.tqdm(train_loader):
            inputs, input_tfidfs, targets = inputs.to(device), input_tfidfs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, input_tfidfs)
            loss = criterion(outputs, targets)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss/len(train_loader)}, Accuracy: {100.*correct/total}%')
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            total_predicted = []
            total_label = []
            for inputs, input_tfidfs, targets in tqdm.tqdm(val_loader):
                inputs, input_tfidfs, targets = inputs.to(device), input_tfidfs.to(device), targets.to(device)
                outputs = model(inputs, input_tfidfs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_predicted += list(predicted.cpu())
                total_label += list(targets.cpu())
        f1 = f1_score(total_label, total_predicted, average='macro')
                
        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100.*correct/total}%, f1 score is {f1}')

# In[28]:
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

# In[ ]:



# In[29]:
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=4) 

# In[30]:
# train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1) 

# In[31]:
# train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)

# In[32]:
# train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

# In[ ]:



# In[33]:
# run_finish()

# In[ ]:



# # 保存模型

# In[34]:
save_pickle_object(model, create_trained_models_path("ori_9319-20250208_1.pkl"))

# In[ ]:



# # 预测一下试试

# In[35]:
# model = load_pickle_object(create_trained_models_path("ori_9319-20250208_1.pkl"))

# In[ ]:



# In[36]:
with torch.no_grad():
    total_predicted = []
    for inputs, input_tfidfs, targets in tqdm.tqdm(test_loader):
        inputs, input_tfidfs, targets = inputs.to(device), input_tfidfs.to(device), targets.to(device)
        outputs = model(inputs, input_tfidfs)
        _, predicted = torch.max(outputs, 1)
        total_predicted += list(predicted)

# In[37]:
oot_rst = [int(x) for x in total_predicted]
len(oot_rst)

# In[38]:
store_data_to_newbasepath(pd.DataFrame({"label": oot_rst}), "rst-20250208_1", fmt="csv")

# In[ ]:




