from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import TensorDataset, DataLoader

# 词汇表类，用于加载和处理词汇数据
class Vocabulary(object):
    def __init__(self, path):
        with open(path) as f:
            # 读取词汇文件中的每一行并去除空行和空白字符，存储为 id2word 列表
            self.id2word = [x.strip() for x in f]
        # 通过枚举词汇，创建一个从单词到索引的字典 word2id
        word2id = {x: i for i, x in enumerate(self.id2word)}
        # 检查词汇表的大小是否一致，保证词汇表的正确性
        assert len(word2id) == len(self.id2word)

    def __len__(self):
        return len(self.id2word)  # 返回词汇表的大小

# 加载训练数据函数，返回 TF-IDF 格式的数据矩阵
def load_train_data(path):
    """
    Returns: 返回一个 TF-IDF 格式的矩阵，大小为 (n_documents, vocab_size)。
    """
    bow = load_npz(path)  # 从文件中加载稀疏的词袋 (BoW) 矩阵
    # 初始化 TF-IDF 转换器，使用 L1 正则化
    tfidf_transformer = TfidfTransformer(norm='l1')
    # 将词袋矩阵转换为 TF-IDF 格式
    tfidf = tfidf_transformer.fit_transform(bow)
    tfidf = tfidf.astype(np.float32).toarray()  # 转换为 numpy 数组格式并设置数据类型为 float32
    tfidf = torch.tensor(tfidf)  # 转换为 PyTorch 张量
    return tfidf  # 返回 TF-IDF 张量

# 加载词向量数据的函数
def load_word_vectors(path):
    with open(path) as f:
        # 每一行代表一个单词的词向量，将其转换为浮点数张量
        wv = torch.tensor([[float(y) for y in x.split()] for x in f], dtype=torch.float32)
    return wv  # 返回词向量张量

# 加载数据和词汇表，并返回词汇表和数据加载器
def load_data(data_dir, batch_size):
    data_dir = Path(data_dir)  # 将数据目录转换为 Path 对象，便于操作文件路径
    vocab = Vocabulary(data_dir / 'vocab.txt')  # 加载词汇表

    data = load_train_data(data_dir / 'train.bow.npz')  # 加载训练数据 (词袋矩阵)
    # 使用 TensorDataset 包装数据，创建 PyTorch 数据集
    dataset = TensorDataset(data)
    # 创建 DataLoader 以便批量加载数据，并设置批次大小、打乱数据、并行读取等参数
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    return vocab, data_loader  # 返回词汇表对象和数据加载器
