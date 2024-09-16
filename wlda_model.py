import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# w-lda 模型
class WLDA(nn.Module):
    def __init__(self, topic_vocab=10000, num_topic=20, alpha=0.1):
        super(WLDA, self).__init__()
        self.topic_vocab = topic_vocab  # 词汇表大小
        self.num_topic = num_topic  # 主题数量
        self.alpha = alpha  # 超参数，用于平衡真实狄利克雷分布和模型输出

        # encoder全连接层，参数可调
        self.encoder1 = nn.Linear(self.topic_vocab, 256)  # 第一个全连接层，输入维度为词汇表大小，输出维度为256
        self.encoder2 = nn.Linear(256, self.num_topic)  # 第二个全连接层，输入维度为256，输出维度为主题数量

        # 正则化，设置节点个数
        self.en_BN1 = nn.BatchNorm1d(256)  # 批量归一化层，用于正则化，输入维度为256

        # decoder全连接层，参数可调
        self.decoder1 = nn.Linear(self.num_topic, 256)  # 第一个全连接层，输入维度为主题数量，输出维度为256
        self.decoder2 = nn.Linear(256, self.topic_vocab)  # 第二个全连接层，输入维度为256，输出维度为词汇表大小

    def encoder(self, x):
        z = self.en_BN1(self.encoder1(x))  # 通过第一个全连接层和批量归一化层
        z = F.leaky_relu(z)  # 使用leaky relu激活函数
        z = self.encoder2(z)  # 通过第二个全连接层
        return z

    # 真实狄利克雷分布采样
    def sample(self, dirichlet_alpha=0.1, batch_size=8):
        z_true = np.random.dirichlet(
            np.ones(self.num_topic) * dirichlet_alpha, size=batch_size)  # 从狄利克雷分布中采样
        z_true = torch.from_numpy(z_true).float()  # 将数组改为张量
        return z_true

    def decoder(self, z):
        x_re = self.decoder1(z)  # 通过第一个全连接层
        x_re = F.leaky_relu(x_re)  # 使用leaky relu激活函数
        x_re = self.decoder2(x_re)  # 通过第二个全连接层
        return x_re

    # 前向传播，模型的输入输出
    def forward(self, x_topic, device):
        z = self.encoder(x_topic)  # 编码输入
        z_true = self.sample(batch_size=x_topic.shape[0]).to(device=device)  # 采样真实狄利克雷分布
        theta = (1 - self.alpha) * F.softmax(z, dim=1) + self.alpha * z_true  # 计算混合狄利克雷分布
        x_reconst = self.decoder(theta)  # 解码得到重构输出
        return x_reconst, theta  # 返回重构输出和混合狄利克雷分布
