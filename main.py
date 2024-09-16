import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_data  # 导入数据加载函数
from wlda_model import WLDA  # 导入 WLDA 模型
from train import train_model  # 导入模型训练函数
from logger import Logger  # 导入日志记录工具
import time

# 相关参数设置
alpha = 0  # 设置模型的 Dirichlet 超参数 alpha

# 设置命令行参数
parser = argparse.ArgumentParser(description='Wlda Topic Model')  # 创建参数解析器
parser.add_argument('-tag', '--tag', type=str, default='default',
                    help='训练的日志目录')  # 训练日志的保存目录
parser.add_argument('-nt', '--num_topics', type=int, default=14, help='主题数量')  # 设置主题数量
parser.add_argument('-ne', '--num_epochs', type=int, default=100, help='训练的轮数')  # 设置训练 epoch 数
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='批次大小')  # 设置每个批次的样本数量
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='学习率')  # 设置学习率
parser.add_argument('-dr', '--data_dir', type=str, default='datasets/dbpedia', help='数据目录')  # 数据集路径
parser.add_argument('-txt', '--txt_root', type=str, default='log', help='日志根目录')  # 保存日志的目录
args = parser.parse_args()  # 解析参数

# 选择硬件设备，cuda 表示使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 打印当前使用的设备 (GPU 或 CPU)

# 创建日志记录器
logger = Logger(args)

# 加载数据
vocab, data_iter = load_data(args.data_dir, args.batch_size)  # 调用数据加载函数，返回词汇表和数据迭代器

# 加载 WLDA 模型
model = WLDA(len(vocab), args.num_topics,alpha)  # 初始化模型，传入词汇表大小、主题数量和 alpha 参数

# 设置优化器，使用 Adam 优化器并设置学习率
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# 训练模型
train_model(args, model, optimizer, data_iter, vocab, args.num_epochs, device)  # 调用训练函数进行模型训练
