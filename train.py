import time
import torch
import numpy as np
import argparse
import os
from logger import Logger
from loss import mmd, rec_loss
from pathlib import Path


# 训练模型函数
def train_model(args, model, optimizer, train_batch, vocab, max_epochs, device):
    logger = Logger(args)  # 初始化日志记录器
    print("Start Training")  # 打印开始训练的提示信息
    model.to(device=device)  # 将模型移动到指定设备 (GPU 或 CPU)
    start_time = time.time()  # 记录训练开始时间

    # 训练循环，遍历所有 epoch
    for epoch in range(1, max_epochs + 1):
        print("\nEpoch %d" % epoch)  # 打印当前 epoch
        model.train()  # 设置模型为训练模式

        # 遍历每个批次的数据
        for step, data_batch in enumerate(train_batch):
            data_batch = data_batch[0].float()  # 将数据转换为浮点类型
            data_batch = data_batch.to(device=device)  # 将数据移动到指定设备

            # 将数据输入模型，得到重构的结果和潜在变量
            x_recon, theta = model(data_batch, device)

            # 采样真实分布的潜在变量 z_true
            z_true = model.sample(batch_size=data_batch.shape[0]).to(device=device)

            optimizer.zero_grad()  # 将优化器的梯度清零
            loss_recon = rec_loss(data_batch, x_recon)  # 计算重构损失
            loss_mmd = mmd(theta, z_true, device)  # 计算 MMD 损失
            loss = loss_recon + loss_mmd  # 总损失为重构损失和 MMD 损失之和

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

        # 每 5 个 epoch 打印并保存当前的主题词
        if (epoch % 5) == 0:
            with torch.no_grad():  # 在不计算梯度的上下文中进行推理
                theta = torch.eye(args.num_topics, device=device)  # 创建一个单位矩阵，表示每个主题的 one-hot 编码
                topics = model.decoder(theta)  # 通过模型的解码器得到主题分布
                topk_value, topk_idx = torch.topk(topics, k=10, dim=1, largest=True)  # 取每个主题分布中前 10 个词
                words = [[vocab.id2word[y] for y in x] for x in topk_idx.tolist()]  # 将索引转换为单词
                words = '\n'.join(' '.join(x) for x in words)  # 将单词列表转换为字符串
                logger.save_text(words, f'{epoch:03d}.txt')  # 保存当前 epoch 的主题到日志文件

    end_time = time.time()  # 记录训练结束时间
    runtime_minutes = (end_time - start_time) / 60  # 计算训练的总时长
    print(f'runtime: {runtime_minutes} 分钟')  # 打印训练的总时长
    return
