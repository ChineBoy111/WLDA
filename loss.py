import torch
import numpy as np

# 最大均值差异（MMD）函数，用于比较两个分布 z 和 z_true
def mmd(z, z_true, device, t=0.1, kernel='diffusion'):
    eps = 1e-6  # 防止除零的小常数
    n, d = z.shape  # n: 批量大小, d: 数据的维度 (例如: 32 个样本，每个样本 50 维)

    # 如果使用 'tv' (全变差) 核函数，则基于 L1 范数计算距离
    if kernel == 'tv':
        sum_xx = torch.zeros(1).to(device)  # 初始化 z_true 的距离总和
        # 计算 z_true 中两两样本的 L1 距离的和
        for i in range(n):
            for j in range(i + 1, n):
                sum_xx = sum_xx + torch.norm(z_true[i] - z_true[j], p=1).to(device)
        sum_xx = sum_xx / (n * (n - 1))  # 根据样本对数进行归一化

        sum_yy = torch.zeros(1).to(device)  # 初始化 z 的距离总和
        # 计算 z 中两两样本的 L1 距离的和
        for i in range(z.shape[0]):
            for j in range(i + 1, z.shape[0]):
                sum_yy = sum_yy + torch.norm(z[i] - z[j], p=1).to(device)
        sum_yy = sum_yy / (z.shape[0] * (z.shape[0] - 1))  # 根据样本对数进行归一化

        sum_xy = torch.zeros(1).to(device)  # 初始化 z_true 和 z 之间的距离总和
        # 计算 z_true 和 z 之间两两样本的 L1 距离的和
        for i in range(n):
            for j in range(z.shape[0]):
                sum_xy = sum_xy + torch.norm(z_true[i] - z[j], p=1).to(device)
        sum_yy = sum_yy / (n * z.shape[0])  # 根据样本对数进行归一化
    else:
        # 如果使用扩散核函数，则计算 z 和 z_true 的内积并进行转换
        qx = torch.sqrt(torch.clamp(z_true, eps, 1))  # 对 z_true 的值进行裁剪和开方操作
        qy = torch.sqrt(torch.clamp(z, eps, 1))  # 对 z 的值进行裁剪和开方操作
        xx = torch.matmul(qx, qx.t())  # 计算 z_true 的 Gram 矩阵
        yy = torch.matmul(qy, qy.t())  # 计算 z 的 Gram 矩阵
        xy = torch.matmul(qx, qy.t())  # 计算 z_true 和 z 的交叉 Gram 矩阵

        # 定义扩散核函数
        def diffusion_kernel(a, tmpt, dim):
            return torch.exp(-torch.acos(a).pow(2) / tmpt)

        off_diag = 1 - torch.eye(n).to(device)  # 创建一个去掉对角线元素的矩阵，用于排除自身的相似性
        k_xx = diffusion_kernel(torch.clamp(xx, 0, 1 - eps), t, d - 1)  # 计算 z_true 的扩散核
        k_yy = diffusion_kernel(torch.clamp(yy, 0, 1 - eps), t, d - 1)  # 计算 z 的扩散核
        k_xy = diffusion_kernel(torch.clamp(xy, 0, 1 - eps), t, d - 1)  # 计算 z_true 和 z 的交叉扩散核
        sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))  # 计算 z_true 的核总和
        sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))  # 计算 z 的核总和
        sum_xy = 2 * k_xy.sum() / (n * n)  # 计算 z_true 和 z 的交叉核总和

    # 计算最终的 MMD 值
    mmd = sum_xx + sum_yy - sum_xy
    return mmd


# 重构损失函数，用于衡量原始输入 x_topic 与重构输出 x_reconst 之间的差异
def rec_loss(x_topic, x_reconst):
    logsoftmax = torch.log_softmax(x_reconst, dim=1)  # 计算重构输出的 log softmax
    rec_loss = torch.sum(-x_topic * logsoftmax, dim=1).mean()  # 计算交叉熵损失并取平均值

    # 使用 TF-IDF > 0 的词语数作为句子的长度
    s = torch.sum(x_topic > 0) / len(x_topic)  # 计算句子长度
    rec_alpha = 1.0 / (s * np.log(x_topic.shape[1]))  # 计算损失的归一化系数
    rec_loss = rec_alpha * rec_loss  # 加权后的重构损失
    return rec_loss
