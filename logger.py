#这个代码定义了一个 Logger 类，用于记录日志和保存模型。
#在初始化方法中，它创建了一个保存目录，用于保存文本和模型。
#log 方法用于打印日志信息，save_text 方法用于将主题信息保存到文本文件中，
# save_model 方法用于将模型保存到检查点目录。


from pathlib import Path
import os
import torch


class Logger(object):
    def __init__(self, args):
        self.filename = Path(args.txt_root)  # 创建一个Path对象，指向文本根目录
        self.filename = self.filename / str(args.num_topics)  # 在文本根目录下创建一个子目录，名称为主题数量
        os.makedirs(self.filename, exist_ok=True)  # 确保子目录存在，如果不存在则创建
        self.save_dir = self.filename  # 保存目录

    def log(self, msg):
        print(msg)  # 打印日志信息

    def save_text(self, topics, filename):
        with open(self.save_dir / filename, 'w') as f:  # 在保存目录下打开一个文件，如果文件不存在则创建
            f.write(topics)  # 将主题信息写入文件

    def save_model(self, model, ckpt_name):
        torch.save(model, self._ckpt_dir / ckpt_name)  # 将模型保存到检查点目录
