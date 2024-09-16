import os

# 创建一个字典，键为数字字符串，值为对应的文件名后缀
dic = {'1': '005', '2': '010', '3': '015', '4': '020', '5': '025', '6': '030', '7': '035', '8': '040', '9': '045',
       '10': '050'}

# 循环遍历从 1 到 10 的整数
for i in range(1, 11):
    param = str(i)  # 将循环变量 i 转换为字符串形式
    param = dic[param]  # 从字典中获取对应的文件名后缀

    # 使用 os.system 执行命令行，运行指定的 Python 脚本
    # 在这里，将 topic 文件路径中的占位符 %s 替换为从字典中获得的文件名后缀
    os.system("python3 /home/dongxinwang/Palmetto/cc.py -t /home/dongxinwang/w_lda/log/20news-30-run01/topic/%s.txt" % (param))
