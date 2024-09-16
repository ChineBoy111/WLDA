dataset=20news  # 设置数据集名称为 20news

# 循环版本，遍历不同的主题数量并运行模型训练与评估
for i in {20,30,50,75,100}
do
  #执行 Python 脚本，训练模型并指定数据集目录、日志目录以及主题数量
  python -u -m main --data_dir=datasets/${dataset} --txt ./log/${dataset}-${i}-run01 --num_topics ${i}

  #执行另一个 Python 脚本进行模型评估，使用上一个步骤生成的日志目录
  #python -u ../cc.py -d ./log/${dataset}-${i}-run01/${i}
done

# 单次运行版本，设置主题数量为 20 并训练模型
#
#i=20  # 设置主题数量为 20
#python -u -m main --data_dir=datasets/${dataset} --txt ./log/${dataset}-runtime --num_topics ${i}
# 执行 Python 脚本，指定数据目录为 datasets/${dataset}，日志目录为 ./log/${dataset}-runtime，主题数量为 20

# 示例代码，用于运行 dbpedia 数据集的模型训练与评估
# python -u -m main --data_dir=datasets/dbpedia --txt ./log/dbpedia-14-run04 --num_topics 14
# python -u ../cc.py -d ./log/dbpedia-14-run04/14  # 运行 cc.py 对模型结果进行评估
