import csv
import os
import pandas as pd

# 定义文件夹路径和CSV文件名
folder_path = '/home/lll/Alice/Dr.yu/code_classify/my_code/data/TCGA_test/TCGA_all/'
csv_file_name = "/home/lll/Alice/Dr.yu/code_classify/my_code/csv/TCGA_test/TCGA_all.csv"
csv_reader = csv.DictReader(csv_file_name)
# 获取文件夹中所有文件的文件名
file_names = os.listdir(folder_path)
label = []

j = 0
for i in range(len(file_names)):
    if file_names[i][:4] == 'norm' or file_names[i][:4] == 'Norm':
        label_one = 0
    else:
        j = j +1
        label_one = 1
    # label_one = file_names[i]
    label.append(label_one)
# 创建Pandas数据框
df = pd.DataFrame()
#
# # 将文件名添加到数据框中
df['image_names'] = file_names

# # 根据文件名生成第二列数据
df['file_value'] = label

# 将数据框写入CSV文件
df.to_csv(csv_file_name, index=False)
print('总数据量：', len(file_names),'正样本个数：', j ,'负样本个数：', len(file_names) - j )
print('====数据写入成功=====')

