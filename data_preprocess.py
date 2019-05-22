import pandas as pd
import csv
import cv2
import os

# 处理训练集
path = './kaggle_dogcat/train'
train_list = []
for file_name in os.listdir(path):
    label = file_name[:file_name.find('.')]
    idx = int(file_name[file_name.find('.') + 1:file_name.rfind('.')])
    label = 1 if label == 'dog' else 0 if label == 'cat' else ''
    print(idx, label)
    print(file_name)
    train_list.append([idx, label, file_name])
# 保存结果
pd.DataFrame(train_list).to_csv(
    './kaggle_dogcat/train.csv', index=False,
)
