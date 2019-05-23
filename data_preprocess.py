import pandas as pd
import csv
import cv2
import os
import skimage.io
import numpy as np


def process_train():
    # 处理训练集
    path = './kaggle_dogcat/train'
    dataset_list = []
    img_size = []
    for file_name in os.listdir(path):
        label = file_name[:file_name.find('.')]
        idx = int(file_name[file_name.find('.') + 1:file_name.rfind('.')])
        label = 1 if label == 'dog' else 0 if label == 'cat' else ''
        print(idx, label)
        print(file_name)
        dataset_list.append([idx, label, file_name])
        # img = cv2.imread(os.path.join(path, file_name))
        # print(img.shape)
        # img_size.append([img.shape[1], img.shape[0]])

    # 分析图像大小
    # img_size = np.array(img_size)
    # print(np.histogram(img_size[:, 0]))
    # print(np.histogram(img_size[:, 1]))

    # 划分训练集和验证集 保证验证集能被batch整除
    TRAIN_VAL_SPLIT = 25000 - 512 * 3
    # 随机排序
    dataset_list = np.array(dataset_list)[
                   np.random.permutation(len(dataset_list)), :
                   ]
    train_list = dataset_list[:TRAIN_VAL_SPLIT, :]
    val_list = dataset_list[TRAIN_VAL_SPLIT:, :]
    # 保存结果
    pd.DataFrame(train_list).to_csv(
        './kaggle_dogcat/train.csv', index=False,
    )
    pd.DataFrame(val_list).to_csv(
        './kaggle_dogcat/val.csv', index=False,
    )


def process_test():
    # 处理训练集
    path = './kaggle_dogcat/test'
    test_list = []
    img_size = []
    for file_name in os.listdir(path):
        idx = int(file_name[:file_name.find('.')])
        print(idx)
        print(file_name)
        test_list.append([idx, -1, file_name])
        # img = cv2.imread(os.path.join(path, file_name))
        # print(img.shape)
        # img_size.append([img.shape[1], img.shape[0]])

    # 分析图像大小
    # img_size = np.array(img_size)
    # print(np.histogram(img_size[:, 0]))
    # print(np.histogram(img_size[:, 1]))
    test_list = np.array(test_list)
    # 排序
    test_list = test_list[np.argsort(np.int_(test_list[:, 0])), :]
    # 保存结果
    pd.DataFrame(test_list).to_csv(
        './kaggle_dogcat/submit.csv', index=False,
    )


if __name__ == '__main__':
    process_test()
