import os
import random

from torch.utils import data
from torchvision import transforms
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters
import numpy as np

import config


class CatDogDataset(data.Dataset):
    def __init__(self, train_csv_path, dataset_root_path, augment=True):
        # 初始化数据列表和数据集根目录
        self.data_list = pd.read_csv(train_csv_path).values
        self.dataset_root_path = dataset_root_path
        self.augment = augment
        self.transforme = transforms.Compose([
            transforms.ToPILImage(),  # 先转PIL
            transforms.Resize([config.IMAGE_HEIGHT, config.IMAGE_WIDTH]),
            transforms.ToTensor(),  # 再转Tensor
            # transforms.Normalize(
            #     mean=(0.485, 0.456, 0.406),
            #     std=(0.229, 0.224, 0.225)
            # )
        ])

    def __getitem__(self, idx):
        """
        获取idx对应的图像和label
        :param idx: 图像标识
        :return: 图像，对应label
        """
        _, target, file_name = self.data_list[idx]
        img_path = os.path.join(self.dataset_root_path, file_name)
        img = skimage.io.imread(img_path)
        if self.augment:
            img = self.process_augment(img)
        # torch transform
        img = self.transforme(img)
        return img, target

    def __len__(self):
        """
        获取数据集总长度
        :return: 数据集总长度
        """
        return len(self.data_list)

    def process_augment(self, image):
        """
        进行数据增强
        :param image: 原图像
        :return: 变换后的图像
        """
        augmentor = augmenters.Sequential([
            augmenters.OneOf([
                augmenters.Affine(rotate=(-45, 45)),
                augmenters.Affine(rotate=(-90, 90)),
                augmenters.Affine(shear=(-16, 16)),
                augmenters.Fliplr(1),
                augmenters.GaussianBlur(1.0),
            ])
        ])
        augmented_img = augmentor.augment_image(image)
        return augmented_img


if __name__ == '__main__':
    # 单元测试
    cat_dog_dataset = CatDogDataset(
        train_csv_path='./kaggle_dogcat/train.csv',
        dataset_root_path='./kaggle_dogcat/train'
    )
    for _ in range(10):
        img, target = cat_dog_dataset[0]
        plt.imshow(np.transpose(img.numpy(), [1, 2, 0]))
        plt.show()
        print(target)
