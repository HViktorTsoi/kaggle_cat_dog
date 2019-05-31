import math

from torchvision import models
from torch import nn
from pretrainedmodels.models import bninception, se_resnet50

import config


def build_ResNet(pretrained, num_classes):
    """
    构建resnet模型
    :param pretrained: 是否预训练
    :param num_classes: 类别数
    :return: 模型
    """
    # 初始化模型
    model = models.resnet50(pretrained=pretrained)
    # 获取预训练的resnet
    fc_features = model.fc.in_features

    # 根据本任务构造新的输出
    # 经过resnet降采样之后的大小变化
    down_sample_calc = lambda x: math.ceil(x / 32 - 7 + 1)
    # 最后一层
    model.fc = nn.Linear(
        down_sample_calc(config.IMAGE_WIDTH)
        * down_sample_calc(config.IMAGE_HEIGHT)
        * fc_features,
        num_classes
    )
    return model


def build_inception(pretrained, num_classes):
    """
    构建inception模型
    :param pretrained: 是否预训练
    :param num_classes: 类别数
    :return: 模型
    """
    model = bninception(pretrained='imagenet')
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
    )
    return model


def build_senet(pretrained, num_classes):
    """
    构建senet模型
    :param pretrained: 是否预训练
    :param num_classes: 类别数
    :return: 模型
    """
    model = se_resnet50(pretrained='imagenet')

    # 获取预训练的resnet
    fc_features = model.last_linear.in_features

    # 根据本任务构造新的输出
    # 经过resnet降采样之后的大小变化
    down_sample_calc = lambda x: math.ceil(x / 32 - 7 + 1)
    # 最后一层
    new_last_in_feature = down_sample_calc(config.IMAGE_WIDTH) \
                          * down_sample_calc(config.IMAGE_HEIGHT) \
                          * fc_features
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(new_last_in_feature),
        nn.Dropout(0.5),
        nn.Linear(new_last_in_feature, num_classes),
    )
    return model
