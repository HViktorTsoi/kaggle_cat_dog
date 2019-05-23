import math

from torchvision import models
from torch import nn

import config


def build_ResNet(pretrained, num_classes):
    """
    构建resnet模型
    :param pretrained: 是否预训练
    :param num_classes: 类别数
    :return: 模型
    """
    # 初始化模型
    model = models.resnet18(pretrained=pretrained)
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
