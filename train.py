import config
from cat_dog_dataset import CatDogDataset
import torch
from torch.utils import data
from torch import optim
from torch import nn
import model


def train():
    # 构建网络
    ResNet = model.build_ResNet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()

    # Loss
    criterion = nn.BCEWithLogitsLoss().cuda()

    # 优化器
    optimizer = optim.SGD(
        ResNet.parameters(),
        lr=config.TRAIN_LR,
        momentum=0.9, weight_decay=1e-4
    )
    # 训练模式
    ResNet.train()
    for epoch in range(20):
        for step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            # 将gt转换成onehot编码
            target = torch \
                .zeros(config.TRAIN_BATCH_SIZE, config.NUM_CLASSES) \
                .scatter_(1, target.view(-1, 1), 1) \
                .cuda()
            # 清零optimizer
            optimizer.zero_grad()

            # 前向传播
            outputs = ResNet(inputs)

            # 计算loss
            loss = criterion(outputs, target)
            print(loss.item())

            # 误差回传
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    # 数据集
    train_dataset = CatDogDataset(
        train_csv_path='./kaggle_dogcat/train.csv',
        dataset_root_path='./kaggle_dogcat/train',
        augment=True
    )
    val_dataset = CatDogDataset(
        train_csv_path='./kaggle_dogcat/val.csv',
        dataset_root_path='./kaggle_dogcat/train',
        augment=False
    )

    # 载入数据集
    train_loader = data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True, pin_memory=True, num_workers=16
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False, pin_memory=True, num_workers=16
    )

    train()
