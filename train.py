import config
from cat_dog_dataset import CatDogDataset
import torch
from torch.utils import data
from torch import optim
from torch import nn
import model


def val(model, criterion):
    """
    进行验证
    :param model: 训练好的模型
    :param criterion: loss计算器
    :return:
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0
        print('\nVALIDATING...')
        total_steps = len(val_dataset) // config.VAL_BATCH_SIZE
        for step, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            # 将gt转换成onehot编码
            target = torch \
                .zeros(config.VAL_BATCH_SIZE, config.NUM_CLASSES) \
                .scatter_(1, target.view(-1, 1), 1) \
                .cuda(non_blocking=True)
            # 前向传播
            outputs = model(inputs)

            # 计算loss
            loss = criterion(outputs, target)
            val_loss += loss.item()
        print("VAL_LOSS: {:.5f}".format(val_loss / (total_steps + 1)))


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
    for epoch in range(20):
        # misc
        running_loss = 0
        total_steps = len(train_dataset) // config.TRAIN_BATCH_SIZE
        print('EPOCH: {}'.format(epoch))
        # 训练模式
        ResNet.train()
        for step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            # 将gt转换成onehot编码
            target = torch \
                .zeros(config.TRAIN_BATCH_SIZE, config.NUM_CLASSES) \
                .scatter_(1, target.view(-1, 1), 1) \
                .cuda(non_blocking=True)
            # 清零optimizer
            optimizer.zero_grad()

            # 前向传播
            outputs = ResNet(inputs)

            # 计算loss
            loss = criterion(outputs, target)
            running_loss += loss.item()

            # 误差回传
            loss.backward()
            optimizer.step()

            # 输出中间结果
            if step % config.DUMP_INTERVAL == 0:
                done_proc = int(25 * step / total_steps)
                print(
                    '\r |{}|| E{}: {}/{} || TRAIN_LOSS: {:.5f}'.format(
                        '=' * done_proc + '=>' + ' ' * (25 - done_proc),
                        epoch, step, total_steps,
                        running_loss / config.DUMP_INTERVAL,
                    ),
                    end=''
                )
                running_loss = 0
        # 验证
        val(ResNet, criterion)


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
        shuffle=True, pin_memory=True, num_workers=16, drop_last=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=config.VAL_BATCH_SIZE,
        shuffle=False, pin_memory=True, num_workers=16, drop_last=False
    )

    train()
