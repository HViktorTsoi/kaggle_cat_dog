import config
from cat_dog_dataset import CatDogDataset
import torch
from torch.utils import data
from torch import optim
from torch import nn
import model


def adjust_lr(optimizer, lr, epoch):
    """
    调整学习率
    :param optimizer: 优化器
    :param lr: 学习率
    :return:
    """
    # 每隔固定epoch衰减学习率
    decay_factor = epoch // config.TRAIN_LR_DECAY_STEP
    lr = lr * config.TRAIN_LR_DECAY ** decay_factor
    print('LR:{} DECAY_FACTOR:{} EPOCH:{}'.format(lr, decay_factor, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            val_loss /= total_steps + 1
        print("VAL_LOSS: {:.5f}".format(val_loss))
        return val_loss


def train(ckpt_path=None):
    # 构建网络
    # net = model.build_ResNet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    # net = model.build_inception(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    net = model.build_senet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    net = nn.DataParallel(net)

    # 载入checkpoint
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(
            'LOADING CKPT {} EPOCH:{} best:{:.5f}'.format(
                ckpt_path, start_epoch, best_val_loss
            )
        )
    else:
        start_epoch = 0
        best_val_loss = 1e9

    # Loss
    criterion = nn.BCEWithLogitsLoss().cuda()

    # 优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.TRAIN_LR,
        momentum=0.9, weight_decay=1e-4
    )
    optimizer = nn.DataParallel(optimizer).module
    for epoch in range(start_epoch, start_epoch + config.TRAIN_TOTAL_EPOCH):
        # misc
        running_loss = 0
        total_steps = len(train_dataset) // config.TRAIN_BATCH_SIZE
        print('\nEPOCH: {}'.format(epoch))

        # 调整学习率
        adjust_lr(optimizer, config.TRAIN_LR, epoch)

        # 训练模式
        net.train()
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
            outputs = net(inputs)

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
        val_loss = val(net, criterion)

        # 保存模型
        print('SAVING MODEL...')
        torch.save(
            {
                'model': net.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            },
            './ckpt/{}_{:04d}.pth'.format(config.MODEL_NAME, epoch)
        )
        # 选择验证集最优模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                },
                './ckpt/{}_best.pth'.format(config.MODEL_NAME, epoch)
            )
            print('BEST VAL LOSS:{}'.format(val_loss))


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

    # train(ckpt_path='./ckpt/resnet_0001.pth')
    train()
