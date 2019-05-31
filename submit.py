import itertools

import config
from cat_dog_dataset import CatDogDataset
import torch
from torch.utils import data
from torch import optim
from torch import nn
from torch.nn import functional
import model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def submit(ckpt_path=None):
    """
    进行验证
    :param model: 训练好的模型
    :param criterion: loss计算器
    :return:
    """
    # 构建网络
    # net = model.build_ResNet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    # net = model.build_inception(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    # net = model.build_senet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    net = model.build_nasnet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
    net = nn.DataParallel(net)

    # 载入checkpoint
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(
        'LOADING CKPT {} EPOCH:{} best:{:.5f}'.format(
            ckpt_path, start_epoch, best_val_loss
        )
    )

    with torch.no_grad():
        net.eval()
        val_loss = 0
        print('\nVALIDATING...')
        result = []
        for step, (inputs, target, img_id) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)

            # 前向传播 获取预测值
            outputs = net(inputs)

            # 使用softmax获取概率
            logits = functional.softmax(outputs, dim=1)
            result.append(
                torch.cat([img_id.cuda().float().view(-1, 1), logits], dim=1).cpu().numpy()
            )

            # plt.imshow(np.squeeze(inputs.cpu().numpy()).transpose([1, 2, 0]))
            # plt.title(str(logits.cpu().numpy()) + str(img_id.cpu()))
            # plt.show()
        # 写入结果
        # 整理结果
        result = np.vstack(result)[:, [0, 2]]
        if config.TEST_TTA:
            # 处理TTA
            result = result.reshape(-1, len(tta_test_dataset), 2)
            # 对TTA结果取均值
            result = np.mean(result, axis=0)
        pd.DataFrame(result).to_csv(
            './submit/{}{}_submit.csv'.format('TTA_' if config.TEST_TTA else '', ckpt_path[ckpt_path.rfind('/') + 1:]),
            header=['id', 'label'], index=False
        )


if __name__ == '__main__':
    if config.TEST_TTA:
        # 测试时数据增强
        tta_test_dataset = CatDogDataset(
            train_csv_path='./kaggle_dogcat/submit.csv',
            dataset_root_path='./kaggle_dogcat/test',
            augment=True,
            is_training=False
        )
        tta_test_loader = data.DataLoader(
            tta_test_dataset, batch_size=config.TEST_BATCH_SIZE,
            shuffle=False, pin_memory=True, num_workers=22, drop_last=False
        )
        # 倍增数据loader
        test_loader = tqdm(itertools.chain(
            tta_test_loader,
            tta_test_loader,
            tta_test_loader,
            tta_test_loader,
            tta_test_loader,
        ), total=len(tta_test_dataset) // config.TEST_BATCH_SIZE * 5)
    else:
        # 数据集
        test_dataset = CatDogDataset(
            train_csv_path='./kaggle_dogcat/submit.csv',
            dataset_root_path='./kaggle_dogcat/test',
            augment=False,
            is_training=False
        )

        # 载入数据集
        test_loader = tqdm(data.DataLoader(
            test_dataset, batch_size=config.TEST_BATCH_SIZE,
            shuffle=False, pin_memory=True, num_workers=22, drop_last=False
        ), total=len(test_dataset) // config.TEST_BATCH_SIZE)

    submit(ckpt_path='./ckpt/nasnet_large_best.pth')
