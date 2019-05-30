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


def submit(ckpt_path=None):
    """
    进行验证
    :param model: 训练好的模型
    :param criterion: loss计算器
    :return:
    """
    # 构建网络
    net = model.build_ResNet(pretrained=True, num_classes=config.NUM_CLASSES).cuda()
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
        total_steps = len(test_dataset)
        result = []
        for step, (inputs, target, img_id) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)

            # 前向传播 获取预测值
            outputs = net(inputs)

            # 使用softmax获取概率
            logits = functional.softmax(outputs, dim=1)
            print(img_id.item(), logits)
            result.append([
                img_id.item(), np.squeeze(logits.cpu().numpy())[1]
            ])

            # plt.imshow(np.squeeze(inputs.cpu().numpy()).transpose([1, 2, 0]))
            # plt.title(str(logits.cpu().numpy()) + str(img_id.cpu()))
            # plt.show()
            # 写入结果
        pd.DataFrame(result).to_csv(
            './submit/{}_submit.csv'.format(ckpt_path[ckpt_path.rfind('/'):]),
            header=['id', 'label'], index=False
        )


if __name__ == '__main__':
    # 数据集
    test_dataset = CatDogDataset(
        train_csv_path='./kaggle_dogcat/submit.csv',
        dataset_root_path='./kaggle_dogcat/test',
        augment=False,
        is_training=False
    )

    # 载入数据集
    test_loader = data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, pin_memory=True, num_workers=1, drop_last=False
    )

    submit(ckpt_path='./ckpt/resnet50_0019.pth')
