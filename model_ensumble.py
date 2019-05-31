import pandas as pd
import numpy as np


def limit_score(result, name, lower_thresh=0.0047, upper_thresh=0.994):
    """
    限制置信率
    :param result:
    :return:
    """
    result[np.where(result[:, 1] < lower_thresh), 1] = lower_thresh
    result[np.where(result[:, 1] >= upper_thresh), 1] = upper_thresh
    result = pd.DataFrame(result).astype({0: np.int, 1: np.float})

    result.to_csv(
        './submit/{}_{}_{}.csv'.format(name, lower_thresh, upper_thresh),
        header=['id', 'label'],
        index=False
    )


def ensumble_result():
    result_list = [
        './submit/resnet50_0019.pth_submit.csv',
        './submit/TTA_resnet50_0019.pth_submit.csv',
        './submit/se_res_net_best.pth_submit.csv',
        './submit/TTA_ORIGIN_se_res_net_best.csv',
        './submit/nasnet_large_best.pth_submit.csv',
        './submit/TTA_nasnet_large_best.pth_submit.csv',

    ]
    result_list = [pd.read_csv(f).values for f in result_list]
    # 合并结果
    ids = result_list[0][:, 0]
    scores = np.hstack([r[:, 1].reshape(-1, 1) for r in result_list])
    # 取均值
    scores = np.mean(scores, axis=1)
    # scores = np.max(scores, axis=1)
    # 输出结果
    result = np.vstack([ids, scores]).T
    limit_score(result, name='TTA_ensumble_mean_ser50nas_all')


def process_tta():
    """
    处理tta
    :return:
    """
    result = pd.read_csv('./submit/TTA_se_res_net_best.pth_submit.csv').values
    # 堆叠
    result = result.reshape(-1, 12500, 2)
    # 对TTA结果取均值
    result = np.mean(result, axis=0)
    result = pd.DataFrame(result).astype({0: np.int, 1: np.float})

    result.to_csv(
        './submit/{}.csv'.format('TTA_ORIGIN_se_res_net_best'),
        header=['id', 'label'],
        index=False
    )


if __name__ == '__main__':
    # limit_score(
    #     result=pd.read_csv('./submit/nasnet_large_best.pth_submit.csv').values,
    #     name='nasnet'
    # )
    ensumble_result()
    # process_tta()
