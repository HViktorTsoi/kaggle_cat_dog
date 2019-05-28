import pandas as pd
import numpy as np

result = pd.read_csv('./submit/resnet50_0019.pth_submit.csv').values

# 设置置信率
lower_thresh = 0.0047
upper_thresh = 0.994
result[np.where(result[:, 1] < lower_thresh), 1] = lower_thresh
result[np.where(result[:, 1] >= upper_thresh), 1] = upper_thresh
result = pd.DataFrame(result).astype({0: np.int, 1: np.float})

result.to_csv(
    './submit/R50REC_{}_{}.csv'.format(lower_thresh, upper_thresh),
    header=['id', 'label'],
    index=False
)
