import os
import random
import numpy as np
import torch

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

#
def set_global_seed(seed, env=None, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True    # 确保每次结果相同（可能会影响性能）
        torch.backends.cudnn.benchmark = False       # 禁用自动选择最优算法（确保可复现

    if env is not None:
        env.seed(seed)