import torch
import numpy as np
import random

def add_noise(input_x,ratio = 0.15):
    normal_noise = torch.normal(mean=torch.zeros_like(input_x), std=torch.ones_like(input_x))
    normal_prob = torch.rand_like(input_x)
    input_x[normal_prob<ratio] += normal_noise[normal_prob<ratio]     # 随机扰动  normal_prob<0.25相当于生成一个布尔型张量

    return input_x


def PDPL_loss(y_true, y_pred, y_sigma):

    return (torch.log(y_sigma) / 2 + (y_true - y_pred) ** 2 / (2 * y_sigma)).mean() + 5



def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子