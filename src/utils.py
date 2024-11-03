# this code is modified from https://github.com/dahyun-kang/renet/blob/main/common/utils.py

import os
import torch
import math
import pprint
import random
import numpy as np
from termcolor import colored


class Meter:
    def __init__(self):
        self.list = []
    def update(self, item):
        self.list.append(item)
    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None
    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci
    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()

def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.
    
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def load_model(model, dir):
    checkpoint = torch.load(dir)
    model.load_state_dict(checkpoint['model'])
    # return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow
