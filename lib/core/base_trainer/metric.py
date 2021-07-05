import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('.')
import numpy as np
import torch.nn as nn
from train_config import config as cfg

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


