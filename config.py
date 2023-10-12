import os
from easydict import EasyDict as edict
import time
import torch

# init
c = edict()
cfg = c

c.EPOCH = 1
c.BATCH_SIZE = 16
c.LR = 0.001
#这里只需传入cifar-10-batches-py数据集（官网下载）所在的根路径即可
c.ROOT = "G:\cifar10\data"
c.NET = 'AlexNet' #'AlexNet', 'VGG', 'Res50'
c.TEST_MODEL = 'Res50.pt' #'AlexNet.pt', 'VGG.pt', 'Res50.pt'

