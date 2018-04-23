# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
import numpy as np
import os
from torch.utils import data
import torchvision
import random
from cStringIO import StringIO
import time
import scipy.misc
# from PIL import Image
from matplotlib import pyplot as plt
# from single_data_loader import ImageNet_Dataloader
from PIL import Image
import cv2
import math
from dataloader import ImageNet_Dataloader
from PIL import Image
from network import netG, netD

batch_size = 200
iterations = 60000

with open('./1000netD.ckpt', 'rb') as f:
    netD = torch.load(f).cuda(0)


data_path = '/media/tensor-server/ee577d95-535d-40b2-88db-5546defabb74/imagenet20_rgbmsk_v0/ImageNet20/'
dst = ImageNet_Dataloader(data_path, is_transform=True)
print('length of the dataset', len(dst))
trainloader = data.DataLoader(dst, batch_size=1)
step_index = 0

for i, data in enumerate(trainloader):
    rgb_img, mask = data
    step_index = step_index + 1
    rgb_img, mask = Variable(rgb_img).cuda(0), Variable(mask).cuda(0)
    predict_msk = netD(rgb_img)
    print('+++++++++++++++++++++++++++')
    # print(rgb_img.size())
    # print(predict_msk.size())
    predict_msk = predict_msk.data.cpu().numpy()
    rgb_img = rgb_img.data.cpu().numpy()
    org_mask = mask.data.cpu().numpy()
    predict_msk = predict_msk[0,:,:,:]
    rgb_img = rgb_img[0,:,:,:]
    predict_msk[predict_msk>0.5] = 1
    print(mask.shape)
    print(rgb_img.shape)
    predict_msk = np.transpose(predict_msk,(1,2,0))
    rgb_img = np.transpose(rgb_img,(1,2,0))
    org_mask = np.transpose(org_mask,(1,2,0))
    cv2.imshow('mask',predict_msk)
    cv2.imshow('rgb_img',rgb_img)
    cv2.imshow('org_mask',org_mask)
    cv2.waitKey(500)
