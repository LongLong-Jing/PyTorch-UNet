import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
import numpy as np
import os
from torch.nn import init
from torch.utils import data
import torchvision
import random
from cStringIO import StringIO
import time
import scipy.misc
# from PIL import Image
from matplotlib import pyplot as plt
from dataloader import ImageNet_Dataloader
from PIL import Image
from network import netG, netD

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

batch_size = 40
netD = netD().cuda(0)
netD.apply(weights_init_xavier)
netD.train(True)

optimizer = optim.Adam(netD.parameters(),lr=0.005, betas=(0.5, 0.999))
criterion = nn.BCELoss(size_average=True).cuda(0)

data_path = '/media/tensor-server/ee577d95-535d-40b2-88db-5546defabb74/imagenet20_rgbmsk_v0/ImageNet20/'
dst = ImageNet_Dataloader(data_path, is_transform=True)
print('length of the dataset', len(dst))
trainloader = data.DataLoader(dst, batch_size=batch_size)
step_index = 0
for epo_num in range(100):
    for i, data in enumerate(trainloader):
        rgb_img, mask = data
        step_index = step_index + 1
        rgb_img, mask = Variable(rgb_img).cuda(0), Variable(mask).cuda(0)

        netD.zero_grad()
        predict_msk = netD(rgb_img)

        predict_msk = predict_msk.squeeze(1)
        predict_msk = predict_msk.view(-1)
        flat_mask = mask.view(-1)

        loss = criterion(predict_msk,flat_mask)
        loss.backward()
        optimizer.step()
        if (step_index%1000) == 0:
            lr = 0.0001 * (0.1 ** (step_index // 1000))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        #accuracy
        if (step_index%5) ==0:
            print('Step: {}, loss: {}'.format(step_index, loss.data[0]))
        if((step_index+1)%1000) ==0:
            print('----------------- Save The Network ------------------------\n')
            with open('./' + str(step_index+1)+'netD.ckpt', 'wb') as f:
                torch.save(netD, f)
