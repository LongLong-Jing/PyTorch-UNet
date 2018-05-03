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
from matplotlib import pyplot as plt
from dataloader import ImageNet_Dataloader
from PIL import Image
from network import netG, netD
import torchvision.utils as vutils


# random initialization method
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



# batch_size = 24
netD = netD().cuda(0)
netD.apply(weights_init_xavier)
netD.train(True)


optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(0.5, 0.999))

#Binary Cross Entropy
criterion = nn.BCELoss(size_average=True).cuda(0)

#The path of the data
data_path = './data/images/'
dst = ImageNet_Dataloader(data_path, is_transform=True)
print('length of the dataset', len(dst))
trainloader = data.DataLoader(dst, batch_size=24,shuffle=True)
step_index = 0

real_label = 1
fake_label = 0

# 500 Epoches
for epo_num in range(500):
    for i, data in enumerate(trainloader):
        real_img, real_mask = data
        step_index = step_index + 1
        real_img, real_mask = Variable(real_img).cuda(0), Variable(real_mask).cuda(0)
        # batch_size = real_img.size(0)

        netD.zero_grad()

        pred_mask_tmp = netD(real_img)

        pred_mask = pred_mask_tmp.squeeze(1)
        pred_mask = pred_mask.view(-1)
        real_mask = real_mask.view(-1)
        seg_loss = criterion(pred_mask, real_mask)

        seg_loss.backward()
        optimizerD.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f '
              % (epo_num, 100, step_index%len(trainloader), len(trainloader),
                 seg_loss.data[0]))

        #save the image and its predicted masks every 50 iterations
        if step_index % 10 == 0:
            vutils.save_image(real_img.data,
                    './samples/real_samples_epoch_%06d.png' % (step_index),
                    normalize=True)
            vutils.save_image(pred_mask_tmp.data,
                    './samples/real_foreground_samples_epoch_%06d.png' % (step_index),
                    normalize=True)