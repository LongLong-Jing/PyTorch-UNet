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
import torchvision.utils as vutils

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

# batch_size = 40
netD = netD().cuda(0)
netD.apply(weights_init_xavier)
netD.train(True)

netG = netG().cuda(0)
netG.apply(weights_init_xavier)
netG.train(True)

# optimizer = optim.Adam(netD.parameters(),lr=0.005, betas=(0.5, 0.999))

optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0005, betas=(0.5, 0.999))

criterion = nn.BCELoss(size_average=True).cuda(0)

data_path = '/media/tensor-server/ee577d95-535d-40b2-88db-5546defabb74/imagenet20_rgbmsk_v0/ImageNet20/'
dst = ImageNet_Dataloader(data_path, is_transform=True)
print('length of the dataset', len(dst))
trainloader = data.DataLoader(dst, batch_size=24,shuffle=True)
step_index = 0

real_label = 1
fake_label = 0

for epo_num in range(500):
    for i, data in enumerate(trainloader):
        real_img, real_mask = data
        print(real_img)
        step_index = step_index + 1
        real_img, real_mask = Variable(real_img).cuda(0), Variable(real_mask).cuda(0)
        batch_size = real_img.size(0)

        input = torch.FloatTensor(batch_size, 3, 128, 128)
        noise = torch.FloatTensor(batch_size, 100, 1, 1)
        fixed_noise = torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(batch_size,1,16,16)
        fixed_noise = Variable(fixed_noise).cuda(0)


        netD.zero_grad()

        # train with real data
        labelv = Variable(label.fill_(real_label)).cuda(0)

        fake_prob, forg_prob = netD(real_img)

        real_forg_prob = forg_prob.squeeze(1)
        real_forg_prob = real_forg_prob.view(-1)
        real_mask = real_mask.view(-1)
        seg_loss = criterion(real_forg_prob, real_mask)
        errD_real = criterion(fake_prob, labelv)

        D_x = fake_prob.data.mean()

        # train with fake
        # random initialize the noise vector

        noisev = Variable(noise.normal_(0, 1)).cuda(0)
        fake_img = netG(noisev)

        labelv = Variable(label.fill_(fake_label)).cuda(0)
        fake_prob, _ = netD(fake_img.detach())
        errD_fake = criterion(fake_prob, labelv)
        D_G_z1 = fake_prob.data.mean()
        bce_loss = (errD_real + errD_fake)*0.5
        errD =  bce_loss*0.1 + seg_loss
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label)).cuda(0)  # fake labels are real for generator cost
        fake_prob, fake_forg_prob = netD(fake_img)
        errG = criterion(fake_prob, labelv)
        errG.backward()
        D_G_z2 = fake_prob.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f  BCE_D: %.4f SEG_D:  %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epo_num, 100, step_index%len(trainloader), len(trainloader),
                 errD.data[0],bce_loss, seg_loss, errG.data[0], D_x, D_G_z1, D_G_z2))
        if step_index % 200 == 0:
            vutils.save_image(real_img.data,
                    './samples/real_samples_epoch_%06d.png' % (step_index),
                    normalize=True)
            vutils.save_image(forg_prob.data,
                    './samples/real_foreground_samples_epoch_%06d.png' % (step_index),
                    normalize=True)
            vutils.save_image(fake_forg_prob.data,
                    './samples/fake_forg_prob_samples_epoch_%06d.png' % (step_index),
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    './samples/fake_samples_epoch_%06d.png' % (step_index),
                    normalize=True)
