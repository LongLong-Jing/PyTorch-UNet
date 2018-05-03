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
from dataloader import ImageNet_Dataloader
from PIL import Image
from network import netG, netD

batch_size = 80
netD = netD().cuda(0)
netD.train(True)

optimizer = optim.Adam(netD.parameters(),lr=0.005, betas=(0.5, 0.999))
criterion = nn.BCELoss(size_average=True).cuda(0)

data_path = './data/images/'
dst = ImageNet_Dataloader(data_path, is_transform=True)
print('length of the dataset', len(dst))
trainloader = data.DataLoader(dst, batch_size=batch_size)
step_index = 0


for epo_num in range(100):
    for i, data in enumerate(trainloader):
        # real data
        rgb_img, mask = data



        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))





for epo_num in range(100):
    for i, data in enumerate(trainloader):
        # real data
        rgb_img, mask = data

        # fake data

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
        if (step_index%5000) == 0:
            lr = 0.0001 * (0.1 ** (step_index // 5000))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        #accuracy
        if (step_index%5) ==0:
            print('Step: {}, loss: {}'.format(step_index, loss.data[0]))
        if((step_index+1)%500) ==0:
            print('----------------- Save The Network ------------------------\n')
            with open('./' + str(step_index+1)+'netD.ckpt', 'wb') as f:
                torch.save(netD, f)
