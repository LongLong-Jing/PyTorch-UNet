import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

# Generator network to generate fake images from noise
# Input is 1*100 noise vector, the output is 128*128 images
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128,     64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(64,     32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    32,      3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return output
# The output is range from 0 to 1


# Discriminator Without Max Pooling
class netD(nn.Module):

    def __init__(self, num_classes=1):
        super(netD, self).__init__()

        #128*128
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.down1_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        #64*64
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)
        self.down2_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        #32*32
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)
        self.down3_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 16*16
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.down4_pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 512*8*8
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),)

        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))

        self.up4 = nn.Sequential(
            nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),)

        self.upsample3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up3 = nn.Sequential(
            nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)

        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up2 = nn.Sequential(
            nn.Conv2d(256+128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)

        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )

        self.classifier = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
    # 128

    def forward(self, img):
        #128*128
        down1 = self.down1(img)
        down1_pool = self.down1_pool(down1)

        #64*64
        down2 = self.down2(down1_pool)
        down2_pool = self.down2_pool(down2)

        #32*32
        down3 = self.down3(down2_pool)
        down3_pool = self.down3_pool(down3)
        #16*16
        down4 = self.down4(down3_pool)
        down4_pool = self.down4_pool(down4)
        #8*8
        center = self.center(down4_pool)
        #8*8

        up4 = self.upsample4(center)
        #16*16

        up4 = torch.cat((down4,up4), 1)
        up4 = self.up4(up4)

        up3 = self.upsample3(up4)
        up3 = torch.cat((down3,up3), 1)
        up3 = self.up3(up3)

        up2 = self.upsample2(up3)
        up2 = torch.cat((down2,up2), 1)
        up2 = self.up2(up2)

        up1 = self.upsample1(up2)
        up1 = torch.cat((down1,up1), 1)
        up1 = self.up1(up1)

        prob = self.classifier(up1)


        # print(prob.size())
        return prob
