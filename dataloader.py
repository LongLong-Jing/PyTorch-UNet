import os
import collections
import json
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from torch.utils import data
from PIL import Image
import os
import os.path
import cv2
import PIL
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class ImageNet_Dataloader(data.Dataset):
    def __init__(self, root, split="train_aug", is_transform=False, img_size=224):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.rgb_transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = make_dataset(self.root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        #put all the RGB images into one folder, and mask into another folder
        rgb_path = self.files[index]
        tmp_path = rgb_path.split('/')
        #/media/tensor-server/ee577d95-535d-40b2-88db-5546defabb74/imagenet20_rgbmsk_v0/ImageNet20/aeroplane/n02691156/n02691156_4.JPEG
        #/media/tensor-server/ee577d95-535d-40b2-88db-5546defabb74/imagenet20_rgbmsk_v0/mask_output20/ImageNet20_refined_crabcut/aeroplane/n02691156_4.jpg

       # you can save the path of RGB image of its mask into a file
        mask_path = './data/masks/' + tmp_path[-1][:-4] + 'jpg'
        print(rgb_path)
        print(mask_path)
        rgb_img = Image.open(rgb_path).convert('RGB')
        mask = Image.open(mask_path)

        #resize each image into128*128
        rgb_img = rgb_img.resize((128,128), Image.BICUBIC)

        #must use NEARESTn method when resizing the mask
        mask = mask.resize((128,128), Image.NEAREST)

        mask = np.array(mask)
        mask = mask[:,:,0]
        mask[mask!=255] = 1
        mask[mask==255] = 0
        if self.is_transform:
            img,msk = self.transform(rgb_img,mask)
        return img, msk

    def transform(self, img, mask):
        img = self.rgb_transform(img)
        # msk = self.mask_transform(mask)
        msk = torch.FloatTensor(mask)
        return img,msk

if __name__ == '__main__':
    data_path = '/home/longlong/Documents/CVPR2019/ObjectDiscovery-data/Data/RGB_Image'
    dst = pascalVOCLoader(data_path, is_transform=True)
    print('length of the dataset', len(dst))
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        # print(i,imgs.shape)
        # print(np.amin(labels.numpy()),np.amax(labels.numpy()))
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
        # plt.imshow(dst.decode_segmap(labels.numpy()[0]))
        # plt.show()
