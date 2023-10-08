""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
from PIL import Image
import cv2

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        self.transform = transform
        self.imagelist = []
        self.namelist = []
        for parent, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    self.imagelist.append(os.path.join(parent, filename))
                    name = os.path.splitext(filename)[0]
                    self.namelist.append( name)


    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        label_str = 'tumor'
        if label_str == self.namelist[index][0:5]:
            label = 1
        else:
            label = 0

        img = Image.open(self.imagelist[index])

        if self.transform:
            img = self.transform(img)
        return label, img

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        self.transform = transform
        self.imagelist = []
        self.namelist = []
        for parent, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    self.imagelist.append(os.path.join(parent, filename))
                    name = os.path.splitext(filename)[0]
                    self.namelist.append( name)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        label_str = 'normal'
        if label_str == self.namelist[index][0:4]:
            label = 0
        else:
            label = 1
        img = Image.open(self.imagelist[index])

        if self.transform:
            img = self.transform(img)
        import sys
        import os
        #
        # # 将要输出保存的文件地址，若文件不存在，则会自动创建
        # # print("iteration: {}\ttotal {} iterations".format(i + 1, len(cifar100_test_loader)))
        # fw = open("/home/lll/Alice/small/tumor_classify/1.txt", 'a')
        # # 这里平时print("test")换成下面这行，就可以输出到文本中了
        # fw.write(str(label))
        # fw.write(str(self.namelist[index]))
        # # 换行
        # fw.write("\n")
        return label, img
