import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import os
import io
import re
import datetime
import cv2
import sys
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .vgg import *
from .resnetv2 import *
import torch.nn as nn
from .dataset import CIFAR100Test
# 导入迁移学习的工具
from torchvision import models
import torch.optim as optim

def read_images(df,imgs_path):
    images = []
    for img_name in tqdm(df.image_names.values):
        # defining the image path
        img_path = f'{imgs_path}/{img_name}'
        # reading the image
        img = cv2.imread(img_path)
        # appending the image into the list
        images.append(img)
    return images
# 定义模型
class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'xception', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def data_split(images, df):
    x = images
    y = df.file_value.values
    train_x, val_x, train_y, val_y = train_test_split(x,
                                                      y,
                                                      test_size=0.1,#测试集站10%
                                                      random_state= 13,
                                                      stratify=y)
    return train_x, val_x, train_y, val_y

def data_convert(train_x, train_y, val_x, val_y, data_transforms):
    # defining train dataset, test dataset
    train_dataset, val_dataset = [], []
    # obtain train dataset
    for idx, data in enumerate(train_x):
        data = data_transforms['train'](data)
#         data = data.permute(2,0,1)
        label = torch.tensor(train_y[idx], dtype=torch.long)
        train_dataset.append((data, label))
    # obtain val dataset
    for idx, data in enumerate(val_x):
        data = data_transforms['val'](data)
#         data = data.permute(2,0,1)
        label = torch.tensor(val_y[idx], dtype=torch.long)
        val_dataset.append((data, label))
    # 返回dataset格式
    return train_dataset, val_dataset
def get_network(args):
    """ return given network
    """

    if args.net == 'efficientNet':
        from efficientnet_pytorch import EfficientNet
        # 加载预训练mb3网络
        model = EfficientNet.from_name('efficientnet-b3')
        # 冻结参数以进行微调
        for param in model.parameters():
            param.requires_grad = False
        # 更改头部层以适应二类问题
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 2)
        net = model
    elif args.net == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze the parameters of the pre-trained model
        for param in model.parameters():
            param.requires_grad = False
        # Replace the classifier of the pre-trained model with a new one
        model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2))
        net = model
    elif args.net == 'resnetv2':
            from .resnetv2 import PreActResNet18
            net = PreActResNet18()
    elif args.net == 'googlenet':
        model = models.googlenet(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        net = model
    elif args.net == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier[-1] = nn.Linear(4096, 2, bias=True)
        for param in vgg16.features.parameters():
            param.requires_grad = False
        net = vgg16
    elif args.net == 'alexnet':
        alexnet_model = models.alexnet(pretrained=True)
        # Replace the last layer for binary classification
        num_features = alexnet_model.classifier[-1].in_features
        alexnet_model.classifier[-1] = nn.Linear(num_features, 2)
        net = alexnet_model
    elif args.net == 'resnet50':
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)
        net = resnet
    elif args.net == 'ResNet50ASPP':
        net = resnet50aspp
    elif args.net == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        net =  model
    elif args.net == 'mobilenetv2':
        # 加载预训练的MobileNet模型
        model = models.mobilenet_v2(pretrained=True)
        # 将最后一层替换成二分类的全连接层
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2),
        )
        net = model
    elif args.net == 'shufflenet':
        # 加载预训练模型
        model = models.shufflenet_v2_x1_0(pretrained=True)

        # 固定模型参数
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(1024, 2)
        net = model
    elif args.net == 'Xception':
        # 加载预训练模型
        model = models.xception(pretrained=True)
        # 固定模型参数
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, 2)
        net = model


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()



    return net
def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def data_convert_test(val_x, val_y, data_transforms):
    #  test dataset
    val_dataset = []

    # obtain val dataset
    for idx, data in enumerate(val_x):
        data = data_transforms['val'](data)
#         data = data.permute(2,0,1)
        label = torch.tensor(val_y[idx], dtype=torch.long)
        val_dataset.append((data, label))
    # 返回dataset格式
    return val_dataset

def data_split_test(images, df):
    x = images
    y = df.file_value.values
    val_x = x
    val_y = y
    return  val_x,  val_y

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[0], padding=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[1], padding=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[2], padding=rates[2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.conv1x1_out(out)
        return out


class ResNet50ASPP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True).children())[:-2])
        self.aspp = ASPP(2048)
        self.fc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.aspp(x)
        x = self.fc(x)
        x = F.interpolate(x, size=x.shape[2:]*2, mode='bilinear', align_corners=True)
        return x


def resnet50aspp():
    model = ResNet50ASPP()
    return model


