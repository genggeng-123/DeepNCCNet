import torch
import torch.nn as nn
import torchvision.models as models

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(CoordinateAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels+2, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, height, width = x.size()
        xx_channel = torch.arange(width).repeat(height, 1)
        yy_channel = torch.arange(height).repeat(width, 1).t()
        xx_channel = xx_channel.view(1, 1, height, width).repeat(batch_size, 1, 1, 1).type_as(x)
        yy_channel = yy_channel.view(1, 1, height, width).repeat(batch_size, 1, 1, 1).type_as(x)
        x = torch.cat([x, xx_channel, yy_channel], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class VGGWithAttention(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(VGGWithAttention, self).__init__()
        self.vgg = models.vgg16(pretrained=use_pretrained)
        self.coord_atten = CoordinateAttention(in_channels=512)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        features = self.vgg.features(x)
        atten = self.coord_atten(features)
        features = features * atten
        features = features.mean(dim=[2, 3])
        x = self.fc(features)
        return x
