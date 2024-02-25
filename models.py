'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 15:33:09
LastEditTime: 2024-02-19 16:33:04
FilePath: /Cat-Vs-Dog/models.py
Description: 各种模型的定义文件,包括LeNet、AlexNet、ResNet34、SqueezeNet等模型的定义
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from torchvision.models import  squeezenet1_1
from torch.optim import Adam
from torchsummary import summary

from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,   #输入图片是三通道的
                out_channels=16,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),
        )
        #全连接层
        self.fc1 = nn.Linear(4* 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.size())
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #三个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #三个全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 64),
            nn.ReLU(),
            nn.Dropout()
        )
        #self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 2)   #分类类别为2，

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

class AlexNet(nn.Module):
    def __init__(self,classes=2):
        super(AlexNet,self).__init__()
        #第一个块,这里的padding=3 是由于 数据集使用的是224*224的图像，padding=3 
        self.zeropadding2d=nn.ZeroPad2d(padding=(3,0,3,0))
        self.conv1=nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(11,11),stride=4) 
        self.bn1=nn.BatchNorm2d(96)  
        self.relu1 = nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=(3,3),stride=2)
        
        #第二个块
        self.conv2=nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),padding='same',stride=1)
        self.bn2=nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.maxpool2=nn.MaxPool2d(kernel_size=(3,3),stride=2)
        
        #第三个快
        self.conv3=nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(5,5),padding='same',stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3=nn.MaxPool2d(kernel_size=(3,3),stride=2)
        #第四~五块
        self.conv4=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding='same',stride=1)
        self.conv5=nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding='same',stride=1)
        self.relu4 = nn.ReLU()
        self.maxpool4=nn.MaxPool2d(kernel_size=(3,3),stride=2)
        #self.globalmax=nn.AdaptiveAvgPool2d()
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=1024,out_features=4096)
        self.linear_relu1 = nn.ReLU()
        self.linear2=nn.Linear(in_features=4096,out_features=4096)
        self.linear_relu2 = nn.ReLU()
        self.linear3=nn.Linear(in_features=4096,out_features=classes)
        self.softmax=nn.Softmax(dim=1)
    #定义计算过程
    def forward(self,x):
        x=self.zeropadding2d(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.maxpool2(x)

        x=self.conv3(x)
        x=self.relu3(x)
        x=self.maxpool3(x)

        x=self.conv4(x)
        x=self.conv5(x)
        x=self.relu4(x)
        x=self.maxpool4(x)
        x=self.flatten(x)

        #linear connect
        x=self.linear1(x)
        x=self.linear_relu1(x)
        x=self.linear2(x)
        x=self.linear_relu2(x)
        x=self.linear3(x)
        #self.softmax(x)
        return x
    
class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet34(nn.Module):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """

    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

        #self.softmax = nn.Softmax()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return self.fc(x)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model_name = 'squeezenet'
        self.model = squeezenet1_1(pretrained=True)
        # 修改 原始的num_class: 预训练模型是1000分类
        self.model.num_classes = num_classes
        self.model.classifier =   nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self,x):
        return self.model(x)


if __name__=="__main__":
    net=AlexNet()
    
    print(summary(net,(3,224,224)))