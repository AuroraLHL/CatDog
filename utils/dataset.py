'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 13:15:17
LastEditTime: 2024-02-19 15:30:36
FilePath: /Cat vs Dog/utils/dataset.py
Description: 数据集加载
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''

from PIL import Image
from torchvision import transforms
from torch.utils import data
import os
import matplotlib.pyplot as plt
import numpy as np

class Mydata(data.Dataset):
    """定义自己的数据集"""
    def __init__(self, root, Transforms=None, train=True):
        """
        Args:
            root:训练集的路径
            Transforms:图片处理的方式
            train:是否是训练数据。True:训练数据;False:验证数据
        """
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        #按序号排序，以便训练集和验证集不会混乱
        imgs=sorted(imgs,key=lambda x:int((x.split("/")[-1]).split("_")[0]))   
        imgs_num=len(imgs)
        """进行数据集的划分"""
        if train:
            self.imgs = imgs[:int(0.8*imgs_num)]  #80%训练集
        else:
            self.imgs = imgs[int(0.8*imgs_num):]  #20%验证集
        if Transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
            # 图片处理
            color_aug = transforms.ColorJitter(brightness=0.1)
            self.transforms = transforms.Compose(
                    [ transforms.CenterCrop([224,224]), 
                    transforms.Resize([224,224]),
                    color_aug,
                    transforms.RandomHorizontalFlip(p=0.5),    #数据增强
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1),   #亮度变化
                    transforms.ToTensor(), normalize])
            
    def __getitem__(self, index):
        """
		返回一张图片的数据
		训练集和验证集,则对应的是dog返回1,猫则返回0
		"""
        img_path = self.imgs[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0   #获得标签
        #print(img_path)
        data = Image.open(img_path).convert("RGB")  #可能有单通道的，注意转换为三通道图片
        try:
            data = self.transforms(data)   #图片处理
        except:
            print(img_path)
            raise ValueError("图片打开失败")
        return data, label
    def __len__(self):
        """返回数据集中所有图片的个数"""    
        return len(self.imgs)
    def getall(self):
        return self.imgs

if __name__ == "__main__":
    root = "./data/train"
    train = Mydata(root, train=True)
    img,label=train.__getitem__(5)
    imgs=train.getall()
    img=img.numpy()
    print(type(img))
    print(img.shape,label)
    print(len(train))
    plt.imshow(np.transpose(img, (1,2,0)))