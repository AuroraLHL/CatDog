'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 10:58:00
LastEditTime: 2024-02-19 16:08:40
FilePath: /Cat-Vs-Dog/01_clean.py
Description: 多种方式判断图片是否损坏，并删除
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''


import os
from PIL import Image
import imghdr
from tqdm import tqdm


def is_valid_image(path):
    """多种方式判断图片是否损坏
    Args:path (string): 单张图片路径
    Returns:(bool): True or False
    """
    try:
        bValid = True
        fileObj = open(path, 'rb') # 以二进制形式打开
        buf = fileObj.read()
        
        # 方式一  是否以\xff\xd8开头
        if not buf.startswith(b'\xff\xd8'): 
            bValid = False
        # 方式二   # 是否以\xff\xd9结尾
        elif buf[6:10] in (b'JFIF', b'Exif'):  #“JFIF”的ASCII码
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'): 
                bValid = False
        #方式三  判断文件类型，如果是None，说明损坏
        elif imghdr.what(path) is None:
            bValid=False
        #方式四 用Image函数的verify()验证图片是否损坏
        else:
            try:
                Image.open(fileObj).verify()   #如果图片损坏会报错
            except Exception as e:
                bValid = False
                print(e)
    except Exception as e:
        return False
    return bValid


if __name__=="__main__":

    root="Kaggle_CatDog_dataset/PetImages"  #文件夹位置

    Classify=["Cat","Dog"]
    del_nums={}   #被删除图片数量
    normal_nums={}  #正常图片数量
    for _cls in Classify:
        file_dir=os.path.join(root,_cls)  #得到每个类别文件夹路径
        #遍历每个类别文件夹下图片
        for file in tqdm(os.listdir(file_dir)):
            filepath=os.path.join(file_dir,file)   #连接目录和文件名
            if is_valid_image(filepath) is False:
                #用字典储存每一类被删除的图片个数
                if del_nums.get(_cls,0)==0:
                    del_nums[_cls]=1
                else:
                    del_nums[_cls]+=1
                
                os.remove(filepath)  #删除损坏图片
            else:
                if normal_nums.get(_cls,0)==0:
                    normal_nums[_cls]=1
                else:
                    normal_nums[_cls]+=1
                
    for Cls,_ in del_nums.items():
        print(f"{Cls}类，一共删除了{del_nums[Cls]}张,还剩{normal_nums[Cls]}张")
