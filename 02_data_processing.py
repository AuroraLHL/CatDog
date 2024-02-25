'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 14:07:33
LastEditTime: 2024-02-19 16:08:46
FilePath: /Cat-Vs-Dog/02_data_processing.py
Description: 从原始数据中取出训练数据和验证数据
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''


import os
import random
from tqdm import tqdm
import shutil


if __name__=="__main__":

    root="./data"   #数据文件根目录
    sourcePath="./Kaggle_CatDog_dataset/PetImages"  #原始数据根目录
    train_num=4000  #训练集数目
    classes=["Cat","Dog"]

    for folder in os.listdir(root):   #一开始先删除里面原有的照片
        folderpath=os.path.join(root,folder)
        if os.path.isdir(folderpath):
            for pic in os.listdir(folderpath):
                if pic.endswith(".jpg"):
                    file=os.path.join(folderpath,pic)
                    os.remove(file)
        print("删除成功")

    for cls in classes:
        folder_path=os.path.join(sourcePath,cls)
        pictures=os.listdir(folder_path)
        random.shuffle(pictures)  #打乱顺序
        for index in tqdm(range(int(train_num/len(classes)))):
            #复制到训练集 
            oldpath=os.path.join(folder_path,pictures[index])
            newpath=os.path.join(root,"train",str(index)+"_"+cls.lower()+".jpg")
            shutil.copy(oldpath,newpath)
        