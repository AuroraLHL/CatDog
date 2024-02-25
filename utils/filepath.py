'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-21 20:26:20
LastEditTime: 2024-02-19 15:30:18
FilePath: /Cat vs Dog/utils/filepath.py
Description: 文件夹路径的处理相关函数
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''

import os

def find_lastfolder(model_name):
    """返回最后一次训练的文件夹"""
    path="./runs"
    
    all=os.listdir(path)
    folders=[]
    for folder in all:
        if model_name in folder:
            folders.append(folder)
    if len(folders)==0:
        return path+"/"+model_name+"_"+str(1)
    folders=sorted(folders,key=lambda x:x.split("_")[-1])
    num=int(folders[-1].split("_")[-1])
    return path+"/"+model_name+"_"+str(num)

def create_newfolder(model_name):
    """创建新文件夹"""
    path="./runs"
    
    all=os.listdir(path)
    folders=[]
    for folder in all:
        if model_name in folder:
            folders.append(folder)
    if len(folders)==0:
        return path+"/"+model_name+"_"+str(1)
    folders=sorted(folders,key=lambda x:x.split("_")[-1])
    num=int(folders[-1].split("_")[-1])+1
    return path+"/"+model_name+"_"+str(num)
    

if __name__=="__main__":
    a=find_lastfolder("LeNet")
    print(a)