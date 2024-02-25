'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 15:34:38
LastEditTime: 2024-02-19 16:08:52
FilePath: /Cat-Vs-Dog/main.py
Description: 训练模型主程序，包括模型的训练和验证，以及模型的保存和加载
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''


from re import S
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm, trange
import time
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np


from utils.dataset import Mydata
import Mymodels
from utils import filepath



def train(root,epoch_num,model_name,batch_size,lr_decay,resume,pre):

    """加载数据"""
    trainset=Mydata(root,train=True)
    validset=Mydata(root,train=False)
    train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=3)
    test_loader=DataLoader(validset,batch_size=batch_size,shuffle=False,num_workers=3)
    
    """加载模型和超参"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #是否有GPU
    
    """是否用预训练的模型"""
    if pre:  
        save_foldername=filepath.create_newfolder(model_name)    
        writer = SummaryWriter(save_foldername)   #定义一个writer，用来写入可视化相关的数据
        #迁移学习模型
        model = models.resnet50(pretrained = True)
        Use_gpu = torch.cuda.is_available()
        for parma in model.parameters():
            parma.requires_grad = False         #冻结预训练模型的权重，只训练最后一层的全连接的权重
        model.fc = torch.nn.Linear(2048,2)
        # print(model)
        if Use_gpu:
            model = model.cuda()

        #损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(),lr = 0.01)
        
        start_epoch=0
    else:
        """是否接着上次的训练"""
        if resume:	#断点续训并加载预训练权重
            
            save_foldername=filepath.find_lastfolder(model_name)  #返回上次训练的文件夹路径
            model_path=save_foldername+"/"+model_name+"_last"+".pth"
            checkpoint = torch.load(model_path)	#使用torch.load加载模型
            model=getattr(Mymodels,model_name)() #该函数属于反射操作，用于获取 models中名为 model_name 的属性(方法)
            model.load_state_dict(checkpoint['model'])	#加载权重
            optimizer = checkpoint['optimizer']	#加载优化器
            lr=checkpoint["lr"]
            #lr=0.05
            start_epoch = checkpoint['epoch']	#加载断点的epoch
            criterion = nn.CrossEntropyLoss().to(device)
            writer = SummaryWriter(save_foldername)
            
            print('加载epoch{}成功！'.format(start_epoch))      
        else:
            model = getattr(Mymodels,model_name)()  
            lr=0.00005        
            optimizer = optim.Adam(model.parameters(),lr)
            #optimizer = torch.optim.SGD(model.parameters(),lr)
            criterion = nn.CrossEntropyLoss().to(device)
            start_epoch=0
            #保存文件的路径
            save_foldername=filepath.create_newfolder(model_name)    
            writer = SummaryWriter(save_foldername)   #定义一个writer，用来写入可视化相关的数据
            print("创建新文件夹，从头训练…………")
        
    """开始训练"""
    print('start training...')
    best_accuracy=0
    previous_loss = 1e10
    for epoch in range(start_epoch,epoch_num):
        train_loss=0
        train_total=0
        correct=0
        total=0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   #梯度清零
            
            output = model(data)  #丢进模型中计算
            loss = criterion(output, target)   #计算误差
            train_total+=1
            train_loss+=loss.item()
            loss.backward()      #反向传播
            optimizer.step()     #更新参数
            _, predicted = torch.max(output.data, 1)   #最大值的坐标（0,1），也就对应类别
            total += target.size(0)
            correct += (predicted == target).sum()  #把预测正确的类别加起来
        
        
        """训练过程评估参数及其可视化"""
        Loss,accuracy=validation(model,test_loader,device,criterion)  #验证
        train_Loss=train_loss/train_total   #训练集误差
        train_accuracy=correct/total  #训练集准确率
        """Tensorboard可视化"""
        writer.add_scalar('LearnRate',lr, epoch)
        writer.add_scalar('TrainLoss', train_Loss, epoch)
        writer.add_scalar('val_Loss', Loss, epoch)
        writer.add_scalar('val_accuracy', accuracy, epoch)
        writer.add_scalar('train_accuracy',train_accuracy, epoch)
        print(f"epoch:{epoch+1}|{epoch_num},trn_loss:{train_Loss},val_loss:{Loss},val_acc:{accuracy}%,trn_acc:{train_accuracy}" )
        if accuracy>best_accuracy:  #保存最好的模型
            best_accuracy=accuracy
            torch.save(model.state_dict(),save_foldername+ "/"+model_name+"_best"+".pth")   #保存
    
        checkpoint = {'model':model.state_dict(),	 #将一些参数以字典形式保存进checkpoint
                        'optimizer':optimizer,
                        'epoch':epoch,
                        "lr":lr}
        
        torch.save(checkpoint,save_foldername+ "/"+model_name+"_last"+".pth")  #保存每个epoch的模型(覆盖)
        
        """如果验证损失不再下降，则降低学习率"""
        if  Loss>= previous_loss:
            lr = lr*lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        previous_loss=Loss
        print("当前学习率：",lr)
    writer.close()    
    
@torch.no_grad()
def validation(model,test_loader,device,criterion):
    """训练中的验证模块
    Args:
        model (_type_): 模型
        test_loader (_type_): 验证集
        device (_type_): 是否在gpu
        criterion (_type_): 优化准则
    Returns:
        _type_: 平均损失，准确率
    """
    model.eval() #将模型设置为验证模式
    total = 0
    correct = 0
    test_loss=0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)   #放进模型分类
        _, predicted = torch.max(outputs.data, 1)   #最大值的坐标（0,1），也就对应类别
        total += labels.size(0)
        correct += (predicted == labels).sum()  #把预测正确的类别加起来
        loss1 = criterion(outputs, labels)
        test_loss+=loss1.item()
    model.train() #模型恢复为训练模式
    
    accuracy=np.around((100*correct/total).numpy(),decimals=2)
    Loss=round(test_loss/total,6)
    
    return Loss,accuracy
        



if __name__=="__main__":
    
    root="./data/train"         #训练数据集路径
    epoch_num = 100             #迭代次数
    modelname="LeNet1"   #模型名字
    batchsize=30                #batchsize
    resume=False                #是否接着上次的继续训练
    lr_decay=0.9                #学习率下降率
    pre=False                   #是否采用预训练模型，采用预训练模型需要相应更改if pre后面的模型加载
    
    """开始训练"""
    start=time.time()
    train(root,epoch_num,modelname,batchsize,lr_decay,resume,pre)
    end=time.time()
    print(f"训练耗时：{round((end-start)/60,2)}分钟")
    
    