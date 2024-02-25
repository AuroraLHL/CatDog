'''
Author: Aurora 2074375758@qq.com
Date: 2022-04-20 19:06:47
LastEditTime: 2024-02-20 09:41:45
FilePath: /Cat-Vs-Dog/predict.py
Description: 利用训练好的模型进行预测
Copyright (c) 2024 by Aurora, All Rights Reserved. 
'''

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models import LeNet1

def predict(root,imgname, model, img_trans):
    imgpath = os.path.join(root, imgname)
    img_rgb = Image.open(imgpath).convert("RGB")
    img_tensor = img_trans(img_rgb).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_prob, pred_idx = torch.max(prob, dim=1)
        pred_prob = pred_prob.item()
        pred_idx = pred_idx.item()

    if pred_idx == 0:
        pred_str = "cat"
    else:
        pred_str = "dog"

    plt.imshow(img_rgb)
    plt.title("Predicted: {} ,Probability: {:.2f}".format(pred_str, pred_prob))
    plt.savefig("output/pre_" + imgname)

if __name__ == "__main__":
    
    # 图像预处理
    img_trans = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = LeNet1() # 模型结构
    modelpath = "./runs/LeNet1_1/LeNet1_best.pth" # 训练好的模型路径
    checkpoint = torch.load(modelpath)  
    model.load_state_dict(checkpoint)  # 加载模型参数
    
    root = "test_pics"
    for pic in os.listdir(root):
        if pic.endswith(".jpg"):
            predict(root,pic, model, img_trans)
