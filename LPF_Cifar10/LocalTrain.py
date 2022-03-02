# !/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from misc import progress_bar


# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
def LocalTrain(GlobalModel, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_net = copy.deepcopy(GlobalModel).to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    f = open('./data/users/' + str(num + 1) + '_user.pkl', 'rb')
    images_i, labels_i = pickle.load(f)
    f.close()
    images_i, labels_i = images_i.to(device), labels_i.to(device)
    data = Data.TensorDataset(images_i, labels_i)
    trainloader = DataLoader(data, batch_size=128, shuffle=True)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              transform=transform)
    batch_size = 32
    testloader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

    # train
    for epoch in range(5):
        for batch_num, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 构造测试的dataloader
    dataiter = iter(testloader)
    # 预测正确的数量和总数量
    correct = 0
    total = 0
    # 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
    # cv2.namedWindow('predictPic', cv2.WINDOW_NORMAL)
    to_pil_image = transforms.ToPILImage()
    with torch.no_grad():
        for images, labels in dataiter:
            # images, labels = data
            # print(images)
            # print(len(images.data))
            images, labels = images.to(device), labels.to(device)
            # 预测
            # outputs = self.net(images)
            outputs = model_net(images)
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(images.data[0])
            # print(len(images.data[0]))
            # input_flag = input()
            # if input_flag == 'p':
            #     break
            # elif input_flag == 'c':
            #     continue
            # cv2.imshow('predictPic', images)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    numpy_para = {}
    i = 0
    parameters = model_net.parameters()
    for p in parameters:
        numpy_para.setdefault(i, p.detach().cpu().numpy())
        i = i + 1
    return numpy_para, 100 * correct / total
