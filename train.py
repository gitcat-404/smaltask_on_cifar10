import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from my_transforms import transform_train
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from config import cfg
import models.resnet
import models.alexnet
import models.vgg


torch.cuda.empty_cache()
#超参数定义
EPOCH = cfg.EPOCH
BATCH_SIZE = cfg.BATCH_SIZE
LR = cfg.LR
#这里只需传入cifar-10-batches-py数据集（官网下载）所在的根路径即可
root=cfg.ROOT
#设置所需使用的模型
net = cfg.NET
if net == 'Res50':
    model = models.resnet.resnet50(pretrained=False)
    #model = torchvision.models.resnet50(pretrained=False)
elif net == 'VGG':
    model = models.vgg.vgg16(pretrained=False)
elif net == 'AlexNet':
    model = models.alexnet.alexnet(pretrained=False)



train_data = datasets.CIFAR10(root=root, train=True,transform=transform_train,download=False)

#使用DataLoader进行数据分批
train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
criterion = nn.CrossEntropyLoss()
#选择优化器,这里选用Adam来做优化器，还可以选其他的优化器
optimizer = optim.Adam(model.parameters(),lr=LR)
#设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#模型和输入数据都需要to device
model = model.to(device)
model.train()
#开始训练！
if __name__ == '__main__':
    torch.cuda.empty_cache()
    for epoch in range(EPOCH):
        for i,data in enumerate(train_loader):
            #取出数据及标签
            inputs,labels = data
            #数据及标签均送入GPU或CPU
            inputs,labels = inputs.to(device),labels.to(device)
            
            #前向传播
            outputs = model(inputs)
            #计算损失函数
            loss = criterion(outputs,labels)
            #清空上一轮的梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()

            #利用tensorboard，将训练数据可视化
            if  i%100 == 0:
                #writer.add_scalar("Train/Loss", loss.item(), epoch*len(train_loader)+i)
                print('it’s training...{}'.format(i))
        print('epoch{} loss:{:.4f}'.format(epoch+1,loss.item()))

    #保存模型参数
    saved_root=cfg.NET+'.pt'
    torch.save(model,saved_root)
    print('model has been saved')

