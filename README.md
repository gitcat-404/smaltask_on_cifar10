# smaltask_on_cifar10
这是一个基于pytorch在cifar10上的简单demo，用来存储我的计算机视觉课程报告代码
# 快速开始
首先，需要我们将cifar10官网下载的cifar-10-batches-py文件放到一个固定的文件夹中
将该项目克隆到本地：
```
git clone https://github.com/gitcat-404/smaltask_on_cifar10
```
1. 在本地打开config.py文件，修改其中参数
2. 将ROOT参数修改为自本地存储cifar10数据集的地址，这里只需传入cifar-10-batches-py数据集（官网下载）所在的根路径即可，因为pytorch的datasets.CIFAR10函数会自动帮我们读取
3. 设置超参数，如EPOCH，LR,batch_size等
4. 设置需要使用的网络，这里提供了alexnet，vgg，以及resnet全套模型供使用，其中resnet类模型可以直接运行，而由于alexnet与vgg16原本是解决IMAGENet上1000分类问题的模型，因此输出层需要自行设置全连接层进行调整，由于时间原因并未实现
# 我的结果
| 网络        | EPOCH   |  batchsize  |误差率 |
| --------   | -----:   | :----: |:----: |
| Resnet18     | 10    |   64   | 21.39 |
| Resnet34    | 10    |   64   | 21.97 |
| Resnet50     | 10    |   64   | 33.18 |
| Resnet101    | 10    |   16   | 31.02 |
| Resnet152     | 10    |   16   | 40.55 |
