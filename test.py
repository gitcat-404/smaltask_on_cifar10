import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from config import cfg
from my_transforms import transform_test


test_data =datasets.CIFAR10(root=cfg.ROOT,train=False,transform=transform_test,download=False)
test_loader = DataLoader(dataset=test_data,batch_size=cfg.BATCH_SIZE,shuffle=True,num_workers=2)
#模型加载
model = torch.load(cfg.TEST_MODEL)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#测试
model.eval()
#model.train()


if __name__ == '__main__':
    correct,total = 0,0
    for j,data in enumerate(test_loader):
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        #前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data,1)
        total =total+labels.size(0)
        correct = correct +(predicted == labels).sum().item()
        
    print('准确率：{:.4f}%'.format(100.0*correct/total))