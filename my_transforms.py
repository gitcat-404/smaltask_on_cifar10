import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),#图像随机水平翻转，增加数据的多样性，以防止过拟合
    transforms.RandomGrayscale(),#以 50% 的概率将图像转换为灰度图像，增加数据的多样性
    transforms.ToTensor(),#将 PIL.Image 或 ndarray 转换为 tensor格式，并对数据进行归一化处理
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#对 tensor 进行标准归一化处理，使得图像的均值为 0，标准差为 1。具体实现是将每个像素值减去均值再除以标准差

transform_test = transforms.Compose([     
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])