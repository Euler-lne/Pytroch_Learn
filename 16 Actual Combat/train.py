import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from model import Euler
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(
    "./dataset",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(
    "./dataset",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
# print(train_data_size)
# print(test_data_size)

# 利用dataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络某型
euler = Euler()

#损失函数
loss_fun = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(euler.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
#记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
#记录训练轮数
epoch = 30

for i in range(epoch):
    # 训练开始
    euler.train()
    for data in train_dataloader:
        imgs, targets = data
        output = euler(imgs)
        loss = loss_fun(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))

    # 评估步骤开始
    euler.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = euler(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    total_test_step = total_test_step + 1
