import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.asff_blocks import ASFFmbv6 as ASFF
import math
import os
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=5, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

model = ResNet50()


# model=models.resnet50(pretrained=False)
# numFit = model.fc.in_features
# model.fc = nn.Linear(numFit, 5)

random.seed(1)


class flowerDataset(Dataset):
    # 自定义Dataset类，必须继承Dataset并重写__init__和__getitem__函数
    def __init__(self, data_dir, transform=None):
        """
            花朵分类任务的Dataset
            :param data_dir: str, 数据集所在路径
            :param transform: torch.transform，数据预处理，默认不进行预处理
        """
        # data_info存储所有图片路径和标签（元组的列表），在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # 打开图片，默认为PIL，需要转成RGB
        img = Image.open(path_img).convert('RGB')
        # 如果预处理的条件不为空，应该进行预处理操作
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    # 自定义方法，用于返回所有图片的路径以及标签
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                # listdir为列出文件夹下所有文件和文件夹名
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 过滤出所有后缀名为jpg的文件名（那当然也就把文件夹过滤掉了）
                # img_names = list(filter(lambda x: x.endswith('.jpeg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 在该任务中，文件夹名等于标签名
                    label = sub_dir
                    data_info.append((path_img, int(label)))
        return data_info

# 宏定义一些数据，如epoch数，batchsize等
MAX_EPOCH = 80
BATCH_SIZE = 64
LR = 0.001
log_interval = 30
val_interval = 1

train_dir = r'/home/fanrz/Desktop/testlarge/aptos/tra2/'
# 对训练集所需要做的预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 构建MyDataset实例
train_data = flowerDataset(data_dir=train_dir, transform=train_transform)

# 构建DataLoader
# 训练集数据最好打乱
# DataLoader的实质就是把数据集加上一个索引号，再返回
train_loader=DataLoader(dataset=train_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True)

# ============================ step 2/5 模型 ============================
if torch.cuda.is_available():
    model=nn.DataParallel(model)
    model.cuda()

# ============================ step 3/5 损失函数 ============================
criterion=nn.CrossEntropyLoss()
# ============================ step 4/5 优化器 ============================
optimizer=optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.99))# 选择优化器

model.train()
accurancy_global = 0.0
for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        img, label = data
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        #         print(img.size())
        # 前向传播
        out = model(img)
        # print(out.shape)
        optimizer.zero_grad()  # 归0梯度
        loss = criterion(out, label)  # 得到损失函数

        print_loss = loss.data.item()

        loss.backward()  # 反向传播
        optimizer.step()  # 优化
        if (i + 1) % log_interval == 0:
            print('epoch:{},loss:{:.4f}'.format(epoch + 1, loss.data.item()))
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        # if (i + 1)%log_interval==0:
            # print("============================================")
            # print("源数据标签：",label)
            # print("============================================")
            # print("预测结果：",predicted)
            # print("相等的结果为：",predicted == label)
        correct += (predicted == label).sum()
        if (i + 1) % log_interval == 0:
            print(correct.item() / total)
    #         print(correct.item())
    print("============================================")
    accurancy = correct.item() / total
    if accurancy > accurancy_global:
        torch.save(model.state_dict(), './weights/resnet50.pkl')
        print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
        accurancy_global = accurancy
    print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, 100 * accurancy))
# torch.save(model.state_dict(), './weights/last组合无注意力.pkl')
print("训练完毕，权重已保存为：weights/last.pkl")