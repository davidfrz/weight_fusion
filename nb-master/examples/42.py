import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out		# 这里并没有使用到论文中的shared MLP, 而是直接相加了
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEModule(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, downsample=None):
        super(SEModule, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out

        # out += residual

        #         print(out.size())
        out = self.sa(out) * out
        #         print(out.size())
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #         print(out.size())
        out = self.relu(out)

        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=None, fpn_levels=None):
        super(MobileNetV3_Large, self).__init__()
        self.with_classifier = num_classes != None
        self.fpn_levels = fpn_levels

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),  # 14
        )

        self.l0_fusion = ASFF(level=1)
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, 5)
        self.init_params()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se2=SeModule(160)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # todo: replace this with Focus module
        x = self.hs1(self.bn1(self.conv1(x)))

        fpn_outputs = []
        for i, l in enumerate(self.bneck):
            x = l(x)
            if self.fpn_levels:
                if i in self.fpn_levels:
                    fpn_outputs.append(x)


        fpn_outputs.insert(0, x)
        # print(fpn_outputs[1].shape)
        out = self.l0_fusion(fpn_outputs[3], fpn_outputs[2], fpn_outputs[1])
        # out=self.se2(out)
        # print(out.size())
        # return fpn_outputs
        out = self.hs2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 7)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.linear3(out))
        out = self.linear4(out)
        return out


model = MobileNetV3_Large(fpn_levels=[5, 9, 14])


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
MAX_EPOCH = 120
BATCH_SIZE = 56
LR = 0.001
log_interval = 30
val_interval = 1

train_dir = r'/home/fanrz/Desktop/testlarge/aptos/tra2/'
# 对训练集所需要做的预处理
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
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
        #     print("============================================")
        #     print("源数据标签：",label)
        #     print("============================================")
        #     print("预测结果：",predicted)
        #     print("相等的结果为：",predicted == label)
        correct += (predicted == label).sum()
        if (i + 1) % log_interval == 0:
            print(correct.item() / total)
    #         print(correct.item())
    print("============================================")
    accurancy = correct.item() / total
    if accurancy > accurancy_global:
        torch.save(model.state_dict(), './weights/best45sp.pkl')
        print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
        accurancy_global = accurancy
    print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, 100 * accurancy))
# torch.save(model.state_dict(), './weights/last组合无注意力.pkl')
print("训练完毕，权重已保存为：weights/last.pkl")
