import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pylab as plt
import numpy as np
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.asff_blocks import ASFFmbv7 as ASFF
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
        out = avg_out + max_out		# ????????????shared MLP, ???????
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


def Confusion_mxtrix(labels, predicted, num_classes):
    """
    \u6df7\u6dc6\u77e9\u9635\u5b9a\u4e49
    Args:
        labels: [number_total_pictures,1]
        predicted: [number_total_pictures,1]
        num_classes: \u5206\u7c7b\u6570\u76ee

    Returns: Confusion_matrix
    """
    Cmatrixs = torch.zeros((num_classes, num_classes))
    stacked = torch.stack((labels, predicted), dim=1)
    for s in stacked:
        a, b = s.tolist()
        Cmatrixs[a, b] = Cmatrixs[a, b] + 1
    return Cmatrixs


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ('NO DR', 'Mild', 'Moderate', 'Severe', 'Proliferative')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # \u5728\u6df7\u6dc6\u77e9\u9635\u4e2d\u6bcf\u683c\u7684\u6982\u7387\u503c
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


def Evaluate(Cmatrixs):
    classes = ('NO DR', 'Mild', 'Moderate', 'Severe', 'Proliferative')
    n_classes = Cmatrixs.size(0)
    Prec, Rec = torch.zeros(n_classes + 1), torch.zeros(n_classes + 1)
    Acc=torch.zeros(n_classes + 1)
    sum_cmt_row = torch.sum(Cmatrixs, dim=1)  # \u884c\u7684\u548c
    sum_cmt_col = torch.sum(Cmatrixs, dim=0)  # \u5217\u7684\u548c
    print("----------------------------------------")
    for i in range(n_classes):
        TP = Cmatrixs[i, i]
        FN = sum_cmt_row[i] - TP
        FP = sum_cmt_col[i] - TP
        TN = torch.sum(Cmatrixs) - sum_cmt_row[i] - FP
        Prec[i] = TP / (TP + FP)
        Rec[i] = TP / (TP + FN)
        Acc[i]=(TP+TN)/(TP+FN+FP+TN)
        print("%s" % (classes[i]).ljust(10, " "), "Presion=%.3f%%,     Recall=%.3f%%    ,Accuracy=%.3f%%" % (Prec[i], Rec[i],Acc[i]))

    Prec[-1] = torch.mean(Prec[0:-1])
    Rec[-1] = torch.mean(Rec[0:-1])
    print("ALL".ljust(10, " "), "Presion=%.3f%%,     Recall=%.3f%%" % (Prec[i], Rec[i]))
    print("----------------------------------------")


#         return Prec,Rec
def MyROC_i(outputs, labels, n=20):
    '''
    ROC\u66f2\u7ebf\u8ba1\u7b97 \u7ed8\u5236\u6bcf\u4e00\u7c7b\u7684
    Args:
        outputs: [num_labels,num_classes]
        labels: \u6807\u7b7e\u503c
        n: \u5f97\u5230n\u4e2a\u70b9\u4e4b\u540e\u7ed8\u56fe
    Returns:plot_roc
    '''

    n_total, n_classes = outputs.size()
    labels = labels.reshape(-1, 1)  # \u884c\u5411\u91cf\u8f6c\u4e3a\u5217\u5411\u91cf
    T = torch.linspace(0, 1, n)
    TPR, FPR = torch.zeros(n, n_classes + 1), torch.zeros(n, n_classes + 1)

    for i in range(n_classes):
        for j in range(n):
            mask_1 = outputs[:, i].cpu() > T[j]
            TP_FP = torch.sum(mask_1)
            mask_2 = (labels[:, -1].cpu() == i)
            TP = torch.sum(mask_1 & mask_2)
            FN = n_total / n_classes - TP
            FP = TP_FP - TP
            TN = n_total - n_total / n_classes - FP

            TPR[j, i] = TP / (TP + FN)
            FPR[j, i] = FP / (FP + TN)

    TPR[:, -1] = torch.mean(TPR[:, 0:-1], dim=1)
    FPR[:, -1] = torch.mean(FPR[:, 0:-1], dim=1)

    return TPR, FPR


def Plot_ROC_i(TPR, FPR):
    for i in range(5 + 1):
        if i == 5:
            width = 2
        else:
            width = 1
        plt.plot(FPR[:, i],
                 TPR[:, i],
                 linewidth=width,
                 label='classes_%d' % (i))

    plt.legend()
    plt.title("ROC")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(r'./_ROC_i.png')

class flowerDataset(Dataset):
    # ???Dataset??????Dataset???__init__?__getitem__??
    def __init__(self, data_dir, transform=None):
        """
            ???????Dataset
            :param data_dir: str, ???????
            :param transform: torch.transform???????????????
        """
        # data_info????????????????????DataLoader???index????
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # ????????PIL?????RGB
        img = Image.open(path_img).convert('RGB')
        # ?????????????????????
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    # ?????????????????????
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # ????
            for sub_dir in dirs:
                # listdir????????????????
                img_names = os.listdir(os.path.join(root, sub_dir))
                # ?????????jpg???????????????????
                # img_names = list(filter(lambda x: x.endswith('.jpeg'), img_names))

                # ????
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # ???????????????
                    label = sub_dir
                    data_info.append((path_img, int(label)))
        return data_info

labels_value, predicted_value, outputs_value = [],[],[]
correct = 0
total = 0

net = MobileNetV3_Large(fpn_levels=[5, 9, 14])
# net=nn.DataParallel(net)
net.eval()
net.cuda()

net.load_state_dict(torch.load(r'./weights/best45sp.pkl'))

valid_dir=r'/home/frz/Desktop/val2/'
valid_transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])
valid_data=flowerDataset(data_dir=valid_dir,transform=valid_transform)
valid_loader=DataLoader(dataset=valid_data,batch_size=16,drop_last=True)

for i,data in enumerate(valid_loader):
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(images)
    _,predicted = torch.max(outputs.data,1) # tensor([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3], device='cuda:0')
    total=total+labels.size(0)
    correct=correct+(predicted==labels).sum()
#     print(correct) #tensor(14, device='cuda:0')
    if i==0:
        labels_value = labels
        predicted_value = predicted
        outputs_value = F.softmax(outputs.data,dim=1)
    else:
        labels_value = torch.cat((labels_value,labels),0)
        predicted_value = torch.cat((predicted_value,predicted),0)
        outputs_value = torch.cat((outputs_value,F.softmax(outputs.data,dim=1)),0)
correct = correct.cpu().numpy()

print("acc=%.5f%%" % (100 * correct / total))
Cmatrixs = Confusion_mxtrix(labels_value,predicted_value,5)
print(Cmatrixs)

Evaluate(Cmatrixs)
TPR_i, FPR_i = MyROC_i(outputs_value, labels_value)
Plot_ROC_i(TPR_i, FPR_i)

def kappa(matrix):
    matrix=np.array(matrix)
    n=np.sum(matrix)
    sum_po=0
    sum_pe=0
    for i in range(len(matrix[0])):
        sum_po+=matrix[i][i]
        row=np.sum(matrix[i,:])
        col=np.sum(matrix[:,i])
        sum_pe+=row*col
    po=sum_po/n
    pe=sum_pe/(n*n)
    return (po-pe)/(1-pe)

print(kappa(Cmatrixs))
