import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pylab as plt
import numpy as np
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.asff_blocks import ASFFmbv4 as ASFF
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from PIL import Image
from torch.utils.data import Dataset
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

from itertools import cycle
from sklearn.metrics import roc_curve,auc, f1_score, precision_recall_curve, average_precision_score



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

net = ResNet50()

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
                # img_names = list(filter(lambda x: x.endswith('.png'), img_names))

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

# net = MobileNetV3_Large(fpn_levels=[5, 9, 14])
net=nn.DataParallel(net)
net.eval()
net.cuda()

net.load_state_dict(torch.load(r'./weights/resnet50.pkl'))

valid_dir=r'/home/frz/Desktop/val2/'
valid_transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])
valid_data=flowerDataset(data_dir=valid_dir,transform=valid_transform)
valid_loader=DataLoader(dataset=valid_data,batch_size=16,drop_last=True)
score_list = []
label_list = []
for i,data in enumerate(valid_loader):
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(images)
    _,predicted = torch.max(outputs.data,1) # tensor([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3], device='cuda:0')
    total=total+labels.size(0)
    correct=correct+(predicted==labels).sum()
#     print(correct) #tensor(14, device='cuda:0')
    score_tmp=outputs
    score_list.extend(score_tmp.detach().cpu().numpy())
    label_list.extend(labels.cpu().numpy())

    if i==0:
        labels_value = labels
        predicted_value = predicted
        outputs_value = F.softmax(outputs.data,dim=1)
    else:
        labels_value = torch.cat((labels_value,labels),0)
        predicted_value = torch.cat((predicted_value,predicted),0)
        outputs_value = torch.cat((outputs_value,F.softmax(outputs.data,dim=1)),0)

score_array=np.array(score_list)
correct = correct.cpu().numpy()
label_tensor = torch.tensor(label_list)
label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
label_onehot = torch.zeros(label_tensor.shape[0], 5)
label_onehot.scatter_(dim=1, index=label_tensor, value=1)
label_onehot = np.array(label_onehot)
fpr_dict = dict()
tpr_dict = dict()
roc_auc_dict = dict()
for i in range(5):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
# micro
fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

# macro
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(5)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
# Finally average it and compute AUC
mean_tpr /= 5
fpr_dict["macro"] = all_fpr
tpr_dict["macro"] = mean_tpr
roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
plt.figure()
lw = 2
plt.plot(fpr_dict["micro"], tpr_dict["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_dict["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr_dict["macro"], tpr_dict["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_dict["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(5), colors):
    plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_dict[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.savefig('set113_roc.jpg')
plt.show()




print("acc=%.5f%%" % (100 * correct / total))
Cmatrixs = Confusion_mxtrix(labels_value,predicted_value,5)
print(Cmatrixs)

# Evaluate(Cmatrixs)
# TPR_i, FPR_i = MyROC_i(outputs_value, labels_value)
# Plot_ROC_i(TPR_i, FPR_i)
#
# def kappa(matrix):
#     matrix=np.array(matrix)
#     n=np.sum(matrix)
#     sum_po=0
#     sum_pe=0
#     for i in range(len(matrix[0])):
#         sum_po+=matrix[i][i]
#         row=np.sum(matrix[i,:])
#         col=np.sum(matrix[:,i])
#         sum_pe+=row*col
#     po=sum_po/n
#     pe=sum_pe/(n*n)
#     return (po-pe)/(1-pe)
#
# print(kappa(Cmatrixs))
