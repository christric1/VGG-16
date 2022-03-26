import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# define hyperParameter
batch_size = 32  # Batch size
learning_rate = 1e-2  # Learning rate
optimizer = "SGD"
num_epoches = 50

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# download the dataset
train_dataset = datasets.CIFAR10('./cifar10', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10('./cifar10', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def Show_model():
    print("\n")
    model_ = VGG16_Dropout_BN(cfg).to(device)
    summary(model_, (3, 32, 32))


# model
class VGG16_Dropout_BN(nn.Module):
    def __init__(self):
        super(VGG16_Dropout_BN, self).__init__()

        self.features = self.make_layer(cfg)
        self.classifier = nn.Sequential(
            # 全連接層
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 10),
        )

    def make_layer(self, cfg):
        layer = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layer += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(True)]  # 預設為 False, 表示新建一個對象對其修改 ; True 則表示直接對這個對象進行修改
                in_channels = x

            layer += [nn.AvgPool2d(kernel_size=1, stride=1)]    # 多加?
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class VGG16_Baseline(nn.Module):
    def __init__(self) -> None:
        super(VGG16_Baseline, self).__init__()

        self.features = self.make_layer(cfg)
        self.classifier = nn.Sequential(
            # 全連接層
            nn.Linear(512, 100),
            nn.ReLU(True),

            nn.Linear(100, 100),
            nn.ReLU(True),

            nn.Linear(100, 10),
        )

    def make_layer(self, cfg):
        layer = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layer += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.ReLU(True)]  
                in_channels = x

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

        