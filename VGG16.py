import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# define hyperParameter
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
optimizer = "SGD"
num_epoch = 50

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# download the dataset
train_dataset = datasets.CIFAR10('./cifar10', train=True,
                                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                               transforms.RandomCrop(32, 4),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])]),
                                 download=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.CIFAR10('./cifar10', train=False,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])]),
                                download=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
DP = [0.3, 0.3, 'X', 0.3, 0.3, 'X', 0.4, 0.4, 0.4, 'X', 0.4, 0.4, 0.4, 'X', 0.5, 0.5, 0.5, 'X']

__all__ = ['VGG16', 'VGG16_BN']


# model
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()

        self.features = features
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

        # Initialize weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def make_layer(batch_norm=False):
    layer = []
    in_channels = 3

    for index, x in enumerate(cfg):
        if x == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]

            in_channels = x
    return nn.Sequential(*layer)


def VGG16():
    return VGG(make_layer())


def VGG16_BN():
    return VGG(make_layer(batch_norm=True))

