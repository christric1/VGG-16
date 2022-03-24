import torch
import torch.nn as nn
from torchsummary import summary
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import VGG16 

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device : ", device)

    # create model
    model = VGG16(VGG16.VGG_16).to(device)
    summary(model, (3, 32, 32))

    writer = SummaryWriter(comment="VGG16")

    # define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=VGG16.learning_rate)