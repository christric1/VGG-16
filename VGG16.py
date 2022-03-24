import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# define hyperParameter
batch_size = 32  # Batch size
learning_rate = 1e-2  # Learning rate
optimizer = "SGD"
num_epoches = 50

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# download the dataset
train_dataset = datasets.CIFAR10('../cifar10', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10('../cifar10', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

VGG_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# model
class VGG16(nn.Module):
    def __init__(self, cfg):
        super(VGG16, self).__init__()

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

def Show_model():
    print("\n")
    model_ = VGG16(VGG_16).to(device)
    summary(model_, (3, 32, 32))


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device : ", device)

    # create model
    model = VGG16(VGG_16).to(device)
    summary(model, (3, 32, 32))

    writer = SummaryWriter(comment="VGG16")

    # define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        count = 0.0

        for i, data in tqdm(enumerate(train_loader, 0)):  # show progress bar
            img, label = data
            img, label = img.to(device), label.to(device)

            # Forward
            out = model(img)  # 64 images output, [64, 10]
            loss = criterion(out, label)
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # back forward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        writer.add_scalar("training loss ", running_loss / count, epoch + 1)
        writer.add_scalar("accuracy", 100 * correct / total, epoch + 1)

        print('epoch %d loss: %.3f' % (epoch + 1, running_loss / count))

    print('Finished Training')
    torch.save(model.state_dict(), '../model/VGG16.pth')  # save trained model

    # Test
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:  # Test model
            img, label = data
            img, label = img.to(device), label.to(device)

            out = model(img)
            loss = criterion(out, label)
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Accuracy of the network : %d %%' % (
            100 * correct / total))