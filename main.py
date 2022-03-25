import torch
import torch.nn as nn
from torchsummary import summary
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import VGG16 

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device : ", device)

    # create model
    model = VGG16.VGG16_Dropout_BN().to(device)
    summary(model, (3, 32, 32))

    writer = SummaryWriter(comment="VGG16")

    # define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=VGG16.learning_rate)

    # train model
    for epoch in range(VGG16.num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        count = 0.0

        for i, data in tqdm(enumerate(VGG16.train_loader, 0)):  # show progress bar
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
        for data in VGG16.test_loader:  # Test model
            img, label = data
            img, label = img.to(device), label.to(device)

            out = model(img)
            loss = criterion(out, label)
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Accuracy of the network : %d %%' % (
            100 * correct / total))