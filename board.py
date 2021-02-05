import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

trainset = torchvision.datasets.FashionMNIST("./data", train=True, transform=transform, download=True)
testset = torchvision.datasets.FashionMNIST("./data", train=False, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64)
testloader = DataLoader(testset, batch_size=64)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../cloud_jibei/runs/fashion_mnist_experiment_1')
dataiter = iter(trainloader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('four_fashion_mnist_images', img_grid)

writer.add_graph(net, images)


def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


images, labels = select_n_random(images, labels, 100)
class_labels = [classes[label] for label in labels]
features = images.view(-1, 28 * 28)
writer.add_embedding(features, metadata=class_labels, label_img=images)


def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = preds_tensor.numpy()
    return preds, [F.softmax(el, 0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))

    for i in range(4):
        ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[i], one_channel=True)
        ax.set_title(f'{classes[preds[i]]}, {probs[i] * 100:.2f} % \n (label:{classes[labels[i]]})',
                     color=('green' if preds[i] == labels[i] else 'red'))

    return fig


running_loss = 0.0
for epoch in range(3):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:
            print('training loss', running_loss / 100, epoch * len(trainloader) + i)
            writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')

writer.close()
