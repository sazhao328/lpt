import gzip
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import time

start = time.time()

with gzip.open('data/mnist/mnist.pkl.gz', 'rb') as f:
    (train_x, train_y), (valid_x, valid_y), _ = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.tensor, (train_x, train_y, valid_x, valid_y))
n, c = x_train.shape

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print('device', device)

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=64)


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        ).to(device)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = self.model(xb)
        return xb.view(-1, xb.size(1))


loss_func = nn.CrossEntropyLoss().to(device)
model = Mnist_Logistic()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
epochs = 20


def fit():
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb.to(device)), yb.to(device)) for xb, yb in valid_dl)
            print(epoch, valid_loss)


fit()

print(time.time() - start)
