from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(3, stride=2)

        self.linear4 = nn.Linear(2048, 1024, bias=true)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(1024, 512, bias=true)
        self.relu5 = nn.ReLU()

        self.linear6 = nn.Linear(512, 3, bias=true)
        self.softmax6 = nn.Softmax()

        # À compléter

    def forward(self, x):

        # À compléter
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.maxpool1(x2)

        x4 = self.conv2(x3)
        x5 = self.relu2(x4)
        x6 = self.maxpool2(x5)

        x7 = self.conv3(x6)
        x8 = self.relu3(x7)
        x9 = self.maxpool3(x8)

        x10 = x9.reshape((x.size(0), x9.size(1) * x9.size(2) * x9.size(3)))

        x11 = self.linear4(x10)
        x12 = self.relu4(x11)

        x13 = self.linear5(x12)
        x14 = self.relu5(x13)

        x15 = self.linear6(x14)

        output = x15

        return output
