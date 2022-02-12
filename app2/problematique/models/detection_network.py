import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_params):
        super(DetectionNetwork, self).__init__()

        self.f1 = nn.Conv2d(1, 32, 5, stride=1, padding=1)
        self.f2 = nn.LeakyReLU()
        self.f3 = nn.MaxPool2d(2, 2)

        self.f4 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.f5 = nn.LeakyReLU()
        self.f6 = nn.MaxPool2d(2, 2)

        self.f7 = nn.Conv2d(64, 128, 5, stride=1, padding=1)
        self.f8 = nn.LeakyReLU()

        self.f9 = nn.Conv2d(128, 64, 5, stride=1, padding=1)
        self.f10 = nn.LeakyReLU()
        self.f11 = nn.MaxPool2d(2, 2)

        self.f12 = nn.Conv2d(64, 128, 5, stride=1, padding=1)
        self.f13 = nn.LeakyReLU()

        self.f14 = nn.Conv2d(128, 256, 5, stride=1, padding=1)
        self.f15 = nn.LeakyReLU()

        self.f16 = nn.Conv2d(256, 128, 5, stride=1, padding=1)
        self.f17 = nn.LeakyReLU()
        self.f18 = nn.MaxPool2d(2, 2)

        self.f19 = nn.Conv2d(128, 256, 5, stride=1, padding=1)
        self.f20 = nn.LeakyReLU()

        self.f21 = nn.Conv2d(256, 512, 5, stride=1, padding=1)
        self.f22 = nn.ReLU()

        #self.f  =nn.Conv2d(512, 30*7*7, 5, stride=1, padding=1)

    def forward(self, x):

        x1 = self.f1(x)
        x2 = self.f1(x1)
        x3 = self.f1(x2)
        x4 = self.f1(x3)
        x5 = self.f1(x4)
        x6 = self.f1(x5)
        x7 = self.f1(x)
        x8 = self.f1(x)
        x9 = self.f1(x)
        x10 = self.f1(x)
        x11 = self.f1(x)
        x12 = self.f1(x)
        x13 = self.f1(x)
        x14 = self.f1(x)
        x15 = self.f1(x)
        x16 = self.f1(x)
        x17 = self.f1(x)
        x18 = self.f1(x)
        x19 = self.f1(x)
        x20 = self.f1(x)
        x21 = self.f1(x)
        x22 = self.f1(x)

        output = x
        return output
