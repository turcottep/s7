from matplotlib.pyplot import axis
from numpy import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegmentationNetwork, self).__init__()

        oldsize = 1
        size = 16
        self.block1 = nn.Sequential(
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=(1)),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
        )
        oldsize = size
        size *= 2
        print('block2 size', size, 'oldsize', oldsize)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
        )
        oldsize = size
        size *= 2
        print('block3 size', size, 'oldsize', oldsize)
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
        )
        oldsize = size
        size *= 2
        print('block4 size', size, 'oldsize', oldsize)
        self.block4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
        )
        oldsize = size
        size *= 2
        print('block5 size', size, 'oldsize', oldsize)
        self.block5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size//2, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(),
            nn.ConvTranspose2d(size//2, size//2, 2, padding=0, stride=2)
        )

        oldsize = size
        size = oldsize//2
        print('block6 size', size, 'oldsize', oldsize)
        self.block6 = nn.Sequential(
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size//2, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(),
            nn.ConvTranspose2d(size//2, size//2, 2, padding=0, stride=2)
        )
        oldsize = size
        size = oldsize//2
        print('block7 size', size, 'oldsize', oldsize)
        self.block7 = nn.Sequential(
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size//2, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(),
            nn.ConvTranspose2d(size//2, size//2, 2, padding=0, stride=2)
        )
        oldsize = size
        size = oldsize//2
        print('block8 size', size, 'oldsize', oldsize)
        self.block8 = nn.Sequential(
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size//2, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(),
            nn.ConvTranspose2d(size//2, size//2, 2, padding=0, stride=2)
        )
        oldsize = size
        size = oldsize//2
        print('block9 size', size, 'oldsize', oldsize)
        self.block9 = nn.Sequential(
            nn.Conv2d(oldsize, size, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Conv2d(size, size//2, 3, stride=1,  padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(),
            nn.Conv2d(size//2, 4, 1, stride=1,  padding=0),
        )

    def forward(self, x):
        xclean = F.pad(x, (5, 6, 5, 6))
        x1 = self.block1(xclean)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        # print("x1", x1.size())
        # print("x2", x2.size())
        # print("x3", x3.size())
        # print("x4", x4.size())
        # print("x5", x5.size())

        x6input = torch.cat((x4, x5), axis=1)
        # print("x6input", x6input.size())
        x6 = self.block6(x6input)

        # print("x6", x6.size())
        x7input = torch.cat((x3, x6), axis=1)
        # print("x7input", x7input.size())
        x7 = self.block7(x7input)
        # print("x7", x7.size())
        x8input = torch.cat((x2, x7), axis=1)
        # print("x8input", x8input.size())
        x8 = self.block8(x8input)
        # print("x8", x8.size())
        x9input = torch.cat((x1, x8), axis=1)
        # print("x9input", x9input.size())
        x9 = self.block9(x9input)
        # print("x9", x9.size())
        output = x9
        return output
