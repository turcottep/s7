from matplotlib.pyplot import axis
from numpy import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegmentationNetwork, self).__init__()

        oldsize = 1
        size = 12
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

        x6input = torch.cat((x4, x5), axis=1)
        x6 = self.block6(x6input)

        x7input = torch.cat((x3, x6), axis=1)
        x7 = self.block7(x7input)
        x8input = torch.cat((x2, x7), axis=1)
        x8 = self.block8(x8input)
        x9input = torch.cat((x1, x8), axis=1)
        x9 = self.block9(x9input)

        output = x9
        return output
