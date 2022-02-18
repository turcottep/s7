import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_params):
        super(DetectionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(3, stride=2)

        self.linear4 = nn.Linear(512, 256, bias=True)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(256, 512, bias=True)
        self.relu5 = nn.ReLU()

        self.linear6 = nn.Linear(512, 3*7, bias=True)
        self.sigmoid6 = nn.Sigmoid()

        # À compléter

    def forward(self, x):

        # À compléter
        x1 = self.conv1(x)
        x1 = self.batchnorm1(x1)
        x2 = self.relu1(x1)
        x3 = self.maxpool1(x2)

        x4 = self.conv2(x3)
        x4 = self.batchnorm2(x4)
        x5 = self.relu2(x4)
        x6 = self.maxpool2(x5)

        x7 = self.conv3(x6)
        x7 = self.batchnorm3(x7)
        x8 = self.relu3(x7)
        x9 = self.maxpool3(x8)

        x10 = x9.reshape((x.size(0), x9.size(1) * x9.size(2) * x9.size(3)))
        # print("x10.size() = ", x10.size())
        x11 = self.linear4(x10)
        x12 = self.relu4(x11)

        x13 = self.linear5(x12)
        x14 = self.relu5(x13)

        x15 = self.linear6(x14)
        x16 = self.sigmoid6(x15)

        output = x16.reshape((x.size(0), 3, 7))

        return output

# class DetectionNetworkLoss(nn.Module):
#     def __init__(self):
#         super(DetectionNetworkLoss, self).__init__()

#         self.lossBCE = nn.BCEWithLogitsLoss()
#         self.lossCrossEntropy = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
#         self.lossMSE = nn.MSELoss()

#     def forward(self, x, y):
#         # À compléter
#         lossBCE = self.lossBCE(x[:, :, 0], y[:, :, 0])
#         lossMSE = self.lossMSE(self.sigmoid(x[:,:,1:4]), self.sigmoid(y[:,:,1:4]))
#         lossCrossEntropy = self.lossCrossEntropy(x[:, :, 3:7], y[:, :, 3:7])

#         loss = 0.25 * lossBCE + 0.25 *lossMSE + lossCrossEntropy
#         return loss
