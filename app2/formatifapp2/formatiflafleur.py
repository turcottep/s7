

import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 10, kernel_size=8, stride=1, padding=0)
        self.softmax5 = nn.Softmax(dim=1)

        # self.lin1 = nn.Linear(x15.size(1), 128, bias=True)
        # self.lin1 = nn.Linear(128, 10, bias=True)




    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.relu1(x2)

        x4 = self.conv2(x3)
        x5 = self.maxpool2(x4)
        x6 = self.relu2(x5)

        x7 = self.conv3(x6)
        x8 = self.maxpool3(x7)
        x9 = self.relu3(x8)

        x10 = self.conv4(x9)
        x11 = self.maxpool4(x10)
        x12 = self.relu4(x11)

        x13 = self.conv5(x12)
        x14 = self.softmax5(x13)

        #x15 = x14.reshape(x14.size(0),(x14.size(1)*x14.size(2)*x14.size(3))) flatten


        output = x14
        return output

def main():
    print("main!")
    reseau = Model1()

    n = 4
    image = torch.zeros((n,3,128,128))
    output = reseau(image)
    output2 = output.squeeze()
    print("output", output2.size())

if __name__ == "__main__":
    main()