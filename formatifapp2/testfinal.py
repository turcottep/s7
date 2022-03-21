

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)

        output = x1
        return output


def main():
    print("helli")
    myNetwork = Network()
    output = myNetwork(torch.rand((32,3,128,128)))
    print(output.size())

if __name__ == "__main__":
    main()