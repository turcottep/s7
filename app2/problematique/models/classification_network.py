import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        
        # À compléter
        
     

    def forward(self, x):
        
        # À compléter
        output = self.conv1(x)
        
        return output
