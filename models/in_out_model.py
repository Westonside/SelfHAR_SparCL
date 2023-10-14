import torch
import torch.nn as nn
import torch.nn.functional as F

class InOut(nn.Module):
    def __init__(self, in_planes: int, classes: int):
        super(InOut, self).__init__()
        # self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(in_planes, 512) # pass to hidden layer of 512
        self.layer2 = nn.Linear(512, classes)

    def forward(self, x):
        # x = self.flatten(x)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x