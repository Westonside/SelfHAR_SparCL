import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_


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


    def extend_fc_layer(self, new_classes: int):
        # this will extend the last layer of the model to have more classes based on the classes
        if self.layer2 is not None:
            old = self.layer2
            with torch.no_grad():
                new_fc = nn.Linear(old.in_features, old.out_features + new_classes).cuda()
                w = torch.nn.init.kaiming_normal_(new_fc.weight.data, nonlinearity='relu', mode='fan_in', a=0)
                new_fc.weight.data = w  # set the randomly initialized weights to the new layer
                # this results in only the new classes being randomly initialized
            try:
                weight = copy.deepcopy(self.layer2.weight.data) #copy the weights from the old layer
                new_fc.weight.data[:self.layer2.out_features] = weight # set the weights to the new layer

                if old.bias is not None:
                    bias = copy.deepcopy(old.bias.data)
                    new_fc.bias.data[:old.out_features] = bias
            except:
                nb_output = self.layer2.module.out_features
                weight = copy.deepcopy(self.layer2.module.weight.data)
                new_fc.weight.data[:nb_output] = weight

                if self.layer2.module.bias is not None:
                    bias = copy.deepcopy(self.layer2.module.bias.data)
                    new_fc.bias.data[:nb_output] = bias

            del self.layer2
            self.layer2 = new_fc # set the new layer to the model