import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(criterion, outputs, targets):
        return criterion(F.softmax(outputs,dim=1).max(1).indices.float(), targets) if isinstance(criterion,nn.MSELoss) else criterion(outputs,targets)