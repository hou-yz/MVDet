import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._traget_transform(x, target, kernel)
        return F.mse_loss(x, target)

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])

        kernel_size, padding = kernel.shape[0], int((kernel.shape[0] - 1) / 2)
        weight = torch.from_numpy(kernel).float().view([1, 1, kernel_size, kernel_size])
        with torch.no_grad():
            target = F.conv2d(target, weight.to(target.device), padding=padding)

        return target
