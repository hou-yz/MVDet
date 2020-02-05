import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):

    def __init__(self, kernel):
        super().__init__()
        kernel_size, padding = kernel.shape[0], int((kernel.shape[0] - 1) / 2)
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.gaussian_filter.weight = nn.Parameter(
            torch.from_numpy(kernel).float().view([1, 1, kernel_size, kernel_size]),
            requires_grad=False)

    def forward(self, x, target):
        target = self._traget_transform(x, target)
        return F.mse_loss(x, target)

    def _traget_transform(self, x, target):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = self.gaussian_filter.to(target.device)(target)
        return target
