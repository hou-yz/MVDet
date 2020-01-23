import os
import numpy as np
import torch
import torch.nn as nn
import kornia
from multiview_detector.model.vgg import vgg11
import matplotlib.pyplot as plt


class BBOXClassifier(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.base = vgg11(in_channels=3 * dataset.num_cam).features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                             nn.Linear(512, 2, bias=False), )
        pass

    def forward(self, imgs, visualize=False):
        x = self.base(imgs)
        x = self.avg_pool(x)
        x = self.classifier_head(x.squeeze())
        return x


def test():
    from multiview_detector.dataset.wildtrack_bbox import WildtrackBBOX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([256, 128]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = WildtrackBBOX(os.path.expanduser('~/Data/Wildtrack'), transform=transform)
    dataloader = DataLoader(dataset, 64, True, num_workers=0)
    imgs, gt = next(iter(dataloader))
    model = BBOXClassifier(dataset)
    res = model(imgs)
    pass


if __name__ == '__main__':
    test()
