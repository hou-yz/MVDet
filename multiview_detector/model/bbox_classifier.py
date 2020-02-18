import os
import numpy as np
import torch
import torch.nn as nn
from multiview_detector.model.vgg import vgg11
from multiview_detector.model.resnet import resnet18, resnet50


class BBOXClassifier(nn.Module):
    def __init__(self, num_cam, arch='vgg11'):
        super().__init__()
        if arch == 'vgg11':
            self.base = vgg11(in_channels=3 * num_cam).features
            out_channel = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(in_channels=3 * num_cam).children())[:-2])
            out_channel = 512
        elif arch == 'resnet50':
            self.base = nn.Sequential(*list(resnet50(in_channels=3 * num_cam).children())[:-2])
            out_channel = 2048
        else:
            raise Exception
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(nn.Linear(out_channel, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                             nn.Linear(512, 2, bias=False), )
        pass

    def forward(self, imgs, visualize=False):
        x = self.base(imgs)
        x = self.avg_pool(x)
        x = self.classifier_head(x.squeeze())
        return x


def test():
    from multiview_detector.dataset.bboxDataset import bboxDataset
    from multiview_detector.dataset.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([256, 128]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = bboxDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 64, True, num_workers=0)
    imgs, gt, _ = next(iter(dataloader))
    model = BBOXClassifier(dataset.num_cam)
    res = model(imgs)
    pass


if __name__ == '__main__':
    test()
