import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg11
from torchvision.models.mobilenet import mobilenet_v2
from multiview_detector.model.resnet import resnet18, resnet50

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam, self.grid_reduce = dataset.num_cam, dataset.grid_reduce
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        intrinsic_matrices, extrinsic_matrices = zip(*[dataset.get_intrinsic_extrinsic_matrix(cam)
                                                       for cam in range(dataset.num_cam)])
        self.projection_matrices = self.get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices)

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        # elif arch == 'mobilenet':
        #     self.base = mobilenet_v2().features
        #     out_channel = 1280
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        # elif arch == 'resnet50':
        #     self.base = nn.Sequential(*list(resnet50(replace_stride_with_dilation=[False, True, True]).children())[:-2])
        #     out_channel = 2048
        else:
            raise Exception
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 32, 1), nn.ReLU(),
                                            nn.Conv2d(32, 1, 1, bias=False)).to('cuda:0')

        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * 7 + 2, 512, 1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 1, bias=False)).to('cuda:0')

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        umsample_shape = list(map(lambda x: int(x / 4), self.img_shape))
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:1'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, umsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)
            feature_shape = np.array(img_feature.shape[2:])
            img_reduce = np.array(self.img_shape) / feature_shape
            img_zoom_mat = np.diag(np.append(img_reduce, [1]))
            featmap_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
            proj_mat = torch.from_numpy(featmap_zoom_mat @ self.projection_matrices[cam] @ img_zoom_mat
                                        ).repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = kornia.warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape)
            if visualize:
                plt.imshow(world_feature[0, 0].detach().cpu().numpy())
                plt.show()
            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        map_result = self.map_classifier(world_features.to('cuda:0'))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.dataset.wildtrack_frame import WildtrackFrame
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = WildtrackFrame(os.path.expanduser('~/Data/Wildtrack'), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
