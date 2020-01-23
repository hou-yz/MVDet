import os
import numpy as np
import torch
import torch.nn as nn
import kornia
from torchvision.models.vgg import vgg11, vgg16_bn
from torchvision.models.alexnet import alexnet

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.num_cam, self.featmap_reduce = dataset.num_cam, dataset.featmap_reduce
        self.img_shape, self.featmap_shape = dataset.img_shape, dataset.featmap_shape
        intrinsic_matrices, extrinsic_matrices = zip(*[dataset.get_intrinsic_extrinsic_matrix(cam)
                                                       for cam in range(dataset.num_cam)])
        self.projection_matrices = self.get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices)

        self.base = vgg11().features
        self.base[-1] = nn.Sequential()
        # 2.5cm -> 0.5m: 20x
        self.classifier_head = nn.Sequential(nn.Conv2d(512 * 7 + 2, 512, 5, 1, 2), nn.ReLU(),
                                             nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                             nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                             nn.Conv2d(512, 1, 1, 1, bias=False), nn.Sigmoid())

        self.coord_map = self.create_coord_map(self.featmap_shape + [1])
        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        for cam in range(self.num_cam):
            img_feature = self.base(imgs[:, cam])
            feature_shape = np.array(img_feature.shape[2:])
            img_zoom_mat = np.diag(np.append(np.array(self.img_shape) / feature_shape, [1]))
            featmap_zoom_mat = np.diag(np.append(np.ones([2]) / self.featmap_reduce, [1]))
            proj_mat = torch.from_numpy(
                np.matmul(featmap_zoom_mat, np.matmul(self.projection_matrices[cam], img_zoom_mat))).repeat(
                [B, 1, 1]).float().to(img_feature.device)
            world_feature = kornia.warp_perspective(img_feature, proj_mat, self.featmap_shape)
            if visualize:
                plt.imshow(world_feature[0, 0].detach().numpy() != 0)
                plt.show()
            world_features.append(world_feature)

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(imgs.device)], dim=1)
        detector_result = self.classifier_head(world_features)
        detector_result = nn.functional.interpolate(detector_result, self.featmap_shape, mode='bilinear')
        return detector_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = np.matmul(intrinsic_matrices[cam], np.delete(extrinsic_matrices[cam], 2, 1))
            worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
            worldgrid2imgcoord_mat = np.matmul(worldcoord2imgcoord_mat, worldgrid2worldcoord_mat)
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = np.matmul(permutation_mat, imgcoord2worldgrid_mat)
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

    def perspective_trans(src, M, ):
        dst = np.matmul(M, np.append(src, [1])[:, np.newaxis]).squeeze()
        dst = dst[:2] / dst[2]
        return dst

    transform = T.Compose([T.Resize([360, 640]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = WildtrackFrame(os.path.expanduser('~/Data/Wildtrack'), transform=transform)
    dataloader = DataLoader(dataset, 1, True, num_workers=0)
    imgs, gt = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    res = model(imgs, visualize=True)
    # # test without calling the model forward
    # imgs = torch.ones([1, 1, 1080, 1920]).float()
    # for cam in range(dataset.num_cam):
    #     proj_mat = torch.from_numpy(model.projection_matrices[cam]).repeat(
    #         [imgs.shape[0], 1, 1]).float()
    #
    #     world_feature = kornia.warp_perspective(imgs, proj_mat, dataset.worldgrid_shape)
    #     plt.imshow(world_feature[0, 0].detach().numpy() != 0)
    #     plt.show()
    #     pass
    pass


if __name__ == '__main__':
    test()
