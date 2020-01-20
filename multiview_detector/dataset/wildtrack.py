import os
import json
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *

UNIT = 1
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']


class WildtrackFrame(VisionDataset):
    def __init__(self, root, train=True, transform=ToTensor(), target_transform=None, reID=False, train_ratio=0.9):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.num_cam, self.num_frame = 7, 2000
        self.reID = reID
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = {cam: {} for cam in range(self.num_cam)}
        self.img_gt = {}
        self.intrinsic_matrices, self.extrinsic_matrices = {}, {}
        occupancy_map_shape = np.array([480, 1440], dtype=int)

        for camera_folder in sorted(os.listdir(os.path.join(root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            for fname in sorted(os.listdir(os.path.join(root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    self.img_fpaths[cam][frame] = os.path.join(root, 'Image_subsets', camera_folder, fname)
            pass

        for fname in sorted(os.listdir(os.path.join(root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                for single_pedestrian in all_pedestrians:
                    x, y = get_worldgrid_from_posid(single_pedestrian['positionID'])
                    i_s.append(x)
                    j_s.append(y)
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=occupancy_map_shape)
                self.img_gt[frame] = occupancy_map

        for cam in range(self.num_cam):
            intrinsic_matrix, extrinsic_matrix = self.get_intrinsic_extrinsic_matrix(cam)
            self.intrinsic_matrices[cam] = intrinsic_matrix
            self.extrinsic_matrices[cam] = extrinsic_matrix

    def __getitem__(self, index):
        frame = list(self.img_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        gt = torch.from_numpy(self.img_gt[frame].toarray())
        if self.target_transform is not None:
            gt = self.target_transform(gt)
        return imgs, gt

    def __len__(self):
        return len(self.img_gt.keys())

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        distortion_coeff = intrinsic_params_file.getNode('distortion_coefficients').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        def get_extrinsic_matrix(rvec, tvec):
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
            projection_matrix = np.hstack((rotation_matrix, translation_matrix))
            return projection_matrix, rotation_matrix, translation_matrix

        extrinsic_matrix, _, _ = get_extrinsic_matrix(rvec, tvec)

        return intrinsic_matrix, extrinsic_matrix


def test():
    import matplotlib.pyplot as plt

    dataset = WildtrackFrame(os.path.expanduser('~/Data/Wildtrack'))
    intrinsic_matrices, extrinsic_matrices = zip(*[dataset.get_intrinsic_extrinsic_matrix(cam)
                                                   for cam in range(dataset.num_cam)])
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    for cam in range(7):
        world_grids = get_worldgrid_from_imagecoord(image_coords.transpose(), intrinsic_matrices[cam],
                                                    extrinsic_matrices[cam]).transpose().reshape([H, W, 2])
        world_grid_map = np.zeros([480, 1440])
        for i in range(H):
            for j in range(W):
                x, y = world_grids[i, j]
                if x in range(480) and y in range(1440):
                    world_grid_map[int(x), int(y)] += 1
        world_grid_map = world_grid_map != 0
        plt.imshow(world_grid_map)
        plt.show()
        world_grid_maps.append(world_grid_map)
        pass
    plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    plt.show()
    pass
    imgs, gt = dataset.__getitem__(0)


if __name__ == '__main__':
    test()
