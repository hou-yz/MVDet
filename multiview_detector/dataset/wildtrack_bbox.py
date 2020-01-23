import os
import re
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
import torchvision.transforms as T
from multiview_detector.utils.projection import *


class WildtrackBBOX(VisionDataset):
    def __init__(self, root, train=True, transform=T.Compose([T.Resize([256, 128]), T.ToTensor()]),
                 target_transform=None, reID=False, train_ratio=0.9, pn_ratio=1):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.num_cam, self.num_frame = 7, 2000
        self.reID, self.pn_ratio = reID, pn_ratio
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = {cam: {} for cam in range(self.num_cam)}

        for camera_folder in sorted(os.listdir(os.path.join(root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            for fname in sorted(os.listdir(os.path.join(root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    self.img_fpaths[cam][frame] = os.path.join(root, 'Image_subsets', camera_folder, fname)
            pass

        self.frame_pos_by_index = []
        self.pos_s_by_frame = {}
        all_pids = {}
        for fname in sorted(os.listdir(os.path.join(root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                pos_s = []
                for single_pedestrian in all_pedestrians:
                    pid = single_pedestrian['personID']
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids) + 1
                    pid = all_pids[pid]
                    self.frame_pos_by_index.append({'frame': frame,
                                                    'positionID': single_pedestrian['positionID'],
                                                    'personID': pid})
                    pos_s.append(single_pedestrian['positionID'])
                self.pos_s_by_frame[frame] = pos_s
        self.num_pids = len(all_pids)

        self.bbox_by_pos_cam = read_pom(os.path.join(root, 'rectangles.pom'))

        pass

    def __getitem__(self, index, visualize=False):
        target = 1 if index % (self.pn_ratio + 1) == 0 else 0
        index = int(index / (self.pn_ratio + 1))
        frame, pos, pid = self.frame_pos_by_index[index]['frame'], self.frame_pos_by_index[index]['positionID'], \
                          self.frame_pos_by_index[index]['personID']
        if not target:
            pos = np.random.randint(0, np.prod(self.worldgrid_shape))
            while pos in self.pos_s_by_frame[frame]:
                pos = np.random.randint(0, np.prod(self.worldgrid_shape))

        imgs = []
        for cam in range(self.num_cam):
            if self.bbox_by_pos_cam[pos][cam] is not None:
                fpath = self.img_fpaths[cam][frame]
                img = Image.open(fpath).convert('RGB')
                left, top, right, bottom = self.bbox_by_pos_cam[pos][cam]
                img = img.crop((left, top, right, bottom))
            else:
                img = Image.fromarray(np.zeros([256, 128, 3], np.uint8))
                pass
            if visualize:
                plt.imshow(img)
                plt.show()
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.cat(imgs)
        if self.reID:
            target = pid
        if self.target_transform is not None:
            target = self.target_transform(target)
        return imgs, target

    def __len__(self):
        return len(self.frame_pos_by_index) * (self.pn_ratio + 1)


def read_pom(fpath):
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r'(\d+) (\d+)')
    cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
    with open(fpath, 'r') as fp:
        for line in fp:
            if 'RECTANGLE' in line:
                cam, pos = map(int, cam_pos_pattern.search(line).groups())
                if pos not in bbox_by_pos_cam:
                    bbox_by_pos_cam[pos] = {}
                if 'notvisible' in line:
                    bbox_by_pos_cam[pos][cam] = None
                else:
                    cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                    bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0), min(right, 1920), min(bottom, 1080)]
    return bbox_by_pos_cam


def test():
    # read_pom(os.path.expanduser('~/Data/Wildtrack/rectangles.pom'))
    dataset = WildtrackBBOX(os.path.expanduser('~/Data/Wildtrack'))
    imgs, gt = dataset.__getitem__(1, True)
    pass


if __name__ == '__main__':
    test()
