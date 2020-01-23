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

        self.root, self.train = root, train
        self.num_cam, self.num_frame = 7, 2000
        self.reID, self.np_ratio = reID, pn_ratio
        # H,W; N_row,N_col
        self.worldgrid_shape = [480, 1440]
        self.downscaled_grid_shape, self.downscale = [60, 180], 8
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

        positive_frame_pos_by_index = []
        positive_pos_by_frame = {}
        positive_pid_by_frame = {}
        negative_frame_pos_by_index = []
        self.all_frame_pos_by_index = []
        all_pids = {}
        for fname in sorted(os.listdir(os.path.join(root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                pos_s, pid_s = [], []
                for single_pedestrian in all_pedestrians:
                    pid = single_pedestrian['personID']
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids) + 1
                    pid = all_pids[pid]
                    quantized_pos = quantize_pos(single_pedestrian['positionID'], self.downscale)
                    positive_frame_pos_by_index.append({'frame': frame,
                                                        'positionID': quantized_pos,
                                                        'personID': pid})
                    pos_s.append(quantized_pos)
                    pid_s.append(pid)
                positive_pos_by_frame[frame] = pos_s
                positive_pid_by_frame[frame] = pid_s
        self.num_pids = len(all_pids)
        if train:
            for i in range(len(positive_frame_pos_by_index)):
                frame, pos, pid = positive_frame_pos_by_index[i]['frame'], \
                                  positive_frame_pos_by_index[i]['positionID'], 0
                for j in range(self.np_ratio):
                    reduced_pos = np.random.randint(0, np.prod(self.downscaled_grid_shape))
                    pos = reduced_to_og_pos(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    while pos in positive_pos_by_frame[frame]:
                        reduced_pos = np.random.randint(0, np.prod(self.downscaled_grid_shape))
                        pos = reduced_to_og_pos(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    negative_frame_pos_by_index.append({'frame': frame,
                                                        'positionID': pos,
                                                        'personID': pid})
            self.all_frame_pos_by_index = positive_frame_pos_by_index + negative_frame_pos_by_index
        else:
            for frame in positive_pos_by_frame.keys():
                for reduced_pos in range(int(np.prod(self.downscaled_grid_shape))):
                    pos = reduced_to_og_pos(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    if pos in positive_pos_by_frame[frame]:
                        pid = positive_pid_by_frame[frame][positive_pos_by_frame[frame].index(pos)]
                    else:
                        pid = 0
                    self.all_frame_pos_by_index.append({'frame': frame, 'positionID': pos, 'personID': pid})

        self.bbox_by_pos_cam = read_pom(os.path.join(root, 'rectangles.pom'))

        pass

    def __getitem__(self, index, visualize=False):
        frame, pos, pid = self.all_frame_pos_by_index[index]['frame'], \
                          self.all_frame_pos_by_index[index]['positionID'], \
                          self.all_frame_pos_by_index[index]['personID']
        imgs = []
        for cam in range(self.num_cam):
            if self.bbox_by_pos_cam[pos][cam] is not None:
                fpath = self.img_fpaths[cam][frame]
                img = Image.open(fpath).convert('RGB')
                left, top, right, bottom = self.bbox_by_pos_cam[pos][cam]
                img = img.crop([left, top, right, bottom])
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
        target = pid if self.reID else int(pid > 0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return imgs, target

    def __len__(self):
        return len(self.all_frame_pos_by_index)


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


def quantize_pos(pos, downscale):
    og_x, og_y = get_worldgrid_from_posid(pos)
    quantized_x, quantized_y = int((og_x // downscale + 0.5) * downscale), \
                               int((og_y // downscale + 0.5) * downscale)
    quantized_pos = get_posid_from_worldgrid([quantized_x, quantized_y])
    return quantized_pos


def reduced_to_og_pos(reduced_pos, downscale, reduced_grid_shape):
    reduced_x = reduced_pos % reduced_grid_shape[0]
    reduced_y = reduced_pos // reduced_grid_shape[0]
    quantized_x, quantized_y = int((reduced_x + 0.5) * downscale), \
                               int((reduced_y + 0.5) * downscale)
    quantized_pos = get_posid_from_worldgrid([quantized_x, quantized_y])
    return quantized_pos


def test():
    # read_pom(os.path.expanduser('~/Data/Wildtrack/rectangles.pom'))
    dataset = WildtrackBBOX(os.path.expanduser('~/Data/Wildtrack'))
    imgs, gt = dataset.__getitem__(1, True)
    pass


if __name__ == '__main__':
    test()
