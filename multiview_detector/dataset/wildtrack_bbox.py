import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

sys.path.insert(0, '/home/houyz/Code/multiview_one_stage')
import re
import json
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
import torchvision.transforms as T
from multiview_detector.utils.projection import *


class WildtrackBBOX(VisionDataset):
    def __init__(self, root, split='train', transform=T.Compose([T.Resize([256, 128]), T.ToTensor()]),
                 reID=False, train_ratio=0.9, np_ratio=3, force_download=False):
        super().__init__(root, transform=transform)

        self.root, self.split = root, split
        self.num_cam, self.num_frame = 7, 2000
        self.reID, self.np_ratio = reID, np_ratio
        # H,W; N_row,N_col
        self.worldgrid_shape = [480, 1440]
        self.downscaled_grid_shape, self.downscale = [60, 180], 8

        self.fpath_header = os.path.join(
            f'./data/WildtrackBBOX/{self.downscaled_grid_shape[0]}_{self.downscaled_grid_shape[1]}',
            'train' if self.split == 'train' else 'test')
        if not os.path.exists(self.fpath_header) or force_download:
            self.download(train_ratio)

        self.gt_fpath = './data/WildtrackBBOX/gt.txt'
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        self.fpath_by_index = []
        frame_folders = ['frame1800', 'frame1805'] if self.split == 'val' else sorted(os.listdir(self.fpath_header))
        for frame_folder in frame_folders:
            for pos_folder in sorted(os.listdir(os.path.join(self.fpath_header, frame_folder))):
                self.fpath_by_index.append(os.path.join(frame_folder, pos_folder))

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                grid_x, grid_y = get_worldgrid_from_posid(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, train_ratio):
        if self.split == 'train':
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        img_fpaths = {cam: {} for cam in range(self.num_cam)}

        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
            pass

        positive_frame_pos_by_index = []
        positive_pos_by_frame = {}
        positive_pid_by_frame = {}
        negative_frame_pos_by_index = []
        all_frame_pos_by_index = []
        all_pids = {}
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                pos_s, pid_s = [], []
                for single_pedestrian in all_pedestrians:
                    pid = single_pedestrian['personID']
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids) + 1
                    pid = all_pids[pid]
                    quantized_pos = pos_quantization(single_pedestrian['positionID'], self.downscale)
                    positive_frame_pos_by_index.append({'frame': frame,
                                                        'positionID': quantized_pos,
                                                        'personID': pid})
                    pos_s.append(quantized_pos)
                    pid_s.append(pid)
                positive_pos_by_frame[frame] = pos_s
                positive_pid_by_frame[frame] = pid_s
        if self.split == 'train':
            for i in range(len(positive_frame_pos_by_index)):
                frame, pos, pid = positive_frame_pos_by_index[i]['frame'], \
                                  positive_frame_pos_by_index[i]['positionID'], 0
                for j in range(self.np_ratio):
                    reduced_pos = np.random.randint(0, np.prod(self.downscaled_grid_shape))
                    pos = pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    while pos in positive_pos_by_frame[frame]:
                        reduced_pos = np.random.randint(0, np.prod(self.downscaled_grid_shape))
                        pos = pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    negative_frame_pos_by_index.append({'frame': frame,
                                                        'positionID': pos,
                                                        'personID': pid})
            all_frame_pos_by_index = positive_frame_pos_by_index + negative_frame_pos_by_index
        else:
            for frame in positive_pos_by_frame.keys():
                for reduced_pos in range(int(np.prod(self.downscaled_grid_shape))):
                    pos = pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    if pos in positive_pos_by_frame[frame]:
                        pid = positive_pid_by_frame[frame][positive_pos_by_frame[frame].index(pos)]
                    else:
                        pid = 0
                    all_frame_pos_by_index.append({'frame': frame, 'positionID': pos, 'personID': pid})

        bbox_by_pos_cam = read_pom(os.path.join(self.root, 'rectangles.pom'))

        os.makedirs(self.fpath_header, exist_ok=True)
        for index in tqdm(range(len(all_frame_pos_by_index))):
            frame, pos, pid = all_frame_pos_by_index[index]['frame'], \
                              all_frame_pos_by_index[index]['positionID'], \
                              all_frame_pos_by_index[index]['personID']
            os.makedirs(os.path.join(self.fpath_header, 'frame{:04d}/pos{:06d}_pid{:03d}'.format(frame, pos, pid)),
                        exist_ok=True)
            for cam in range(self.num_cam):
                if bbox_by_pos_cam[pos][cam] is not None:
                    img_fpath = img_fpaths[cam][frame]
                    img = Image.open(img_fpath).convert('RGB')
                    left, top, right, bottom = bbox_by_pos_cam[pos][cam]
                    img = img.crop([left, top, right, bottom])
                else:
                    img = Image.fromarray(np.zeros([256, 128, 3], np.uint8))
                    pass
                fname = self.fpath_header + '/frame{:04d}/pos{:06d}_pid{:03d}/cam{:d}.jpg'.format(frame, pos, pid, cam)
                img.save(fname)

    def __getitem__(self, index, visualize=False):
        imgs = []
        for cam in range(self.num_cam):
            fname = os.path.join(self.fpath_header, self.fpath_by_index[index], f'cam{cam}.jpg')
            img = Image.open(fname)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.cat(imgs)

        frame_pos_pid_pattern = re.compile(r'frame(\d+)/pos(\d+)_pid(\d+)')
        frame, pos, pid = map(int, frame_pos_pid_pattern.search(self.fpath_by_index[index]).groups())
        grid_x, grid_y = get_worldgrid_from_posid(pos)
        target = pid if self.reID else int(pid > 0)
        return imgs, target, (frame, pid, grid_x, grid_y)

    def __len__(self):
        return len(self.fpath_by_index)


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


def pos_quantization(pos, downscale):
    og_x, og_y = get_worldgrid_from_posid(pos)
    quantized_x, quantized_y = int((og_x // downscale + 0.5) * downscale), \
                               int((og_y // downscale + 0.5) * downscale)
    quantized_pos = get_posid_from_worldgrid([quantized_x, quantized_y])
    return quantized_pos


def pos_reduced_to_og(reduced_pos, downscale, reduced_grid_shape):
    reduced_x = reduced_pos % reduced_grid_shape[0]
    reduced_y = reduced_pos // reduced_grid_shape[0]
    quantized_x, quantized_y = int((reduced_x + 0.5) * downscale), \
                               int((reduced_y + 0.5) * downscale)
    quantized_pos = get_posid_from_worldgrid([quantized_x, quantized_y])
    return quantized_pos


def test():
    # read_pom(os.path.expanduser('~/Data/Wildtrack/rectangles.pom'))
    dataset = WildtrackBBOX(os.path.expanduser('~/Data/Wildtrack'), split='test', )  # force_download=True
    imgs, gt, _ = dataset.__getitem__(1, True)
    pass


if __name__ == '__main__':
    test()
