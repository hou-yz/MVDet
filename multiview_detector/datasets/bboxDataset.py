import os
#
# os.environ['OMP_NUM_THREADS'] = '1'
# import sys
#
# sys.path.insert(0, '/home/houyz/Code/multiview_one_stage')
import multiprocessing
import re
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
import torchvision.transforms as T
from multiview_detector.utils.projection import *


class bboxDataset(VisionDataset):
    def __init__(self, base, split='train', transform=T.Compose([T.Resize([256, 128]), T.ToTensor()]),
                 reID=False, train_ratio=0.9, downscale=8, np_ratio=3, force_download=False):
        super().__init__(base.root, transform=transform)

        self.split, self.reID, self.downscale, self.np_ratio = split, reID, downscale, np_ratio
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.downscaled_grid_shape = list(map(lambda x: int(x / self.downscale), base.worldgrid_shape))

        self.fpath_header = os.path.join(
            f'./data/{base.__name__}BBox/{self.downscaled_grid_shape[0]}_{self.downscaled_grid_shape[1]}',
            'train' if self.split == 'train' else 'test')
        if not os.path.exists(self.fpath_header) or force_download:
            self.download(train_ratio)

        self.gt_fpath = f'./data/{base.__name__}BBox/gt.txt'
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        self.fpath_by_index = []
        frame_folders = sorted(os.listdir(self.fpath_header))
        if self.split == 'val':
            frame_folders = frame_folders[:2]
        for frame_folder in frame_folders:
            for pos_folder in sorted(os.listdir(os.path.join(self.fpath_header, frame_folder))):
                self.fpath_by_index.append(os.path.join(frame_folder, pos_folder))
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, train_ratio):
        if self.split == 'train':
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        img_fpaths = self.base.get_image_fpaths(frame_range)

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
                    quantized_pos = self.pos_quantization(single_pedestrian['positionID'], self.downscale)
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
                    pos = self.pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    while pos in positive_pos_by_frame[frame]:
                        reduced_pos = np.random.randint(0, np.prod(self.downscaled_grid_shape))
                        pos = self.pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    negative_frame_pos_by_index.append({'frame': frame,
                                                        'positionID': pos,
                                                        'personID': pid})
            all_frame_pos_by_index = positive_frame_pos_by_index + negative_frame_pos_by_index
        else:
            for frame in positive_pos_by_frame.keys():
                for reduced_pos in range(int(np.prod(self.downscaled_grid_shape))):
                    pos = self.pos_reduced_to_og(reduced_pos, self.downscale, self.downscaled_grid_shape)
                    if pos in positive_pos_by_frame[frame]:
                        pid = positive_pid_by_frame[frame][positive_pos_by_frame[frame].index(pos)]
                    else:
                        pid = 0
                    all_frame_pos_by_index.append({'frame': frame, 'positionID': pos, 'personID': pid})

        bbox_by_pos_cam = self.base.read_pom()

        t0 = time.time()
        os.makedirs(self.fpath_header, exist_ok=True)
        p = multiprocessing.Pool(16)
        for index in tqdm(range(len(all_frame_pos_by_index))):
            save_imgs_by_index(index, all_frame_pos_by_index, self.fpath_header, self.num_cam, bbox_by_pos_cam,
                               img_fpaths, p)
            pass
        # for index in tqdm(range(len(all_frame_pos_by_index))):
        #     p.apply_async(save_imgs_by_index,
        #                   (index, all_frame_pos_by_index, self.fpath_header, self.num_cam, bbox_by_pos_cam, img_fpaths))
        p.close()
        p.join()  # Wait for all child processes to close.

        print(time.time() - t0)

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
        grid_x, grid_y = self.base.get_worldgrid_from_pos(pos)
        target = pid if self.reID else int(pid > 0)
        return imgs, target, (frame, pid, grid_x, grid_y)

    def __len__(self):
        return len(self.fpath_by_index)

    def pos_quantization(self, pos, downscale):
        og_x, og_y = self.base.get_worldgrid_from_pos(pos)
        quantized_x, quantized_y = int((og_x // downscale) * downscale), \
                                   int((og_y // downscale) * downscale)
        quantized_pos = self.base.get_pos_from_worldgrid([quantized_x, quantized_y])
        return quantized_pos

    def pos_reduced_to_og(self, reduced_pos, downscale, reduced_grid_shape):
        if self.base.indexing == 'xy':
            reduced_x = reduced_pos % reduced_grid_shape[1]
            reduced_y = reduced_pos // reduced_grid_shape[1]
        else:
            reduced_x = reduced_pos % reduced_grid_shape[0]
            reduced_y = reduced_pos // reduced_grid_shape[0]
        quantized_x, quantized_y = int((reduced_x) * downscale), \
                                   int((reduced_y) * downscale)
        quantized_pos = self.base.get_pos_from_worldgrid([quantized_x, quantized_y])
        return quantized_pos


def save_img_by_cam(bbox, fname, img_fpath):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if bbox is not None:
        img = Image.open(img_fpath).convert('RGB')
        left, top, right, bottom = bbox
        img = img.crop([left, top, right, bottom])
    else:
        img = Image.fromarray(np.zeros([1, 1, 3], np.uint8))
        pass
    img.save(fname)


def save_imgs_by_index(index, all_frame_pos_by_index, fpath_header, num_cam, bbox_by_pos_cam, img_fpaths, p):
    frame, pos, pid = all_frame_pos_by_index[index]['frame'], \
                      all_frame_pos_by_index[index]['positionID'], \
                      all_frame_pos_by_index[index]['personID']
    os.makedirs(os.path.join(fpath_header, 'frame{:04d}'.format(frame)), exist_ok=True)
    for cam in range(num_cam):
        bbox = bbox_by_pos_cam[pos][cam]
        img_fpath = img_fpaths[cam][frame]
        fname = fpath_header + \
                '/frame{:04d}/pos{:06d}_pid{:03d}/cam{:d}.jpg'.format(frame, pos, pid, cam)
        p.apply_async(save_img_by_cam, (bbox, fname, img_fpath))


def test():
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    dataset = bboxDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='val', force_download=True)  #
    imgs, gt, _ = dataset.__getitem__(1, True)
    pass


if __name__ == '__main__':
    test()
