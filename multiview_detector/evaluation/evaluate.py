import os
import numpy as np
import torch
from multiview_detector.utils.nms import nms


def matlab_eval(res_fpath, gt_fpath):
    os.system(f'matlab -nosplash -nodesktop -nojvm -r \"'
              f'cd ./multiview_detector/evaluation/motchallenge-devkit; '
              f'try evaluateDetection(\'{res_fpath}\',\'{gt_fpath}\'); catch; end; quit\"')


if __name__ == "__main__":
    res_fpath = '/home/houyz/Code/multiview_one_stage/logs/wildtrack_bbox/2020-01-29_20-34-15/test_nms.txt'
    gt_fpath = '/home/houyz/Code/multiview_one_stage/data/WildtrackBBOX/gt.txt'
    all_res_list = np.loadtxt(os.path.dirname(res_fpath) + '/all_res.txt')
    res_list = []
    for frame in np.unique(all_res_list[:, 0]):
        res = all_res_list[all_res_list[:, 0] == frame, :]
        positions, scores = torch.from_numpy(res[:, 1:3]).float(), torch.from_numpy(res[:, 3]).float()
        ids, count = nms(positions, scores, 25, 60)
        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
    res_list = torch.cat(res_list, dim=0).numpy()
    np.savetxt(res_fpath, res_list, '%d')

    matlab_eval(res_fpath, gt_fpath)
