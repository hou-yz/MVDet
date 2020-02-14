import os
import numpy as np
import torch
import matlab.engine
from multiview_detector.utils.nms import nms


def matlab_eval(res_fpath, gt_fpath):
    # os.system(f'matlab -nosplash -nodesktop -nojvm -r \"'
    #           f'cd ./multiview_detector/evaluation/motchallenge-devkit; '
    #           f'try evaluateDetection(\'{res_fpath}\',\'{gt_fpath}\'); catch; end; quit\"')
    eng = matlab.engine.start_matlab()
    eng.cd('./multiview_detector/evaluation/motchallenge-devkit')
    res = eng.evaluateDetection(res_fpath, gt_fpath)
    recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
    return recall, precision, moda, modp


if __name__ == "__main__":
    res_fpath = '/home/houyz/Code/multiview_one_stage/logs/wildtrack_frame/2020-02-12_15-38-45/test_nms.txt'
    gt_fpath = '/home/houyz/Code/multiview_one_stage/data/WildtrackBBOX/gt.txt'
    all_res_list = np.loadtxt(os.path.dirname(res_fpath) + '/all_res.txt')
    res_list = []
    for frame in np.unique(all_res_list[:, 0]):
        res = all_res_list[all_res_list[:, 0] == frame, :]
        positions, scores = torch.from_numpy(res[:, 1:3]).float(), torch.from_numpy(res[:, 3]).float()
        ids, count = nms(positions, scores, 25, 50 * 12)
        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
    res_list = torch.cat(res_list, dim=0).numpy()
    np.savetxt(res_fpath, res_list, '%d')

    matlab_eval(res_fpath, gt_fpath)
