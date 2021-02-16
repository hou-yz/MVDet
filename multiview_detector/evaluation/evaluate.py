import numpy as np
import matlab.engine
from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py


def matlab_eval(res_fpath, gt_fpath, dataset='wildtrack'):
    eng = matlab.engine.start_matlab()
    eng.cd('multiview_detector/evaluation/motchallenge-devkit')
    res = eng.evaluateDetection(res_fpath, gt_fpath, dataset)
    recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
    return recall, precision, moda, modp


def python_eval(res_fpath, gt_fpath, dataset='wildtrack'):
    MODP, MODA, recall, precision = evaluateDetection_py(res_fpath, gt_fpath, dataset)
    return recall, precision, MODP, MODA


if __name__ == "__main__":
    import os
    import torch
    from multiview_detector.utils.nms import nms

    res_fpath = 'test.txt'
    gt_fpath = 'multiview_detector/evaluation/motchallenge-devkit/gt.txt'

    # test nms
    all_res_list = np.loadtxt(os.path.dirname(res_fpath) + '/all_res.txt')
    res_list = []
    for frame in np.unique(all_res_list[:, 0]):
        res = all_res_list[all_res_list[:, 0] == frame, :]
        positions, scores = torch.from_numpy(res[:, 1:3]).float(), torch.from_numpy(res[:, 3]).float()
        ids, count = nms(positions, scores, 20, np.inf)
        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
    res_list = torch.cat(res_list, dim=0).numpy()
    np.savetxt(res_fpath, res_list, '%d')

    # only consider cam1
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    from multiview_detector.datasets import Wildtrack

    dataset = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    world_grids = res_list[:, 1:3]
    world_coords = dataset.get_worldcoord_from_worldgrid(world_grids.T)
    image_coords = get_imagecoord_from_worldcoord(world_coords, dataset.intrinsic_matrices[0],
                                                  dataset.extrinsic_matrices[0])
    indices = np.logical_and(np.logical_and(image_coords[0, :] > 0, image_coords[0, :] < dataset.img_shape[0]),
                             np.logical_and(image_coords[1, :] > 0, image_coords[1, :] < dataset.img_shape[1]))
    cam1_res_list = res_list[indices, :]
    np.savetxt(res_fpath, cam1_res_list, '%d')

    matlab_eval(res_fpath, gt_fpath, 'Wildtrack')
