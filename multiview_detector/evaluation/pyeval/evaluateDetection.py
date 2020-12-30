### python version of matlab evaluation toolkit
import numpy as np
from multiview_detector.evaluation.pyeval.CLEAR_MOD_HUN import CLEAR_MOD_HUN

def evaluateDetection_py(res_fpath, gt_fpath, dataset_name):
    """
    This is simply the python translation of a MATLAB　Evaluation tool used to evaluate detection result created by P. Dollar.
    Translated by Zicheng Duan

    The purpose of this API:
    1. To allow the project to run purely in Python without using MATLAB Engine.

    Some critical information to notice before you use this API:
    1.. This API is only tested and deployed in this project: MVDet https://github.com/hou-yz/MVDet, might not be compatible with other projects.
    2. The detection result using this API is a little bit lower (approximately 0~2% decrease in MODA, MODP) than that using MATLAB evaluation tool,
        the reason might be that the Hungarian Algorithm implemented in sklearn.utils.linear_assignment_.linear_assignment is a little bit different with the
        one implemented by P. Dollar, hence leading to different results.
        Therefore, please use the official MATLAB API if you want to obtain the same result shown in the paper. This Python API is only used for convenience.
    3. The training process would not be affected by this API.

    @param res_fpath: detection result file path
    @param gt_fpath: ground truth result file path
    @param dataset: dataset name，should be "WildTrack" or "MultiviewX"
    @return: MODP, MODA, recall, precision
    """

    filename = res_fpath.split("/")
    splitStrLong = ""
    if "train" in filename[-1]:
        splitStrLong = 'Training Set'
        if dataset_name == "Wildtrack":
            start = 0
            steps = 5
            frames = 1795
        elif dataset_name == "MultiviewX":
            start = 0
            steps = 1
            frames = 359

    if "test" in filename[-1]:
        splitStrLong = 'Testing Set'
        if dataset_name == "Wildtrack":
            start = 1800
            steps = 5
            frames = 1995
        elif dataset_name == "MultiviewX":
            start = 360
            steps = 1
            frames = 399

    print("Using dataset %s" % dataset_name)
    print("Set: %s" % splitStrLong)

    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0
    if detRaw is None or detRaw.shape[0] == 0:
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    for t in range(start, frames + 1, steps):
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr),axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr),axis=0)
        frame_ctr += 1
    MODP, MODA, recall, precision = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix)
    return MODP, MODA, recall, precision


if __name__ == "__main__":
    res_fpath = "~/ProjectFolder/test.txt"
    gt_fpath = "~/Data/Wildtrack/gt.txt"
    dataset_name = "Wildtrack"
    evaluateDetection_py(res_fpath, gt_fpath, dataset_name)