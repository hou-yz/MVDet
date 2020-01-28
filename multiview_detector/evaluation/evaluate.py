# coding=utf-8
import os
import argparse
import json


def matlab_eval(res_fpath, gt_fpath):
    os.system(f'matlab -nosplash -nodesktop -nojvm -r \"'
              f'cd ./multiview_detector/evaluation/motchallenge-devkit; '
              f'try evaluateDetection(\'{res_fpath}\',\'{gt_fpath}\'); catch; end; quit\"')


if __name__ == "__main__":
    matlab_eval('/home/houyz/Code/multiview_one_stage/logs/wildtrack_bbox/2020-01-27_15-34-32/val.txt',
         '/home/houyz/Code/multiview_one_stage/data/WildtrackBBOX/gt.txt')
