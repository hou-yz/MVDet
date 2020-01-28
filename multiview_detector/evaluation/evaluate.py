# coding=utf-8
import os
import argparse
from multiview_det.annotation_headers import DATASET_DIR
import json
EVAL_DATAPATH = DATASET_DIR + "tmp/"


def load(fpath=EVAL_DATAPATH):
    os.system('matlab -nosplash -nodesktop -r \"benchmarkDir=\'' + fpath \
              + '\'; cd ./multiview_det/evaluation/motchallenge-devkit; demo_evalMultiview; quit;\"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--path', help='Path of result and GT files', type=str, default=EVAL_DATAPATH)
    args = parser.parse_args()

    load(args.path)
