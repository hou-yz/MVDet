#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.01 --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.02 --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.05 --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.1 --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.2 --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --lr 0.5 --alpha 0