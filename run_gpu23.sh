#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.01
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.02
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.05
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.1
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.2
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.5