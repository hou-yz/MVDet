#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.01 -d multiviewX
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.02 -d multiviewX
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.05 -d multiviewX
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.1 -d multiviewX
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.2 -d multiviewX
CUDA_VISIBLE_DEVICES=2,3 python main.py --lr 0.5 -d multiviewX