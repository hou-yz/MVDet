#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant no_joint_conv
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant img_proj
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant res_proj
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 0
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant no_joint_conv --alpha 0
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant img_proj --alpha 0
#CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --variant res_proj --alpha 0
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 0.1
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 0.2
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 0.5
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 1
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 2
CUDA_VISIBLE_DEVICES=2,3 python main.py -d multiviewx --alpha 5