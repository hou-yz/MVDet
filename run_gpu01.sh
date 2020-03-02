#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant no_joint_conv
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant img_proj
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant res_proj
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 0
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant no_joint_conv --alpha 0
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant img_proj --alpha 0
#CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --variant res_proj --alpha 0
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 0.1
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 0.2
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 0.5
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 1
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 2
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack --alpha 5