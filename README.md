# Multiview Detection with Feature Perspective Transformation [[Website](https://hou-yz.github.io/publication/2020-eccv2020-mvdet)] [[arXiv](https://arxiv.org/abs/2007.07247)]

```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}
```

Please visit [link](https://github.com/hou-yz/MVDeTr) for our new work MVDeTr, a transformer-powered multiview detector that achieves new state-of-the-art!

## Overview
We release the PyTorch code for **MVDet**, a state-of-the-art multiview pedestrian detector; and **MultiviewX** dataset, a novel synthetic multiview pedestrian detection datatset.

Wildtrack             |  MultiviewX
:-------------------------:|:-------------------------:
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_wildtrack_demo.gif "Detection results on Wildtrack dataset")  |  ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_demo.gif "Detection results on MultiviewX dataset")

 
## Content
- [MultiviewX dataset](#multiviewx-dataset)
    * [Download MultiviewX](#download-multiviewx)
    * [Build your own version](#build-your-own-version)
- [MVDet Code](#mvdet-code)
    * [Dependencies](#dependencies)
    * [Data Preparation](#data-preparation)
    * [Training](#training)



## MultiviewX dataset
Using pedestrian models from [PersonX](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint), in Unity, we build a novel synthetic dataset **MultiviewX**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_dataset.jpg "Visualization of MultiviewX dataset")

MultiviewX dataset covers a square of 16 meters by 25 meters. We quantize the ground plane into a 640x1000 grid. There are 6 cameras with overlapping field-of-view in MultiviewX dataset, each of which outputs a 1080x1920 resolution image. We also generate annotations for 400 frames in MultiviewX at 2 fps (same as Wildtrack). On average, 4.41 cameras are covering the same location. 

### Download MultiviewX
Please refer to this [link](https://1drv.ms/u/s!AtzsQybTubHfgP9BJt2g7R_Ku4X3Pg?e=GFGeVn) for download.

### Build your own version
Please refer to this [repo](https://github.com/hou-yz/MultiviewX) for a detailed guide & toolkits you might need.




## MVDet Code
This repo is dedicated to the code for **MVDet**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")

### Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)

### Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](#multiviewx-dataset) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

### Training
In order to train classifiers, please run the following,
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
``` 
This should automatically return evaluation results similar to the reported 88.2\% MODA on Wildtrack dataset. 

### Pre-trained models
You can download the checkpoints at this [link](https://anu365-my.sharepoint.com/:u:/g/personal/u6852178_anu_edu_au/Edhf_qajGMZLvlh9o6kByeUBxo_4E6DVjiQR2mrpGFtPjA?e=qEgiWR).
