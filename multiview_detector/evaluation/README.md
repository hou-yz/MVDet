# Evaluation for MultiView detection

## Preparation

First, you have to install matlab.

Then, please follow the installation guide for [matlab-engine](https://au.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html).

e.g. 

```shell script
cd /usr/local/MATLAB/R2019a/extern/engines/python
python setup.py build --build-base="/home/houyz/matlab/" install --prefix="/home/houyz/miniconda3"
```

## Demo

First, after the installation of matlab, you can run the code ```motchallenge-devkit/eval_demo.m```.

Then, once the set up of matlab-engine is finished, you can run the following
```shell script
cd ../.. # this should bring you to the code root folder
python 
```

## File format

ground truth file: ```motchallenge-devkit/gt.txt```

detection result file: ```motchallenge-devkit/test.txt```

    - The first column in ground truth / detection file should be frame number
    - The second and third column should be x and y coordinate

## Alternative Tools
A Python version of the official MATLAB API is provided in ```multiview_detector/pyeval```

