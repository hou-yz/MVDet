## Python Evaluation Tool for MVDet

This is simply the Python translation of a MATLAB Evaluation tool used to evaluate detection result created by P. Dollar.  
Translated by [Zicheng Duan](https://github.com/ZichengDuan).  

### Purpose
   Allowing the project to run purely in Python without using MATLAB Engine.  
   

### Critical information before usage
   1. This API is only tested and deployed in this project: [hou-yz/MVDet](https://github.com/hou-yz/MVDet), might not be compatible with other projects.
   2. The detection result using this API **is a little bit lower** (approximately 0~2% decrease in MODA, MODP) than that when using official MATLAB evaluation tool, the reason might be the Hungarian Algorithm implemented in sklearn is a little bit different from the one implemented by P. Dollar, hence resulting in different results.   
   Therefore, **please use the official MATLAB API if you want to obtain the same evaluation result shown in the paper**. This Python API is only used for convenience.
   3. The training process would not be affected by this API.

### Usage
Please go to ```test()``` function in ```trainer.py``` for more details.  

```
recall, precision, moda, modp = matlab_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.base.__name__)

# If you want to use the unofiicial python evaluation tool for convenient purposes.
# recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
#                                             data_loader.dataset.base.__name__)
```
