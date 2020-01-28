# Evaluation for MultiView detection

## Preparation

Place detection result and ground truth files in the corresponding path as in file demo_evalMultiview.m

e.g. if benchmarkDir = '/datasets/Wildtrack_dataset/tmp/' in demo_evalMultiview.m

ground truth file: gt.csv should be placed at '/datasets/Wildtrack_dataset/tmp/MultiView/gt/gt.csv'

detection result file: MultiView.csv should be placed at '/datasets/Wildtrack_dataset/tmp/MultiView.csv'

ATTENTION: 

    - Case Sensitive.
    - The first column in .csv file should be frame number from 0 to 1995 in step of 5
    - The second and third column should be x and y coordinate.

## Run

Using Matlab to run demo_evalMultiview.m directly, and evaluation result including MODA, MODP, Precision and Recall should be presented.

