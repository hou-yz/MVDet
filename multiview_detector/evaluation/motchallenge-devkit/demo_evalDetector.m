addpath(genpath('.'));
benchmarkGtDir = '../data/MOT17Det/train/';
allMets = evaluateDetection('c9-train.txt', 'res/MOT17Det/DPM/data', benchmarkGtDir);