function [allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqmap, resDir, gtDataDir, benchmark)

% Input:
% - seqmap
% Sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
% evaluated in a single run. These files are inside the ./seqmaps folder.
%
% - resDir
% The folder containing the tracking results. Each one should be saved in a
% separate .txt file with the name of the respective sequence (see ./res/data)
%
% - gtDataDir
% The folder containing the ground truth files.
%
% - benchmark
% The name of the benchmark, e.g. 'MOT15', 'MOT16', 'MOT17', 'DukeMTMCT'
%
% Output:
% - allMets
% Scores for each sequence
% 
% - metsBenchmark
% Aggregate score over all sequences
%
% - metsMultiCam
% Scores for multi-camera evaluation

addpath(genpath('.'));
warning off;
fprintf(benchmark) 
% Benchmark specific properties
world = 0;
threshold = 0.5;
multicam = 0;
if strcmp(benchmark, 'MOT15')
elseif strcmp(benchmark, 'MOT15_3D')
    world = 1;
    threshold = 1;
elseif strcmp(benchmark, 'MOT16')
elseif strcmp(benchmark, 'MOT17')
elseif strcmp(benchmark, 'CVPR19')
elseif strcmp(benchmark, 'PETS2017')
elseif strcmp(benchmark, 'DukeMTMCT')
    multicam = 1;
end


% Read sequence list
sequenceListFile = fullfile('seqmaps',seqmap);

allSequences = parseSequences2(sequenceListFile);
fprintf('Sequences: \n');
disp(allSequences')
gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = [];
metsBenchmark = [];
metsMultiCam = [];

for ind = 1:length(allSequences)
    
    % Parse ground truth
    if ~multicam
        % MOTX parsing
        sequenceName = char(allSequences(ind));
        sequenceFolder = [gtDataDir, sequenceName, filesep];
        fprintf('\t... %s\n',sequenceName);
        assert(isdir(sequenceFolder), 'Sequence folder %s missing\n', sequenceFolder);
        
        gtFilename = fullfile(gtDataDir,sequenceName,'gt','gt.txt');
        gtdata = dlmread(gtFilename);
        gtdata(gtdata(:,7)==0,:) = [];     % ignore 0-marked GT
        gtdata(gtdata(:,1)<1,:) = [];      % ignore negative frames
        if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17')  || strcmp(benchmark, 'CVPR19') % ignore non-pedestrians
            gtdata(gtdata(:,8)~=1,:) = [];
        end
        
        if strcmp(benchmark, 'MOT15_3D')
            gtdata(:,[7 8]) = gtdata(:,[8 9]); % shift world coordinates
        end
        [~, ~, ic] = unique(gtdata(:,2)); % normalize IDs
        gtdata(:,2) = ic;
        gtMat{ind} = gtdata;
        
    else
        % DukeMTMCT parsing
        ROI = getROIs();
        if strcmp(seqmap,'DukeMTMCT-test-easy.txt')
            load('gt/DukeMTMCT/testData.mat');
            gtdata = testData;
            testInterval = [263504:356648];
        elseif strcmp(seqmap,'DukeMTMCT-test-hard.txt')
            load('gt/DukeMTMCT/testHardData.mat');
            gtdata = testHardData;
            testInterval = [227541:263503];
        elseif strcmp(seqmap,'DukeMTMCT-trainval.txt')
            load('gt/DukeMTMCT/trainval.mat');
            gtdata = trainData;
            testInterval = [47720:227540]; % takes too long
        elseif strcmp(seqmap,'DukeMTMCT-trainval-mini.txt') % shorter version of trainval
            load('gt/DukeMTMCT/trainval.mat');
            gtdata = trainData;
            testInterval = [127720:187540];
        else
            fprintf('Unknown test set %s\n',testSet);
            return;
        end
        
        cam = ind;
        startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
        filter = gtdata(:,1) == cam & ismember(gtdata(:,3) + startTimes(cam) - 1, testInterval);
        gtdata = gtdata(filter,:);
        gtdata = gtdata(:,2:end);
        gtdata(:,[1 2]) = gtdata(:,[2 1]);
        gtdata(:,1) = gtdata(:,1) + startTimes(cam) - testInterval(1); % normalize frames
        gtdata = sortrows(gtdata,[1 2]);
        gtMat{ind} = gtdata;
        
    end
    
    % Parse result
    if ~multicam
        % MOTX data format
     
        resFilename = [resDir, sequenceName,  '.txt'];
        if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17') || strcmp(benchmark, 'CVPR19')
            resFilename = preprocessResult(resFilename, sequenceName, gtDataDir);
        end
        
        % Skip evaluation if output is missing
        if ~exist(resFilename,'file')
            error('Invalid submission. Result for sequence %s not available!\n',sequenceName);
        end
        
        % Read result file
        if exist(resFilename,'file')
            s = dir(resFilename);
            if s.bytes ~= 0
                resdata = dlmread(resFilename);
            else
                resdata = zeros(0,9);
            end
        else
            error('Invalid submission. Result file for sequence %s is missing or invalid\n', resFilename);
        end
        resdata(resdata(:,1)<1,:) = [];      % ignore negative frames
        if strcmp(benchmark, 'MOT15_3D')
            resdata(:,[7 8]) = resdata(:,[8 9]);  % shift world coordinates
        end
        resdata(resdata(:,1) > max(gtMat{ind}(:,1)),:) = []; % clip result to gtMaxFrame
        resMat{ind} = resdata;
        
    else
        % Duke data format
        sequenceName = allSequences{ind};
        resFilename = [resDir, sequenceName,  '.txt'];
        s = dir(resFilename);
        if exist(resFilename,'file') && s.bytes ~= 0
            resdata = dlmread(resFilename);
        else
            resdata = zeros(0,9);
        end
        cam = ind;
        
        % Filter rows by frame interval
        startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
        resdata(~ismember(resdata(:,1) + startTimes(cam) - 1, testInterval),:) = [];
        
        % Filter rows by feet position within ROI
        feetpos = [ resdata(:,3) + 0.5*resdata(:,5), resdata(:,4) + resdata(:,6)];
        resdata = resdata(inpolygon(feetpos(:,1),feetpos(:,2), ROI{cam}(:,1),ROI{cam}(:,2)),:);
        resdata(:,1) = resdata(:,1) + startTimes(cam) - testInterval(1); % normalize frames
        resdata = sortrows(resdata,[1 2]);
        resMat{ind} = resdata;
        
        
    end
    
    % Sanity check
    frameIdPairs = resMat{ind}(:,1:2);
    [u,I,~] = unique(frameIdPairs, 'rows', 'first');
    hasDuplicates = size(u,1) < size(frameIdPairs,1);
    if hasDuplicates
        ixDupRows = setdiff(1:size(frameIdPairs,1), I);
        dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
        rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));
        
        errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in sequence %s.\nInstance:\n', sequenceName);
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(1),:)), sprintf('\n')];
        errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(2),:)), sprintf('\n')];
        assert(~hasDuplicates, errorMessage);
    end
    
    % Evaluate sequence
    [metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtMat{ind}, resMat{ind}, threshold, world);
    metsID = IDmeasures(gtMat{ind}, resMat{ind}, threshold, world);
    mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
    allMets(ind).name = sequenceName;
    allMets(ind).m    = mets;
    allMets(ind).IDmeasures = metsID;
    allMets(ind).additionalInfo = additionalInfo;
    fprintf('%s\n', sequenceName); printMetrics(mets); fprintf('\n');
    evalFile = fullfile(resDir, sprintf('eval_%s.txt',sequenceName));
    dlmwrite(evalFile, mets);
    
end

% Overall scores
metsBenchmark = evaluateBenchmark(allMets, world);
fprintf('\n');
fprintf(' ********************* Your %s Results *********************\n', benchmark);
printMetrics(metsBenchmark);
evalFile = fullfile(resDir, 'eval.txt');
dlmwrite(evalFile, metsBenchmark);

% Multicam scores
if multicam
    
    metsMultiCam = evaluateMultiCam(gtMat, resMat, threshold, world);
    fprintf('\n');
    fprintf(' ********************* Your %s MultiCam Results *********************\n', benchmark);
    fprintf('IDF1   IDP    IDR\n');
    fprintf('%.2f  %.2f  %.2f\n', metsMultiCam(1), metsMultiCam(2), metsMultiCam(3));
    
    evalFile = fullfile(resDir, 'eval_mc.txt');
    dlmwrite(evalFile, metsMultiCam);
    
end