%% Fetching data
if ~exist('gt/DukeMTMCT/trainval.mat','file')
    fprintf('Downloading ground truth...\n');
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/ground_truth/trainval.mat';
    if ~exist('gt','dir'), mkdir('gt'); end
    if ~exist('gt/DukeMTMCT','dir'), mkdir('gt/DukeMTMCT'); end
    filename = 'gt/DukeMTMCT/trainval.mat';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
end
if ~exist('res/DukeMTMCT/BIPCC/baseline.txt','file')
    fprintf('Downloading baseline tracker output...\n');
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/tracker_output.zip';
    if ~exist('res','dir'), mkdir('res'); end
    if ~exist('res/DukeMTMCT','dir'), mkdir('res/DukeMTMCT'); end
    if ~exist('res/DukeMTMCT/BIPCC','dir'), mkdir('res/DukeMTMCT/BIPCC'); end
    filename = 'res/DukeMTMCT/BIPCC/tracker_output.zip';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
    unzip(outfilename,'res/DukeMTMCT/BIPCC/');
    delete(filename);
    % Convert to motchallenge format: Frame, ID, left, top, right, bottom,
    % worldX, worldY
    output = dlmread('res/DukeMTMCT/BIPCC/baseline.txt');
    names = {'trainval_mini', 'trainval', 'test_easy', 'test_hard'};
    testIntervals = {127720:187540, 47720:227540, 263504:356648, 227541:263503};
    startTimes = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];

    for cam = 1:8
        for i = 1:length(names)
            filter = output(:,1)==cam & ismember(output(:,3) + startTimes(cam) - 1, testIntervals{i});
            data = output(filter,:);
            data = data(:,2:end);
            data(:,[1 2]) = data(:,[2 1]);
            dlmwrite(sprintf('res/DukeMTMCT/BIPCC/cam%d_%s.txt',cam, names{i}), data, 'delimiter', ',','precision',6);
        end
    end
end

%% Evaluation
[allMets, metsBenchmark, metsMultiCam] = evaluateTracking('DukeMTMCT-trainval-mini.txt', 'res/DukeMTMCT/BIPCC/', 'gt/DukeMTMCT', 'DukeMTMCT');
