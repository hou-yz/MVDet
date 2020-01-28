function detResults=evaluateDetection(seqmap,resDir,dataDir, detector)
%% evaluate detections using P. Dollar's script

if nargin<4, detector='Multi-View Detection'; end
chlname = 'WildTrack';

[~,f,~]=fileparts(seqmap);
splitStr = f;
splitStrLong = 'unknown split';
if strcmpi(splitStr,'test'), splitStrLong='Test Set';
elseif strcmpi(splitStr,'train'), splitStrLong='Training Set';
end

addpath(genpath('.'));
    

% read sequence map
seqmapFile=fullfile('seqmaps',seqmap);
allSeq = parseSequences2(seqmapFile);

fprintf('Challenge: %s\n',chlname);
fprintf('Set: %s\n',splitStrLong);
fprintf('Sequences: \n');
disp(allSeq')


        
gtInfo=[];
gtInfo.X=[];
allFgt=zeros(1,length(allSeq));


cls = [2,7,8,12]; %% ambiguous classes
minvis = 0.5;
ref=0:.025:1;
ref=0:.1:1;
showEachRef=1;


% Find out the length of each sequence
% and concatenate ground truth
gtInfoSingle=[];
gtAll={};
detInfoSingle=[];
detAll={};
seqCnt=0;
allFrCnt=0;
evalMethod=1;
gtAllMatrix=zeros(0,2);
detAllMatrix=zeros(0,2);
for s=allSeq
    seqCnt=seqCnt+1;
    seqName = char(s);
    [seqName, seqFolder, imgFolder, imgExt, F, dirImages] ...
        = getSeqInfoFromFile(seqName, dataDir);
    
    assert(isdir(seqFolder),'Sequence folder %s missing',seqFolder);
    gtFile = fullfile(dataDir,seqName,'gt','gt.csv');
    %gtRaw = dlmread(gtFile);
    gtRaw=readtable(gtFile);
    gtRaw=gtRaw{:,:};

    % if something (a result) is missing, we cannot evaluate this tracker
    resFile = fullfile(resDir,[seqName '.csv']);
    if ~exist(resFile,'file')
        fprintf('WARNING: result for %s not available: %s\n',seqName, resFile);
        evalMethod=0;
        continue;
    end
    % set visibility threshold to 25% for cvpr 19 
    if ~isempty(strfind(seqFolder,'CVPR19'))
        minvis = 0.25;
        fprintf('Min Vis changed: \t %s', minvis) ;
    end
            
    % if MOT16, MOT17 or CVPR19 preprocess (clean)
    if cleanRequired(seqFolder)
        resFile = preprocessResult(resFile, seqName, dataDir, 1, minvis);
    end
        
    % detRaw=dlmread(resFile);
    detRaw=readtable(resFile);
    detRaw=detRaw{:,:};
    
    % 
    gtOne= {};
    detOne = {};
    for t=1800:5:1995
        allFrCnt=allFrCnt+1;
        
        % keep pedestrians only and vis >= minvis
        exgt=find(gtRaw(:,1)==t);
        gtAll{allFrCnt}=[gtRaw(exgt,2:3) zeros(length(exgt),1)];
        gtOne{allFrCnt}=[gtRaw(exgt,2:3) zeros(length(exgt),1)];
        
        ng = length(exgt);
        oneFrame=[allFrCnt*ones(ng,1), (1:ng)', gtRaw(exgt,2:3)]; % set IDs to 1..ng
%         oneFrame=[(1:ng)', gtRaw(exgt,2:3)];
        gtAllMatrix=[gtAllMatrix; oneFrame];

        exdet=find(detRaw(:,1)==t);
        bbox=detRaw(exdet,2:3);
        detAll{allFrCnt}=bbox;
        detOne{allFrCnt}=bbox;
        
        ng = length(exdet);
        oneFrame=[allFrCnt*ones(ng,1), (1:ng)', detRaw(exdet,2:3)]; % set IDs to 1..ng
%         oneFrame=[(1:ng)', detRaw(exdet,2:3)];
        detAllMatrix=[detAllMatrix; oneFrame];
        
    end
    

    allFgt(seqCnt) = F;    
    gtInfoSingle(seqCnt).gt=gtOne;
    gtInfoSingle(seqCnt).gtMat=gtRaw;
    detInfoSingle(seqCnt).det = detOne;    
    detInfoSingle(seqCnt).detMat = detRaw;
end

detResults=[];
mcnt=1;



try
    %     detector = char(m);
    fprintf('Evaluating %s\n',detector);
    
    detResults(mcnt).detector = detector;

%     evalMethod=1;
    
    seqCnt=0;
    detectorRuntime=0;
    
    % iterate over each sequence
%     for s=allSeq
%         
%         seqCnt=seqCnt+1;
%         seqName = char(s);
%         
%         fprintf('\t... %s\n',seqName);
%         
%         
%         gt0=gtInfoSingle(seqCnt).gt;
%         dt0=detInfoSingle(seqCnt).det;
% %         [gt,dt]=bbGt('evalRes',gt0,dt0);
% %         [rc,pr,scores,refprcn] = bbGt('compRoc',gt,dt,0,ref);
%         
%         
% %         rc
% %         pr
% %         score
% %         refprcn
%         AP = mean(refprcn);
% %         pause
%         
%         detResults(mcnt).mets(seqCnt).rc=rc;
%         detResults(mcnt).mets(seqCnt).pr=pr;
%         detResults(mcnt).mets(seqCnt).ref=refprcn;
%         detResults(mcnt).mets(seqCnt).AP=AP;
%         detResults(mcnt).mets(seqCnt).name=seqName;
% 
%         gtRawPed = gtInfoSingle(seqCnt).gtMat;
%         detRawPed = detInfoSingle(seqCnt).detMat;
%         [detMets, detMetsInfo, detMetsAddInfo]=CLEAR_MOD_HUN(gtRawPed,detRawPed);
% %         printMetrics(detMets);
%         detResults(mcnt).mets(seqCnt).detMets = detMets;
%         detResults(mcnt).mets(seqCnt).detMetsInfo = detMetsInfo;
%         detResults(mcnt).mets(seqCnt).detMetsAddInfo = detMetsAddInfo;
% 
%         
%         
%         refprstr = '';
%         for r=1:length(refprcn)
%             refprstr=[refprstr,sprintf('%.4f',refprcn(r))];
%             if r<length(refprcn), refprstr=[refprstr,',']; end
%         end
%     end
    
    if evalMethod
        fprintf('Ok, results are valid. EVALUATING...\n');
        
        gt0=gtAll;
        dt0=detAll;
%         [gt,dt]=bbGt('evalRes',gt0,dt0);
%         [rc,pr,scores,refprcn] = bbGt('compRoc',gt,dt,0,ref);    
        
        [detMetsAll, detMetsInfo, detMetsAddInfo]=CLEAR_MOD_HUN(gtAllMatrix,detAllMatrix);

        
%         AP=mean(refprcn);
%         detResults(mcnt).rc=rc;
%         detResults(mcnt).pr=pr;
%         detResults(mcnt).ref=refprcn;
%         detResults(mcnt).AP=AP;
        detResults(mcnt).detMets=detMetsAll;

        
        fprintf('*** Dataset: %s ***\n',chlname);         
        fprintf('Recall:    ')
        for r=1:showEachRef:length(ref)
            fprintf('%6.3f',ref(r)); 
        end
        fprintf('\n')
%         fprintf('Precision: ')
%         for r=1:showEachRef:length(ref)
%             fprintf('%6.3f',refprc12n(r)); 
%         end
        fprintf('\n');
%         fprintf('Average Precision: %.4f\n',AP);
        printMetrics(detMetsAll);

        
%         fprintf('\n\nHere are the per-sequence evaluations:\n\n');
%         seqCnt = 0;
%         mcnt=1;
%         for s=allSeq
%             seqCnt=seqCnt+1;
%             seqName = char(s);
%             
%             refprcn = detResults(mcnt).mets(seqCnt).ref;
%             AP = detResults(mcnt).mets(seqCnt).AP;
%             detMets = detResults(mcnt).mets(seqCnt).detMets;
%             
%             fprintf('\t... %s\n',seqName);
%             fprintf('Recall:    ')
%             for r=1:showEachRef:length(ref)
%                 fprintf('%6.3f',ref(r)); 
%             end
%             fprintf('\n')
%             fprintf('Precision: ')
%             for r=1:showEachRef:length(ref)
%                 fprintf('%6.3f',refprcn(r)); 
%             end
%             fprintf('\n');
%             fprintf('Average Precision: %.4f\n',AP);      
%             printMetrics(detMets);
% 
%             fprintf('\n');
%             
%         end
%         
%         fprintf('\n\n');

    
%         AP=detResults(mcnt).AP;
%         refprcn=detResults(mcnt).ref;
    
%         evalFile = fullfile(resDir, 'eval_dets.txt');
%         dlmwrite(evalFile,AP);
%         evalDetailsFile = fullfile(resDir, 'eval_dets_details.txt');
%         dlmwrite(evalDetailsFile,[rc,pr]);
%         evalRefFile = fullfile(resDir, 'eval_dets_ref.txt');
%         dlmwrite(evalRefFile,[ref',refprcn']);
%         evalMatFile = fullfile(resDir, 'eval_dets.mat');
%         save(evalMatFile,'detResults')
%         
% 
%         
%         
%         % plot
%         fh=figure;
%         figFile = fullfile(resDir, sprintf('rcpr-%s',splitStr));
%         clf; grid on;
%         plot(rc,pr,'linewidth',3);
%         xlabel('Recall');
%         ylabel('Precision');        
%         axis([0,1,0,1]);
%         legend(sprintf('%s (AP: %.2f)',detector,AP));
%         titleStr = sprintf('%s - %s',chlname,splitStrLong);
%         title(titleStr);
%         saveas(fh,[figFile,'.png']);
% %         saveas(fh,[figFile,'.pdf']);
%         
%         close(fh);        

    else
        fprintf('WARNING: %s cannot be evaluated\n',tracker);
        % update mysql, delete row
    end
    
    
catch err
    fprintf('WARNING: %s cannot be evaluated: %s\n',detector,err.message);
    getReport(err) 
end


