function cl=cleanRequired(seqFolder)

cl =~isempty(strfind(seqFolder,'CVPR19')) || ~isempty(strfind(seqFolder,'MOT16')) || ~isempty(strfind(seqFolder,'MOT17'));
