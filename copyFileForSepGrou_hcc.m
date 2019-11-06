%% copyfile to divide two group (ASD & TD)
clear
clc
path = ('F:\BrainAging\NYU\FCN_newSM_264\mat_264\all');
load('F:\BrainAging\NYU\NYU_1_ASD_num.mat');
load('F:\BrainAging\NYU\NYU_1_TD_num.mat');
outPath_ASD = ('F:\BrainAging\NYU\FCN_newSM_264\mat_264\ASD\');
outPath_TD = ('F:\BrainAging\NYU\FCN_newSM_264\mat_264\TD\');

% temp = dir([path,'\','wlevel*.mat']);% read for SNC
temp = dir([path,'\','ROICorrelation_FisherZ*.mat']);% read for FNC

for i = 1 : length(temp)
    for j = 1 : length(NYU_1_ASD_num) % change the ID of subject
        if strcmp(temp(i).name(24:end-4),num2str(NYU_1_ASD_num(j,1))) 
            copyfile([path,'\',temp(i).name],outPath_ASD);
        end
    end
    for s = 1 : length(NYU_1_TD_num)
        if strcmp(temp(i).name(24:end-4),num2str(NYU_1_TD_num(s,1))) %FCN:temp(i).name(24:end-4);SCN:temp(i).name(20:end-17)
            copyfile([path,'\',temp(i).name],outPath_TD);
        end
    end   
end