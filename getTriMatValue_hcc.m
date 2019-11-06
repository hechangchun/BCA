clear
clc
%% calculate the FNC in Module for each subject 
path = 'F:\BrainAging\NYU\FCN_newSM\Module_2514\ASD'; % import the data from M = 2514 
outPath = 'C:\Users\wolf\Desktop\cacheFile\test';
temp = dir([path,'\','ROICorrelation_FisherZ_*.mat']);
M = 2514; % number of modules
mat_fun_ASD_in_m = zeros(length(temp),M*(M-1)/2); %change name

for i = 1 : length(temp)
    FNC_inM_mat = importdata([path,'\',temp(i).name]);%import matrix
    FNC_inM_mat(isinf(FNC_inM_mat)) = 0;
    FNC_inM_mat(isnan(FNC_inM_mat)) = 0;
    sum = 1;
    for j = 1 : M
        for t = j+1:M
           mat_fun_ASD_in_m(i,sum) = FNC_inM_mat(j,t);
           sum = sum + 1;
        end
    end
end
save([outPath,'\','NYU_mat_fun_ASD_in_new_',num2str(M)], 'NYU_mat_fun_ASD_in_m');%change name