clear
clc
%% calculate the FNC in Module for each subject 
path = 'F:\BrainAging\SDSU\FNC_rePro\Module_2514\TD'; % import the data from M = 2514 
outPath = 'F:\BrainAging\SDSU\FNC_rePro\mat_ave_20190424';
load('F:\BrainAging\partition_2_2514.mat');
M = 20; % number of modules
mat_fun_TD_in_m = zeros(M); %change name
temp = dir([path,'\','ROICorrelation_FisherZ_*.mat']);
for i = 1 : length(temp)
    FNC_inM_mat = importdata([path,'\',temp(i).name]);%import matrix
    FNC_inM_mat(isinf(FNC_inM_mat)) = 0;
    FNC_inM_mat(isnan(FNC_inM_mat)) = 0;
    for j = 1 : M
        ind_m = modules_20_60{M,j};
        ind_sort_m = sort(ind_m);
        conVal = 0;
        for s = 1 : length(ind_sort_m)-1
            conVal = conVal + sum(abs(FNC_inM_mat(ind_sort_m(s,1),ind_sort_m(s+1:end,1))));% using the absolute value          
        end
        len = length(ind_sort_m);
        conVal_av = conVal/((len*(len-1))/2);
        mat_fun_TD_in_m(i,j) = conVal_av; %change name
    end
end
save([outPath,'\','mat_fun_TD_in_new_',num2str(M)], 'mat_fun_TD_in_m');%change name