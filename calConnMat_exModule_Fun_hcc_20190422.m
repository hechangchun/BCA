clear
clc
%% calculate the FNC in Module for each subject 
path = 'F:\BrainAging\SDSU\FNC_rePro\Module_2514\ASD'; % import the data from M = 2514 
outPath = 'F:\BrainAging\SDSU\FNC_rePro\mat_ave_20190424';
load('F:\BrainAging\partition_2_2514.mat');
M = 20; % number of modules
mat_fun_ASD_ex_m = zeros(M,(M*(M-1))/2); %change name
temp = dir([path,'\','ROICorrelation_FisherZ_*.mat']);
for i = 1 : length(temp)
    FNC_inM_mat = importdata([path,'\',temp(i).name]);%import matrixmat_fun_TD_ex_mmat_fun_TD_ex_m
    FNC_inM_mat(isinf(FNC_inM_mat)) = 0;
    FNC_inM_mat(isnan(FNC_inM_mat)) = 0;
    count = 1;
    conVal = 0;
    for j = 1 : M-1
        ind_m = modules_20_60{M,j};
        ind_sort_m = sort(ind_m);
        for s = j+1 : M
            ind_n = modules_20_60{M,s};
            ind_sort_n = sort(ind_n);
            conVal = (sum(sum(abs(FNC_inM_mat(ind_sort_m(:,1),ind_sort_n(:,1))))))/(length(ind_m) * length(ind_n));
            mat_fun_ASD_ex_m(i,count) = conVal;
            count = count + 1;
            conVal = 0;
        end
        %change name      
    end
end
save([outPath,'\','mat_fun_ASD_ex_new_',num2str(M)], 'mat_fun_ASD_ex_m');%change name