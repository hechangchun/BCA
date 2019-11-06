clear
clc
%% add this code by hcc
clear
clc
%% import ASD data
NYU_SDSU_ASD_str_fun_ex_in_20 = importdata('F:\BrainAging\result_new\res_ASD\NYU_SDSU_asd_str_fun_ex_in_20.mat');
for i = 1 ; size(NYU_SDSU_asd_str_fun_ex_in_20,2)
    mean = ZScore(NYU_SDSU_asd_str_fun_ex_in_20(:,i));
    std = 
end
%% import TD data
NYU_SDSU_TD_str_fun_ex_in_20 = importdata('F:\BrainAging\result_new\res_TD\NYU_SDSU_TD_str_fun_ex_in_20.mat');