clear
clc
NYU_fun_TD_ex_20 = importdata('F:\BrainAging\reTest_264\NYU\mat_fun_TD_ex_new_10.mat');
NYU_fun_TD_in_20 = importdata('F:\BrainAging\reTest_264\NYU\mat_fun_TD_in_new_10.mat');
% NYU_stru_TD_ex_20 = importdata('F:\BrainAging\result_new_reProSM_average_20190423\NYU\ASD\mat_str_ASD_ex_new_20.mat');
% NYU_stru_TD_in_20 = importdata('F:\BrainAging\result_new_reProSM_average_20190423\NYU\ASD\mat_str_ASD_in_new_20.mat');
SDSU_fun_TD_ex_20 = importdata('F:\BrainAging\reTest_264\SDSU\mat_fun_TD_ex_new_10.mat');
SDSU_fun_TD_in_20 = importdata('F:\BrainAging\reTest_264\SDSU\mat_fun_TD_in_new_10.mat');
% SDSU_stru_TD_ex_20 = importdata('F:\BrainAging\result_new_reProSM_average_20190423\SDSU\ASD\mat_str_ASD_ex_new_20.mat');
% SDSU_stru_TD_in_20 = importdata('F:\BrainAging\result_new_reProSM_average_20190423\SDSU\ASD\mat_str_ASD_in_new_20.mat');
NYU_TD_age = importdata('F:\BrainAging\result_new_rePro\NYU\TD\NYU_1_TD_age.mat');
SDSU_TD_age = importdata('F:\BrainAging\result_new_rePro\SDSU\TD\SDSU_2_TD_age.mat');
% cov_site = importdata('F:\BrainAging\result_new\cov_site.mat');
% NYU_SDSU_ASD_str_fun_ex_in_20 =  [NYU_stru_TD_ex_20,NYU_stru_TD_in_20,NYU_fun_TD_ex_20,NYU_fun_TD_in_20;SDSU_stru_TD_ex_20,SDSU_stru_TD_in_20,SDSU_fun_TD_ex_20,SDSU_fun_TD_in_20];
NYU_SDSU_TD_fun_264_ex_in_10 =  [NYU_fun_TD_ex_20,NYU_fun_TD_in_20;SDSU_fun_TD_ex_20,SDSU_fun_TD_in_20];
NYU_SDSU_TD_age = [NYU_TD_age;SDSU_TD_age];
fea_mat = [];
num = 0
for i = 1 : size(NYU_SDSU_asd_str_fun_ex_in_20,2)
    [h,p] = partialcorr(NYU_SDSU_asd_str_fun_ex_in_20(:,i),NYU_SDSU_asd_age(:,1),cov_site);
%      [h,p] = corr(NYU_SDSU_asd_str_fun_ex_in_20(:,i),NYU_SDSU_asd_age(:,1));
    if p < 0.00012
        num = num + 1;
        fea_mat(1,num) = i;
        fea_mat(2:58,num) = NYU_SDSU_asd_str_fun_ex_in_20(:,i)
    end
end
save NYU_SDSU_ASD_str_fun_ex_in_20 NYU_SDSU_ASD_str_fun_ex_in_20
save NYU_SDSU_TD_fun_264_ex_in_10 NYU_SDSU_TD_fun_264_ex_in_10
