clear
clc
NYU_mat = importdata('F:\BrainAging\NYU\NYU_1_TD_age.mat');
SDSU_mat = importdata('F:\BrainAging\SDSU\SDSU_2_TD_age.mat');
NYU_SDSU_TD_age = [NYU_mat;SDSU_mat];
save NYU_SDSU_TD_age NYU_SDSU_TD_age