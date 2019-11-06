clear
clc
mat = [];
num = 1;
for i = 1 : length(data_TD)
    [r, p] = partialcorr(data_TD(:,i),cov_NYU_SDSU_TD_site_sex_FIQ_headM(:,5))
    if p < 0.001
        mat(1,num) = i;
        num = num + 1;
    end
end