clear
clc
%connect_count
path = 'F:\BrainAging\SDSU\SNC_deform\module_2514\TD'; %import data from M = 2514
outPath = 'F:\BrainAging\SDSU\SNC_deform\mat_ave_20190424';
load('F:\BrainAging\partition_2_2514.mat');
M = 20; % number of modules
mat_str_TD_in_m = zeros(M); %change namemean_FA
temp = dir(path);
temp = temp(3:end);
for i = 1 : length(temp)
    load([path,'\',temp(i).name],'mean_FA');
    for j = 1 : M
        ind_m = modules_20_60{M,j};
        ind_sort_m = sort(ind_m);
        conVal = 0;
        for s = 1 : length(ind_sort_m)-1
            conVal = conVal + sum(abs(mean_FA(ind_sort_m(s,1),ind_sort_m(s+1:end,1))));          
        end
        len = length(ind_sort_m);
        conVal_av = conVal/((len*(len-1))/2);
        mat_str_TD_in_m(i,j) = conVal_av; %change name
    end
end
save([outPath,'\','mat_str_TD_in_new_',num2str(M)],'mat_str_TD_in_m'); %change name