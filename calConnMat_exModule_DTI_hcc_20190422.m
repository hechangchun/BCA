clear
clc
%connect_count
path = 'F:\BrainAging\SDSU\SNC_deform\module_2514\TD'; %import data from M = 2514
outPath = 'F:\BrainAging\SDSU\SNC_deform\mat_ave_20190424';
load('F:\BrainAging\partition_2_2514.mat');
M = 20; % number of modules
mat_str_TD_ex_m = zeros(M,(M*(M-1))/2); %change namemean_FA
temp = dir(path);
temp = temp(3:end);
for i = 1 : length(temp)
    load([path,'\',temp(i).name],'mean_FA');
    count = 1;
    conVal = 0;
    for j = 1 : M-1
        ind_m = modules_20_60{M,j};
        ind_sort_m = sort(ind_m);
        for s = j+1 : M
            ind_n = modules_20_60{M,s};
            ind_sort_n = sort(ind_n);      
            conVal = (sum(sum(abs(mean_FA(ind_sort_m(:,1),ind_sort_n(:,1))))))/(length(ind_m) * length(ind_n));
            mat_str_TD_ex_m(i,count) = conVal;
            count = count + 1;
            conVal = 0;
        end
        %change name      
    end
end
save([outPath,'\','mat_str_TD_ex_new_',num2str(M)],'mat_str_TD_ex_m'); %change name