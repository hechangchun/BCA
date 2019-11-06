clear
clc
fea_mat = importdata('F:\BrainAging\result_new\res_ASD\fea_mat_ASD.mat');
w = importdata('F:\BrainAging\result_new\res_ASD\w_ASD.mat');
fea_ID = unique(fea_mat);
res_ID = zeros(size(fea_ID,1)-1,2);
res_ID(:,1) = fea_ID(2:end,1);

for i = 1 : length(w) % the dimension of cell 
    for j = 1 : length(w{i,1})
        for s = 1 : size(res_ID,1) % the first element is zero
            if res_ID(s,1) == w{i,1}(1,j)              
               res_ID(s,2) = res_ID(s,2) + abs(w{i,2}(1,j)); % obsolute value
            end
        end
    end
end
res_ID_ord_ASD = sortrows(res_ID,2);
save res_ID_ord_ASD res_ID_ord_ASD
