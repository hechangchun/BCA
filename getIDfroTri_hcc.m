clear
clc
%% get ID from number of features
num_fea = importdata('F:\BrainAging\getIDfroDtri\num_fea.mat');
dTri = importdata('F:\BrainAging\getIDfroDtri\dTri.mat');
mat = [];
val = 0;
s = 1;
for i = 1 : length(dTri)-1
    for j = (i+1):length(dTri)
        val = dTri(j,i);
        for n = 1:length(num_fea)
            if val == num_fea(n,1)
               
                mat(s,1) = num_fea(n,1);
                mat(s,2) = i;
                mat(s,3) = j
                s = s + 1;
                break;
            end
        end
    end
end