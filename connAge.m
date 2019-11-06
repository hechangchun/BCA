function [mat,r_mat,p_mat,mat_ID] = connAge(mat_delM,age,variance,numCorr)
num = 0;
mat= [];
p_mat = [];
r_mat = [];
mat_ID = [];
if numCorr == 1 % linear
    for j = 1 : size(mat_delM,2)
        if isempty(variance)
            [h,p] = corr(mat_delM(:,j),age);
        end
            [h,p] = partialcorr(mat_delM(:,j),age,variance);
%         [h,p] = partialcorr(mat_delM(:,j),age,variance);
        if p < 0.05
            num = num + 1;
            mat(:,num) = mat_delM(:,j);
            p_mat(:,num) = p;
            mat_ID(:,num) = j;
            r_mat(:,num) = h;
        end
    end
elseif numCorr == 2 % quadratic
    for m = 1 : size(mat_delM,2)
        pa=polyfit(age,mat_delM(:,m),2);
        dFit=polyval(pa,age);
        if isempty(variance)
            [c,p]= corr(dFit, mat_delM(:,m), variance);
        end
        [c,p]= partialcorr(dFit, mat_delM(:,m), variance);
        if p < 0.05
            num = num + 1;
            mat(:,num) = mat_delM(:,m);
            p_mat(:,num) = p;
            mat_ID(:,num) = m;
            r_mat(:,num) = c;
        end
    end

end