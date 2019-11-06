clear
clc
A=1:190; % the total of number
% n=ceil((-1+sqrt(1+8*length(A)))/2) ;%根据二次函数的求根公式，计算向量对应的上三角矩阵的维数，包括对角元.
n=ceil((1+sqrt(1+8*length(A)))/2) ;%根据二次函数的求根公式，计算向量对应的上三角矩阵的维数，对角元为0.
AA=zeros(n);
ind=find(tril(ones(n),-1));
if length(ind) > length(A)
    m = length(ind) - length(A);
    x(m) = 0;
    Anew = [A x];
end
AA(ind)=A;
AA=AA'