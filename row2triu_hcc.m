clear
clc
A=1:190; % the total of number
% n=ceil((-1+sqrt(1+8*length(A)))/2) ;%���ݶ��κ����������ʽ������������Ӧ�������Ǿ����ά���������Խ�Ԫ.
n=ceil((1+sqrt(1+8*length(A)))/2) ;%���ݶ��κ����������ʽ������������Ӧ�������Ǿ����ά�����Խ�ԪΪ0.
AA=zeros(n);
ind=find(tril(ones(n),-1));
if length(ind) > length(A)
    m = length(ind) - length(A);
    x(m) = 0;
    Anew = [A x];
end
AA(ind)=A;
AA=AA'