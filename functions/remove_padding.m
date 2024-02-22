function B=remove_padding(A,m,n)
% m、n分别为所得图像的行数与列数
[mm,nn]=size(A);
a=(mm-m)/2;
b=(nn-n)/2;
B(1:m,1:n)=A(a+1:a+m,b+1:b+n);