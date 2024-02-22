function obj1=zero_padding(obj)
[r,c]=size(obj);
obj1=zeros(2*r,2*c);
obj1(r/2+1:3*r/2,c/2+1:3*c/2)=obj;
return