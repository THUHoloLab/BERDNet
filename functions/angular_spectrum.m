% include zero padding
% size(img)=2*size(obj)

function [ img ] = angular_spectrum( pix, lamda, obj, z)
k=2*pi/lamda;
obj1=zero_padding(obj);
[rr,cc]=size(obj1);
[fx,fy]=meshgrid(linspace(-1/(2*pix),1/(2*pix),cc),linspace(-1/(2*pix),1/(2*pix),rr));
U1=fftshift(fft2(fftshift(obj1))); 
H_AS=exp(1i*k*z.*sqrt(1-(lamda*fx).^2-(lamda*fy).^2));
img=fftshift(ifft2(fftshift(U1.*H_AS))); 
return