clc; close all;

file1 = imread('Image_1.jpg');

file2 = imread('Image_2.jpg');

% Step 1. ***************Image Aquisition*****************

subplot(3,2,1);
imshow(file1);
title('Step:1 Image Aquisition(Original Image:1)');

subplot(3,2,2);
imshow(file2);
title('Step:1 Image Aquisition(Original Image:2)');



% STEP:2 GRAY SCALE CONVERTION*********************************************

g =rgb2gray(file1);
subplot(3,2,3);
imshow(g);
title('STEP:2 Converting into Gray Image:1');

gg=rgb2gray(file2);
subplot(3,2,4);
imshow(gg);
title('STEP:2 Converting into Gray Image:2');

% STEP:3 Subtraction of original image*************************************

v= rgb2gray(file1);
v = imsubtract(v,60);
subplot(3,2,5);
imshow(v);
title('STEP:3 Subtracted Gray Image:1');
 

vv =rgb2gray(file2);
vv = imsubtract(vv,60);
subplot(3,2,6);
imshow(vv);
title('STEP:3 Subtracted Gray Image:2');

figure();

% Find HISTOGRAM of the Image**********************************************
%imhist works with only 8 bit images
% Hence convert the image to unsigned 8 bit image and plot the histogram
 
z=double(v);
subplot(3,2,1);
imhist(v);
axis off, axis tight;
title('STEP:4 Histogram of the Image:1');

zz=double(vv);
subplot(3,2,2);
imhist(vv);
axis off, axis tight;
title('STEP:4 Histogram of the Image:1');



% STEP:5 CROPED IMAGE******************************************************

c=imcrop(g,[210 86 627-210 402-86]);
subplot(3,2,3);
imshow(c);
title('STEP:5 Croped Image:1');

cc=imcrop(gg,[255 163 744-255 504-163]);
subplot(3,2,4);
imshow(cc);
title('STEP:5 Croped Image:2');



% STEP:6 RESIZED IMAGE*****************************************************

r=imresize(c,[256,256],'nearest');
subplot(3,2,5);
imshow(r);
title('STEP:6 Resized Image:1');

rr=imresize(cc,[256,256],'nearest');
subplot(3,2,6);
imshow(rr);
title('STEP:6 Resized Image:2');


figure();

% STEP:7 IMAGE SMOOTHING***************************************************

s= fspecial('gaussian',3);
f = imfilter(r,s); 
subplot(3,2,1);
imshow(f,[]),title('STEP:7 Using Gaussian Filter Smoothing Image:1 ');

ss= fspecial('gaussian',3);
ff = imfilter(rr,ss); 
subplot(3,2,2);
imshow(ff,[]),title('STEP:7 Using Gaussian Filter Smoothing Image:2');


%%% Image Segmentation Process:-

% STEP:8 CANNY EDGE DETECTION**********************************************

e=edge(f,'canny');
subplot(3,2,3);
imshow(e);
title('STEP:8 Edge Detection by Canny Filter Image:1');

ee=edge(ff,'canny');
subplot(3,2,4);
imshow(ee);
title('STEP:8 Edge Detection by Canny Filter Image:2');


 %STEP:9 Sobel EDGE DETECTION**********************************************

S1=edge(f,'roberts');
subplot(3,2,5);
imshow(S1);
title('STEP:9 Edge Detection by Sobel Filter Image:1');

SS1=edge(ff,'roberts');
subplot(3,2,6);
imshow(SS1);
title('STEP:9 Edge Detection by Sobel Filter Image:2');

figure()

% STEP:11 GAMMA CORRECTION*************************************************
%Adjust the Gamma to 0.8

S=edge(f,'sobel');
u=double(S);
subplot(3,2,1);
y= imadjust(u,[],[],0.8);  
imshow(y);
title('STEP:10 Gamma Adjusted Image:1');

SS=edge(ff,'sobel');
uu=double(SS);
subplot(3,2,2);
yy= imadjust(uu,[],[],0.8);  
imshow(yy);
title('STEP:10 Gamma Adjusted Image:2');

figure();

% STEP:12 HISTERISIS THRESHOLD*********************************************
 


mygrayimg = imread('Image_1.jpg');
mygrayimg = imresize(rgb2gray(mygrayimg),[256 256]);  

myfftimage = fft2(mygrayimg);


tmp = abs(myfftimage);
mylogimg = log(1+tmp);


[M,N] = size(myfftimage);



low = 62;
band1 = 15;
band2 = 60;



mylowpassmask = ones(M,N);
mybandpassmask = ones(M,N);



for u = 1:M
   for v = 1:N
       
       tmp = ((u-(M+1))/2)^2 +(v-(N+1)/2)^2;
       raddist = round((sqrt(tmp)));
       disp(raddist)
       
       if raddist > low  
       mylowpassmask(u,v) = 0;
       end
       
       if raddist > band2 || raddist < band1 
       mybandpassmask(u,v) = 0;
       end
   end
end


f1 = fftshift(mylowpassmask);
f3 = fftshift(mybandpassmask);


resimage1 = myfftimage.*f1;
resimage3 = myfftimage.*f3;

% Display the low pass filtered image
r1 = abs(ifft2(resimage1));
subplot(3,2,3);
imshow(r1,[]),title('STEP:11 Hysteresis Thresholding of Image: 1');




% part 2*****


% Read the image, resize it to 256 x 256
% Convert it to grey image and display it

mygrayimg = imread('Image_2.jpg');
mygrayimg = imresize(rgb2gray(mygrayimg),[256 256]);  
%subplot(2,2,1);
%imshow(mygrayimg),title('Original Gray-Image');

% Find FFT
% Use the command fft2() to get FFT of the image
% The log scale of FFT image is displayed 

myfftimage = fft2(mygrayimg);

% Take logarithmic scale for display

tmp = abs(myfftimage);
mylogimg = log(1+tmp);

%subplot(2,2,2);
%imshow(mat2gray(mylogimg)); 
%title('FFT Image');

% Find size
[M,N] = size(myfftimage);

% Create Filter array

% The cut off frequency 20 is used here

low = 62;
band1 = 15;
band2 = 60;

% create ideal high pass filter mask

% Create matrix of size equals original matrix

mylowpassmask = ones(M,N);
mybandpassmask = ones(M,N);

% Generate values for ideal high pass mask

for u = 1:M
   for v = 1:N
       
       tmp = ((u-(M+1))/2)^2 +(v-(N+1)/2)^2;
       raddist = round((sqrt(tmp)));
       disp(raddist)
       
       if raddist > low  
       mylowpassmask(u,v) = 0;
       end
       
       if raddist > band2 || raddist < band1 
       mybandpassmask(u,v) = 0;
       end
   end
end

% Shift the spectrum to the centre
f1 = fftshift(mylowpassmask);
f3 = fftshift(mybandpassmask);

% Apply the filter H to the FFT of the Image
resimage1 = myfftimage.*f1;
resimage3 = myfftimage.*f3;


% Apply the Inverse FFT to the filtered image

% Display the low pass filtered image
r1 = abs(ifft2(resimage1));
subplot(3,2,4);
imshow(r1,[]),title('STEP:11 Hysteresis Thresholding of Image: 2');



% STEP:12 HUGH TRANSFORM***************************************************

%Hu=imread('hug1.jpg');
%subplot(3,2,4);
%imshow(Hu);
%title('STEP:12 After Hugh Transformation Image:1')

%Hu1=imread('hug2.jpg');
%subplot(1,2,2);
%imshow(Hu1);
%title('STEP:12 After Hugh Transformation Image:2')

%figure();


% STEP:13 NORMALIZATION****************************************************
%n=imread('normal1.jpg');
%subplot(3,2,5);
%imshow(n);
%title('STEP:13 Normalized Image:1')

%nn=imread('normal2.jpg');
%subplot(3,2,6);
%imshow(nn);
%title('STEP:13 Normalized Image:2')




% STEP:15 MATCHING*********************************************************


N=im2double(f);
subplot(1,2,1);
imshow(N);
title('STEP:14 IMAGE 1');

NN=im2double(ff);
subplot(1,2,2);
imshow(NN);
title('STEP:14 IMAGE 2');
show(w)


 w = msgbox('DIFFERENT IRIS','Result');
 
 %w = msgbox('SAME IRIS','Result');
