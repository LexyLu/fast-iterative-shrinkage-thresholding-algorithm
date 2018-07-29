%% REFERENCE
% https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf

%% COST FUNCTION
% x^* = argmin_x { 1/2 * || A(X) - Y ||_2^2 + lambda * || X ||_1 }
%
% x^k+1 = threshold(x^k - 1/L*AT(A(x^k)) - Y), lambda/L)

%%
clear ;
close all;
home;

%% GPU Processing
% If there is GPU device on your board, 
% then isgpu is true. Otherwise, it is false.
bgpu    = false;
bfig    = true;

%%  SYSTEM SETTING
N       = 512;
VIEW    = 360;
THETA   = linspace(0, 180, VIEW + 1);   THETA(end) = [];

A       = @(x) radon(x, THETA);
AT      = @(y) iradon(y, THETA, 'none', N);
AINV    = @(y) iradon(y, THETA, N);

%% DATA GENERATION
load('XCAT512.mat');
x       = imresize(double(XCAT512), [N, N]);
p       = A(x);
x_full  = AINV(p);

%% LOW-DOSE SINOGRAM GENERATION
i0     	= 5e4;
pn     	= exp(-p);
pn     	= i0.*pn;
pn     	= poissrnd(pn);
pn      = max(-log(max(pn,1)./i0),0);

x_low   = AINV(pn);

%% NEWTON METHOD INITIALIZATION
LAMBDA  = 1e-2;
T       = 1e-3;

y       = pn;
x0      = zeros(size(x));
niter   = 5e1;

L1              = @(x) norm(x, 1);
L2              = @(x) power(norm(x, 'fro'), 2);
COST.equation   = '1/2 * || A(X) - Y ||_2^2 + lambda * || X ||_1';
COST.function	= @(x) 1/2 * L2(A(x) - y) + LAMBDA * L1(x);

%% RUN NEWTON METHOD
if bgpu
    y  = gpuArray(y);
    x0 = gpuArray(x0);
end

[x_fista, obj]	= FISTA(A, AT, x0, y, LAMBDA, 1/T, niter, COST, bfig);

%% CALCUATE QUANTIFICATION FACTOR 
x_low           = max(x_low, 0);
x_fista         = max(x_fista, 0);
nor             = max(x(:));

mse_x_low       = immse(x_low./nor, x./nor);
mse_x_fista     = immse(x_fista./nor, x./nor);

psnr_x_low      = psnr(x_low./nor, x./nor);
psnr_x_fista    = psnr(x_fista./nor, x./nor);

ssim_x_low      = ssim(x_low./nor, x./nor);
ssim_x_fista	= ssim(x_fista./nor, x./nor);

%% DISPLAY
wndImg  = [0, 0.03];

figure(1); 
colormap(gray(256));

suptitle('FISTA Method');
subplot(231);   imagesc(x,          wndImg);	axis image off;     title('ground truth');
subplot(232);   imagesc(x_full,     wndImg);  	axis image off;     title(['full-dose_{FBP, view : ', num2str(VIEW) '}']);
subplot(234);   imagesc(x_low,      wndImg);  	axis image off;     title({['low-dose_{FBP, view : ', num2str(VIEW) '}'], ['MSE : ' num2str(mse_x_low, '%.4e')], ['PSNR : ' num2str(psnr_x_low, '%.4f')], ['SSIM : ' num2str(ssim_x_low, '%.4f')]});
subplot(235);   imagesc(x_fista,	wndImg);  	axis image off;     title({['recon_{FISTA}'], ['MSE : ' num2str(mse_x_fista, '%.4e')], ['PSNR : ' num2str(psnr_x_fista, '%.4f')], ['SSIM : ' num2str(ssim_x_fista, '%.4f')]});

subplot(2,3,[3,6]); semilogy(obj, '*-');    title(COST.equation);  xlabel('# of iteration');   ylabel('Cost function'); 
                                            xlim([1, niter]);   grid on; grid minor;
