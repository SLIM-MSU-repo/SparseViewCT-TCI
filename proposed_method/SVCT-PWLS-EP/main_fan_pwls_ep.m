%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clc;clear;close all;
addpath(genpath('../toolbox'));

%% Load GT data
%
nslices = 26;
fin=fopen('~/Downloads/walnut.raw','r');
I=fread(fin,400*296*352,'uint16=>uint16');
Z=reshape(I,400,296,352);
xtrue = Z(:,:,101:101+nslices-1);
xtrue = double(xtrue/1.25e4);
%}
%% setup target geometry and weight

down = 1; % downsample rate
% sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
% sg = li_hat;
cg = ct_geom('fan', 'ns', 120, 'nt', 120, 'na', 16, ...
		'ds', 5, 'dt', 5, ...
		'down', down, ...
		'offset_s', 0.25, ... % quarter detector
		'offset_t', 0.0, ...
		'dsd', 949, 'dod', 408, 'dfs', 0);

ig = image_geom('nx', 400, 'ny', 296, 'nz', nslices, 'fov', 400, 'down', down);

ig.mask = ig.circ > 0;    

A = Gcone(cg, ig);
%% load external parameter
I0 = 1e10;
dir = ['../data/other/'];
fprintf('Loading sinogram, weight, kappa, fbp...\n');
load([dir '/sino_fan.mat']);
load([dir '/wi.mat']);
load([dir '/kappa.mat']);
load([dir '/denom.mat']);

load([dir '/xfbp.mat']);
% figure name 'xfbp'
% imshow(xfbp, [800 1200]);

%% setup edge-preserving regularizer
nIter = 50;
nblock = 24; 
l2b = 16;

delta = 1e1; % 10 HU
pot_arg = {'lange3', delta};  % potential function
R = Reg1(sqrt(kappa), 'beta', 2^l2b, 'pot_arg', pot_arg, 'nthread', jf('ncore'));

% fprintf('calculating fwhm...\n');
% [~,~,fwhm,~,~] = qpwls_psf(A, R, 1, ig.mask, Gdiag(wi), 'fwhmtype', 'profile');

%% Recon 
% load('slice420.mat');

fprintf('iteration begins...\n'); 
[xrlalm_msk , info] = pwls_ep_os_rlalm_2d(xfbp(ig.mask), A, sino, R, 'wi', wi, ...
            'pixmax', inf, 'isave', 'last',  'niter', nIter, 'nblock', nblock, ...
            'chat', 0, 'denom',denom, 'xtrue', xtrue, 'mask', ig.mask);
        
AAA(1,:) = info.RMSE;
AAA(2,:) = info.SSIM;  
figure name 'RMSE'
plot(info.RMSE,'-*')
xlabel('Number Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('PWLS-EP')


xrlalm = ig.embed(xrlalm_msk);
figure name 'xrlalm'
imshow(cat(2, xrlalm(:,:,end), xfbp), [800 1200]);colorbar;

% save('xrlalm.mat','xrlalm')
% save('AAA.mat', 'AAA')
% export_fig x.pdf -transparent