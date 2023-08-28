%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
addpath(genpath('../toolbox'));

%% load external parameter
I0 = 1e12;
dir = ['../Walnut'];

fprintf('Loading sinogram, weighting, kappa...\n');
load([dir '/sino_cone.mat']);
load([dir '/wi.mat']);
load([dir '/kappa.mat']);
load([dir '/denom.mat']);

fprintf('Loading xproc...\n'); 

% load([dir '/xfdk.mat']);
% xfdk = data_out_normalized;
% xfdk = data_lv;
% xfdk = x_lv8_walnut1;

data_true = xtrue;
%% setup edge-preserving regularizer
% set up ROI
% roi = ig.mask; start_slice = 17; end_slice = 80;
% roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0; 
% roi = roi(ig.mask);
% load('xtrue_crop17-80.mat'); % ground truth in the ROI

nIter = 10;
nblock = nviews; 
nblock
l2b = 5; %2 increase is smoother

Ab = Gblock(A_lv, nblock); clear A

delta = 1e3;  % 10 HU
pot_arg = {'lange3', delta};   % potential function

b1 = 1/ig.dx^2; b2 = 1/(ig.dx^2+ig.dy^2);
b3 = 1/ig.dz^2; b4 = 1/(ig.dx^2+ig.dz^2);
b5 = 1/(ig.dx^2+ig.dy^2+ig.dz^2);
beta = l2b*[b1 b1 b2 b2 b5 b4 b5 b4 b3 b4 b5 b4 b5];
R = Reg1(sqrt(kappa), 'type_penal','mex','offsets','3d:26','beta',beta,... 
        'pot_arg', pot_arg, 'distance_power', 0,'nthread', jf('ncore')*2-1, 'mask',ig.mask);
        % sqrt(kappa) -- achieve uniform noise
        % kappa -- achieve uniform resolution

% check fwhm 
% fprintf('calculating fwhm...\n');
%  [~,~,fwhm,~,~] = qpwls_psf(Ab, R, 1, ig.mask, Gdiag(wi), 'fwhmtype', 'profile'); 

%% Recon

fprintf('iteration begins...\n'); 
[xrlalm_msk, info] = pwls_ep_os_rlalm(xfdk(ig.mask), Ab, reshaper(sino, '2d'), R, ...
             'wi',reshaper(wi, '2d'), 'niter', nIter, 'denom',denom,...
             'chat', 0, 'xtrue', data_true, 'mask', ig.mask, 'isave', 'last');
xrlalm = ig.embed(xrlalm_msk);

%x_proc_ep = xrlalm;

addpath ../../src/

[a,b,c] = getWalnutBounds(wal_num);
xtrue1 = zeros(size(xrlalm));
xtrue1(c,b,a)=xrlalm(c,b,a);


fprintf('saving...\n')

fid=fopen(save_dir,'w');
fwrite(fid,xtrue1,'float64');
fclose(fid);

%}