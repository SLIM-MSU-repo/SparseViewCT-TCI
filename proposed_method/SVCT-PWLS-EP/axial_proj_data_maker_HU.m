%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir = 'Walnut';
if ~exist(dir)
    mkdir(dir)
end
I0 = 10^12;
mm2HU=10^6;
%I0 = 1e4;
%mm2HU = 1000/0.02;
A_lv = A; clear A
data_true=xtrue;
sino_true = double(A_lv * data_true); clear Abig xtrue_hi;

fprintf('adding noise...\n');
yi = I0 * exp(-sino_true ./ mm2HU);
% var = 5; 
% ye = var.* randn(size(yi)); % Gaussian white noise ~ N(0,std^2)
k = 1;
zi = k * yi ;
% error = 1/1e1;
zi = max(zi, 0);   
sino = -log(zi ./(k*I0)) * mm2HU; 
sino = sino_true;

wi = ones(size(zi));  
save([dir '/wi.mat'], 'wi');    
save([dir '/sino_cone.mat'], 'sino'); 
% figure name 'Noisy sinogram'
% imshow(sino, [2 40000]);

%% setup target geometry and fbp
% % ig = ig_big; clear ig_big;
% % ig.mask = ig.circ > 0;
% % A = Gcone(cg, ig, 'type', 'sf2','nthread', jf('ncore')*2-1);
% % fprintf('fdk...\n');
% % xfdk = feldkamp(cg,ig,sino,'window','hanning,0.5','w1cyl',1,'extrapolate_t',round(1.3*cg.nt/2));
% % xfdk = max(xfdk , 0);
% % save([dir '/xfdk.mat'], 'xfdk');
% % % figure;im('mid3',permute(xfdk,[2 1 3]),[800,1200])


%regenerate after downsampling sinogram and projections/wi
%% setup kappa
fprintf('calculating kappa...\n');
kappa = sqrt( div0(A_lv' * wi, A_lv' * ones(size(wi))) );
kappa = max(kappa, 0.01*max(col(kappa))); % kappa = max(kappa, 0.01*max(col(kappa))); for walnuts
save([dir '/kappa.mat'], 'kappa');

%% setup diag{A'WA1}
printm('Pre-calculating denominator D_A...');
denom = A_lv' * col(reshape(sum(A_lv'), size(wi)) .* wi); 
save([dir '/denom.mat'], 'denom');

cd PWLS-EP/