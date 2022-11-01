
if ~isvar('lvl_num')
    lvl_num=4;
end
lvl_num
if ~isvar('nviews')
    %nviews = 8;
    nviews = 4;
end

formatSpec = '%.1f';

model_name = ['v2sgenmsk_lvl' num2str(lvl_num) '_' num2str(nviews) 'v_ep_8x500g2[8, 500, 500]_' wal_num 'rot0.0']

load(['/home/scratch/gabriel/Walnut_dataset/results/Walnut' wal_num '/' model_name '.mat'])

%load(['/nh/u/gmaliakal/Walnut_dataset/results/Walnut' wal_num '/' model_name '.mat'])

data_out_normalized(isnan(data_out_normalized)) = 0;
dir = ['/nh/u/gmaliakal/Walnut_dataset/results/Walnut' wal_num '/']; % DIR FOR THE WALNUT FILES
[dir 'walnut' wal_num '.raw']
I=fread(fopen([dir 'walnut' wal_num '.raw'],'r'),(501^3)*1,'float64'); % load walnut.raw
Z=double(reshape(I,501,501,501*1));
xtrue = Z(:,:,1:501);
xtrue = permute(xtrue,[3,2,1]);

dir2 = ['/home/scratch/gabriel/Walnut_dataset/results/Walnut' wal_num '/']
[dir2 'walnut' wal_num '_' 'lv' num2str(nviews) '_rot0.0' '_ep.raw']
I=fread(fopen([dir2 'walnut' wal_num '_' 'lv' num2str(nviews) '_rot0.0' '_ep.raw'],'r'),(501^3)*1,'float64');
Z=double(reshape(I,501,501,501*1));
xep = Z(:,:,1:501);
xep=permute(xep,[3,2,1]);
[dir2 'walnut' wal_num '_' 'lv' num2str(nviews) 'rot0.0' '_4.raw']
I=fread(fopen([dir2 'walnut' wal_num '_' 'lv' num2str(nviews) '_rot0.0' '_4.raw'],'r'),(501^3)*1,'float64');
Z=double(reshape(I,501,501,501*1));
xdc = Z(:,:,1:501);

dirr = ['/nh/u/gmaliakal/Walnut_dataset/results/Walnut' wal_num '/'];
[dirr 'FDK_walnut' wal_num '_rot_0_lv4'  '.raw']
I=fread(fopen([dirr 'FDK_walnut' wal_num '_rot_0_lv4'  '.raw'],'r'),(501^3)*1,'float64'); % LOAD FDK IMAGE
Z=double(reshape(I,501,501,501*1));
xfdk = Z(:,:,1:501);
xfdk=permute(xfdk,[3,2,1]);


cd /nh/u/gmaliakal/Walnut_dataset/results/image_results/
sdir = ['four_view_all_stagesWalnut' wal_num '_lvl' num2str(lvl_num) 'v' num2str(nviews)];
if ~exist(sdir)
    mkdir(sdir)
end
cd(['/nh/u/gmaliakal/Walnut_dataset/results/image_results/' sdir])
sdir

dir = ['/nh/u/gmaliakal/Walnut_dataset/results/Walnut' wal_num '/']; % DIR FOR THE WALNUT FILES

load([dir  '/mask' wal_num '.mat'],'mask')
mask = double(mask);
%mask = permute(mask,[1,3,2]);
%mask(mask>0)=1.0;
figure();im('mid3',mask,[0 1]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'mask132.png')



figure();im('mid3',data_out_normalized,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'DestreakingCNN.png')
figure();im('mid3',xtrue,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'GroundTruth.png')
figure();im('mid3',xdc,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'OutputImage.png')
figure();im('mid3',xep,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'EPRecon.png')
figure();im('mid3',xfdk,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'FDKRecon.png')

%if isvar('xtrue')
[nmae_prop,HFENerr_prop] = calc_metric(xtrue,xdc,mask);
[nmae_ep,HFENerr_ep] = calc_metric(xtrue,xep,mask);
[nmae_cnn,HFENerr_cnn] = calc_metric(xtrue,data_out_normalized,mask);
[nmae_fdk,HFENerr_fdk] = calc_metric(xtrue,xfdk,mask);

fprintf('\n')
fprintf(['Prop NMAE for Walnut' wal_num ': %f \n'],nmae_prop)
fprintf(['Prop HFEN for Walnut' wal_num ': %f \n'],HFENerr_prop)
%
fprintf('\n')
fprintf(['EP NMAE for Walnut' wal_num ': %f \n'],nmae_ep)
fprintf(['EP HFEN for Walnut' wal_num ': %f \n'],HFENerr_ep)

fprintf('\n')
fprintf(['CNN NMAE for Walnut' wal_num ': %f \n'],nmae_cnn)
fprintf(['CNN HFEN for Walnut' wal_num ': %f \n'],HFENerr_cnn)

fprintf('\n')
fprintf(['FDK NMAE for Walnut' wal_num ': %f \n'],nmae_fdk)
fprintf(['FDK HFEN for Walnut' wal_num ': %f \n'],HFENerr_fdk)


data_out_diff = abs(xtrue-data_out_normalized);
x_fdk_diff = abs(xtrue-xfdk);
x_ep_diff = abs(xtrue-xep);
x_dc_diff = abs(xtrue-xdc);

figure();im('mid3', data_out_diff,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'cnn_diff.png')
figure();im('mid3', x_fdk_diff,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'fdk_diff.png')
figure();im('mid3', x_ep_diff,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'EP_diff.png')
figure();im('mid3', x_dc_diff,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'dc_diff.png')

function [nmae_prop,HFENerr] = calc_metric(xtrue,xdc,mask)
    nmae_prop=( sum(abs(xtrue(:)-xdc(:)).*mask(:))/nnz(mask) ) /  ( sum(abs(xtrue(:)).*mask(:))/nnz(mask) );
    hfen_msk = @(I1,I2,msk) norm((imfilter(abs(I1),fspecial('log',15,1.5)) - imfilter(abs(I2),fspecial('log',15,1.5))).*msk,'fro')...
    /((nnz(msk)>0)*nnz(msk) + (nnz(msk)==0)*1);
    HFENerr = 0;
    for i = 1:size(xtrue,3)
        x = (hfen_msk(xtrue(:,:,i),xdc(:,:,i),mask(:,:,i))/size(xtrue,3));
        x_gt = (hfen_msk(xtrue(:,:,i),zeros(size(xtrue(:,:,i))),mask(:,:,i))/size(xtrue,3));
        if x_gt==0; x_gt = 1;end
        
    %         if isnan(x)
    %             keyboard;
    %         end
        
    
        HFENerr = HFENerr + (x/x_gt); %(hfen_msk(xtrue(:,:,i),xdc(:,:,i),mask(:,:,i))/size(xtrue,3));
    end
    HFENerr=HFENerr/size(xdc,3);
end