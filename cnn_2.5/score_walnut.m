load([dir '/' model_name]);

data_out_normalized(isnan(data_out_normalized)) = 0;
xtrue = data_true;

xep = data_lv;

sdir = [dir '/' 'ReconFigs_' model_name];
if ~exist(sdir)
    mkdir(sdir)
end

cd(sdir)


p = [1 2 3];
data_out_normalized1 = permute(data_out_normalized,p);
xtrue1 = permute(xtrue,p);
xep1 = permute(xep,p);

figure();im('mid3',data_out_normalized1(15:485,15:485,15:485),[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, ['cnn.png'])
figure();im('mid3',xtrue1,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, ['GroundTruth.png'])
figure();im('mid3',xep1,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, ['EPRecon.png'])
load(['/mnt/shared_b/gabriel/data/LVCT_Walnuts_Mar22/Walnut' wal_num '/mask' wal_num '.mat'],'mask');
mask = double(mask);

[nmae_ep,HFENerr_ep] = calc_metric(xtrue1,xep1,mask);
[nmae_cnn,HFENerr_cnn] = calc_metric(xtrue1,data_out_normalized1,mask);

fprintf('\n')
fprintf(['EP NMAE for HID,SID -' model_name ': %f \n'],nmae_ep)
fprintf(['EP HFEN for HID,SID -' model_name ': %f \n'],HFENerr_ep)

fprintf('\n')
fprintf(['CNN NMAE for HID,SID -' model_name ': %f \n'],nmae_cnn)
fprintf(['CNN HFEN for HID,SID -' model_name ': %f \n'],HFENerr_cnn)



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
