function [] = init_EP_2(wal_num,nviews,obt_strt,dist_sd,dist_od,sigma,num,dir)
nx=501;ny=501;
nslices = 501;

save_dir=[dir '/' 'walnut' num2str(wal_num) '_' 'lv' num2str(nviews) 'aug_ep.raw']


I=fread(fopen([dir 'walnut' num2str(wal_num) '.raw'],'r'),(501^3)*1,'float64');
Z=double(reshape(I,501,501,501*1));


xtrue = Z(:,:,1:501); 

size(xtrue)
if ~isvar('cg')%, printm 'cg: cone-beam CT geometry'
	if ~isvar('down'), down = 1; end % down sample a lot to save time
	if ~isvar('dfs'), dfs = 0; end
    fact=5
    cg = ct_geom('fan', 'ns', 150, 'nt', 150, 'na', nviews, ...
            'ds', fact, 'dt', fact, ...
            'down', down, ...
            'offset_s', 0.0, ... % quarter detector
            'offset_t', 0.0, ...
            'dsd', dist_sd, 'dod', dist_od, 'dfs', dfs,'orbit_start',obt_strt);
end

if ~isvar('ig'), printm 'ig: image geometry'
	ig = image_geom('nx', nx, 'ny', ny, 'nz', nslices, 'fov', ceil(2*cg.rmax), ...
		'down', down);
	mask2 = true([ig.nx ig.ny]);
	mask2(end) = 0; % trick: test it
	ig.mask = repmat(mask2, [1 1 ig.nz]);
	clear mask2
end

A = Gcone(cg, ig, 'type', 'sf2', 'nthread', jf('ncore')*2-1);
li_hat = double(A*xtrue);

if ~isvar('xfdk'), printm 'fdk'
	xfdk = feldkamp(cg, ig, li_hat, ...
		'extrapolate_t', ceil(1.3 * cg.nt/2)); % todo: compute carefully
prompt
end


%{
fprintf(['saving ' [dir '/FDK_walnut' num2str(wal_num) '_rot_' num2str(obt_strt) '_lv' num2str(nviews) '_transposed.raw']])
fid=fopen([dir '/FDK_walnut' num2str(wal_num) '_rot_' num2str(obt_strt) '_lv' num2str(nviews) '_transposed.raw'],'w');
fwrite(fid,xfdk,'float64');
fclose(fid);
%}
num2str(num)
num2str(wal_num)
num2str(sigma)

%{fprintf(['saving ' [dir '/FDK_walnut' num2str(wal_num) '_' num2str(sigma) '_' num2str(num) '.raw']])
fid=fopen( [dir '/FDK_walnut' num2str(wal_num) '_' num2str(sigma) '_' num2str(num) '.raw'],'w');
fwrite(fid,xfdk,'float64');
fclose(fid);


xpl=0;
xpcg1=0;
pre2=0;
xpcg2=0;
xfdk=double(xfdk);

lvl=1

%figure();im('mid3',xfdk,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
%saveas(gcf, 'fdk.png')
close all
if isvar('xtrue')
    mean(abs(xtrue(:)-xfdk(:)))
end
%}
recon_dir = dir;

%if ~exist(recon_dir)
%    mkdir(recon_dir)
%end
%formatSpec='%.1f';
%fid=fopen([recon_dir '/walnut' num2str(wal_num) '_rot' num2str(obt_strt,formatSpec) '.raw'],'w');
%%fwrite(fid,xtrue,'float64');
%fwrite(fid,xfdk,'float64');
%fclose(fid);

cd ../SVCT-PWLS-EP/
axial_proj_data_maker_HU

main_axial_pwls_ep

