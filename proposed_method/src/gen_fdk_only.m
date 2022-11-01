function [] = gen_fdk_only(wal_num,nviews,obt_strt)
nx=501;ny=501;
nslices = 501;
dist_sd=2000;
dist_od=408;
dir = ['//nh/u/gmaliakal/Walnut_dataset/results/Walnut' num2str(wal_num) '/'];

I=fread(fopen([dir 'walnut' num2str(wal_num) '.raw'],'r'),(501^3)*1,'float64');
Z=double(reshape(I,501,501,501*1));
Z = permute(Z,(2,1,3))

xtrue = Z(:,:,1:501); 

fid=fopen([dir '/' num2str(wal_num) '_transposed' '.raw'],'w');
fwrite(fid,xtrue,'float64');

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
		'extrapolate_t', ceil( cg.nt)); % todo: compute carefully
prompt
end


xpl=0;
xpcg1=0;
pre2=0;
xpcg2=0;
xfdk=double(xfdk);

lvl=1
cd (dir)


figure();im('mid3',xfdk,[0 0.07]), cbar, title ' ' ; colormap gray; axis off
saveas(gcf, 'fdk.png')
close all
if isvar('xtrue')
    mean(abs(xtrue(:)-xfdk(:)))
end
fprintf(['saving ' [dir '/FDK_walnut' num2str(wal_num) '_rot_' num2str(obt_strt) '_lv' num2str(nviews) '.raw']])
fid=fopen([dir '/FDK_walnut' num2str(wal_num) '_rot_' num2str(obt_strt) '_lv' num2str(nviews) '.raw'],'w');
%fwrite(fid,xtrue,'float64');
fwrite(fid,xfdk,'float64');
fclose(fid);