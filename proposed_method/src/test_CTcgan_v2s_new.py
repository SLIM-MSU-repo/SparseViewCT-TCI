#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:13:04 2020

@author: anishl
"""
import argparse

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import time
import datautils
import pkbar
import h5py
import scipy.io as sio
from lossutils import WalnutBounds as wb
# import cv2

import sys
# sys.path.insert(0, '../helpercode/')
sys.path.insert(0,'./helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod
from vol2slicenet import vol2slice_blk
from dbgfiles import dbgplot_v2s

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

#%% Specify model name, test data and path

parser = argparse.ArgumentParser()
parser.add_argument("--wal_num", default=102, help="identifier for walnut being reconstructed")
parser.add_argument("--lvl_num",type=int, default=4,help="identifier for stage or hyperlayer")
parser.add_argument("--viewtyp",type=int,default=4,help="identifier for the number of views as well as rotation/orientation")
parser.add_argument("--root_pth", default='/home/scratch/gabriel/Walnut_dataset/results/Walnut',
                    help="root directory identifier for training walnut")
parser.add_argument("--vol_size",default=[501,501,501], help="size of walnut volume")
parser.add_argument("--block_size", default=[8,500,500],help="size of subvolume")
parser.add_argument("--orbit_start",type=str,default='0', help = "initial rotation of the walnut")
parser.add_argument("--sd_dist",type=str,default='', help = "source detector distance")
parser.add_argument("--od_dist",type=str,default='', help = "object detector distance")


opt = parser.parse_args()
# print(opt)
# to enforce int-ness in bash multiplication results
if '.0' in opt.sd_dist:
    opt.sd_dist = opt.sd_dist[:-2]
if '.0' in opt.od_dist:
    opt.od_dist = opt.od_dist[:-2]

print(opt)
# exit(0)
vw = opt.viewtyp
lvl_num=opt.lvl_num
walnut_num=opt.wal_num
root_pth = opt.root_pth
vol_sz = opt.vol_size
obt_strt = opt.orbit_start

# PATH = './saved_models_8view_4stage_transposed/'
PATH = './saved_models_4view_8stage/'

# MODEL_NAME = 'v2sgenmsk_lvl'+ str(lvl_num)+'_'+str(vw)+'v_ep_8x500g2_transposed'
MODEL_NAME = 'v2sgenmsk_lvl'+ str(lvl_num)+'_'+str(vw)+'v_ep_8x500g2'
#%% load data for test
strd = [1,1,1]
mode = 'subvol'
batch_size = 8
blck_sz = opt.block_size

data_lv,data_true,sub_blk_data_lv,train_ixs = datautils.load_test_data(wal_num=walnut_num,
                                                                       lvl_num=lvl_num,half='full',blck=blck_sz,strd=strd,
                                                                       lenz=501,lenx=501,leny=501,depth_z=64,root=root_pth,
                                                                       vw=vw,  
                                                                       obt_strt=obt_strt,
                                                                       dist_sd = opt.sd_dist,
                                                                       dist_od=opt.od_dist
                                                                      )

blk_size = sub_blk_data_lv.shape[3:]

blk_depth = blk_size[0]

#%% load model for testing in eval mode
torch.cuda.empty_cache()
device = torch.device('cuda:0')
# device1 = torch.device('cuda:1')
model = nn.DataParallel(unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=1, is_segmentation=False, conv_padding=0))

# model = model.load_state_dict(torch.load(PATH+MODEL_NAME+'.pth',map_location='cpu'))
model.load_state_dict(torch.load(PATH+MODEL_NAME+'_gen.pth'))
model = model.to(device)
model = model.eval()


v2s = nn.DataParallel(vol2slice_blk.vol2slice(blk_depth))
v2s.load_state_dict(torch.load(PATH+MODEL_NAME+'_v2s.pth'))
v2s = v2s.to(device)
v2s = v2s.eval()

msk = wb()
msk = wb.return_mask(msk,vol_sz,wb.getBounds(int(float(walnut_num))))
# msk = msk[:500,:500,:500]
msk_slc = wb.rescaleMask(msk,float(walnut_num),msk.mask[:,:,250]).to(device)
msk_slc = msk_slc[:500,:500]

#%%

data_denoised = np.zeros_like(data_true)
weights = np.zeros_like(data_true)
data_denoised_views = datautils.ext3Dpatch(data_denoised, blk_size, strd)
weights_views = datautils.ext3Dpatch(weights, blk_size, strd)
# print(weights.sum())
# print(weights_views.sum())
# print(weights.shape)
# print(weights_views.shape)
kbar = pkbar.Kbar(target=train_ixs.shape[0], width=20)
fig,axs = plt.subplots(1,3)

if mode == 'subvol':
    for batch_start_ix in np.arange(0,train_ixs.shape[0],batch_size)[:]:
                
        
        ix_z = train_ixs[batch_start_ix:batch_start_ix+batch_size,0]
        ix_x = train_ixs[batch_start_ix:batch_start_ix+batch_size,2]
        ix_y = train_ixs[batch_start_ix:batch_start_ix+batch_size,1]
        
        lv_data = torch.from_numpy(sub_blk_data_lv[ix_z,
                                ix_y,ix_x,:,:,:]
                                   ).type(torch.FloatTensor).unsqueeze(1)
        lv_data = lv_data.to(device)
        # pass data thru denoiser
        

        
        out_data = model(lv_data).squeeze(1)
        
#         model.cpu()
        
        v2s = nn.DataParallel(vol2slice_blk.vol2slice(blk_depth))
        v2s.load_state_dict(torch.load(PATH+MODEL_NAME+'_v2s.pth'))
        v2s = v2s.to(device)
        v2s = v2s.eval()
        
#         del(lv_data)
        out_data2 = v2s(out_data)
        
#         v2s.cpu()
#         del(out_data)
        
        out_data2 = out_data2*msk_slc.float()
        # out_data = out_data.cpu().detach().numpy()
        #ix_z=0
        
        # PLOT FOR Debug
        # dbgplot_v2s.dbgplt(lv_data, out_data, out_data2, axs)
        
        if batch_start_ix>=1000:
            ctr=1
        
        data_denoised_views[ix_z,ix_y,ix_x,blk_depth//2,:,:]+=out_data2.cpu().detach().numpy().squeeze(1)
        weights_views[ix_z,ix_y,ix_x,blk_depth//2,:,:]+=torch.ones_like(out_data2).cpu().detach().numpy().squeeze(1)
        del(out_data2)
        model.to(device)

        kbar.update(batch_start_ix)
#         print(weights.sum())

#     print('done1')
    
    data_out_normalized = data_denoised/weights
#     print('done2')
else:
    data_lv=torch.from_numpy(data_lv).unsqueeze(0).unsqueeze(0).to(device)
    out_data = model(data_lv)
    
    
# sio.savemat('/s1/anishl/LVCT/results/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) +'.mat',{"data_true":data_true,"data_out_normalized":data_out_normalized,"data_lv":data_lv})
# print('done')
# sio.savemat('//nh/u/gmaliakal/results//Walnut'+str(walnut_num)+'/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) +'_' + str(opt.sd_dist)+ '_' + str(opt.od_dist) + '.mat',
#             {"data_true":data_true,
#             "data_out_normalized":data_out_normalized,
#             "data_lv":data_lv}
#            )
# sio.savemat('/nh/u/gmaliakal/Walnut_dataset/results//Walnut'+str(walnut_num)+'/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) + '.mat',
#             {"data_true":data_true,
#             "data_out_normalized":data_out_normalized,
#             "data_lv":data_lv}
#            )

# sio.savemat('/nh/u/gmaliakal/Walnut_dataset/results//Walnut'+str(walnut_num)+'/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) + '.mat',
#             {
#             "data_out_normalized":data_out_normalized
#            }
#            )
# print(data_out_normalized.shape)
# print(data_out_normalized.dtype)
print('/home/scratch/gabriel/Walnut_dataset/results/Walnut'+str(walnut_num)+'/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) + '.mat')
sio.savemat('/home/scratch/gabriel/Walnut_dataset/results/Walnut'+str(walnut_num)+'/'+MODEL_NAME+str(blck_sz)+'_'+str(walnut_num)+ 'rot' + str(obt_strt) + '.mat',
            {
            "data_out_normalized":data_out_normalized
           }
           )
# print('done done')
