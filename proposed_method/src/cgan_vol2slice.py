import argparse
import os
import numpy as np
import math

import sys
sys.path.insert(0, 'helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod
from classifier3dcnn.model import ConvColumn5,ConvColumn3,ConvColumn7,ConvColumn9,ConvColumn2d
from vol2slicenet import vol2slice_blk
import matplotlib.pyplot as plt
from dbgfiles import dbgplot_v2s
import datautils
import lossutils as lu
from lossutils import WalnutBounds as wb
from torch.autograd import Variable

import torch.nn as nn
# import torch.nn.functional as F
import torch
import pkbar

def return_mask(size,bounds):
    mask = torch.zeros(size)
    mask[bounds[4]:bounds[5],bounds[2]:bounds[3],bounds[0]:bounds[1]]=1
    return mask


def getBounds(wal_num):
#     print('www',wal_num)
    if wal_num == 101:
        bounds=[45,474,70,430,50,475]
    if wal_num == 103:
        bounds=[100,420,95,412,60,460]
    if wal_num == 102:
        bounds=[95,425,80,400,60,445]
    if wal_num == 104:
        bounds=[55,420,80,450,40,480]
    if wal_num == 105:
        bounds=[45,405,75,425,40,460]
    if wal_num == 106:
        bounds=[40,430,55,455,30,470]
    return bounds

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--lam", type=float, default=10, help="balance between pixel and adversarial loss 0<lam<1")
parser.add_argument("--d_int", type=int, default=10, help="iterations between discriminator training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00028, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--wal_num",type=int, default=101, help="identifier for walnut being reconstructed")
parser.add_argument("--lvl_num",type=int, default=1,help="identifier for stage or hyperlayer")
parser.add_argument("--viewtyp",type=int,default=8,help="identifier for the number of views as well as rotation/orientation")
parser.add_argument("--root_pth", default='/home/scratch/gabriel/Walnut_dataset/results/Walnut',
                    help="root directory identifier for training walnut")
parser.add_argument("--vol_size",default=[501,501,501], help="size of walnut volume")
parser.add_argument("--block_size", default=[8,500,500],help="size of subvolume")
parser.add_argument("--orbit_start", type=float, default=0,help="initial rotation of the walnut")
parser.add_argument("--save_path", type=float, default=0,help="directory to save the outputs of destreaking network")

opt = parser.parse_args()

walnut_num = opt.wal_num
print(walnut_num)
lvl_num = opt.lvl_num
vw = opt.viewtyp
root_pth = opt.root_pth
vol_sz = opt.vol_size
blck_sz = opt.block_size
obt_strt = opt.orbit_start

# SAVE_PATH = './saved_models_8view_4stage_transposed/'

SAVE_PATH = opt.save_path
os.makedirs(SAVE_PATH,exist_ok=True)    
batch_size = opt.batch_size
MODEL_NAME = 'v2sgenmsk_lvl'+str(lvl_num)+'_'+ str(vw)+'v_ep_8x500g2_transposed'
lam = opt.lam
disc_train_interval = opt.d_int

cuda = True if torch.cuda.is_available() else False


## Decide which device we want to run on
device = torch.device('cuda:0')
msk = return_mask(vol_sz,bounds=getBounds(wal_num=walnut_num))
msk = msk[:500,:500,:500]
msk_slc = msk[:,:,250].to(device)
# Loss functions
generator_loss = lu.MaskedMSELoss()
adversarial_loss = torch.nn.MSELoss()

#%% Load Data 
print('loading train dataset')
sub_blk_data_true,sub_blk_data_lv,train_ixs = datautils.load_small_batch_data(nData=(1900//batch_size + 1)*batch_size,
                                                                              wal_num=walnut_num,lvl_num=lvl_num,
                                                                              lenz=501,lenx=501,leny=501,half='full',blck=blck_sz,
                                                                              root=root_pth,vw=vw,obt_strt=obt_strt)
print('loaded datasets')

blk_depth = sub_blk_data_true.shape[3]

#%% Initialize generator and discriminator


generator= unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=1, is_segmentation=False, conv_padding=0)

v2s = vol2slice_blk.vol2slice(blk_depth)

discriminator = ConvColumn2d(1)

# Optimizers
optimizer_G = torch.optim.Adam(list(generator.parameters())+list(v2s.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#%% To Cuda
if cuda:
    generator = nn.DataParallel(generator)
    v2s = nn.DataParallel(v2s)
    discriminator = nn.DataParallel(discriminator)
    generator=generator.to(device)
    v2s=v2s.to(device)
    discriminator=discriminator.to(device)
    
    generator_loss=generator_loss.to(device)
    adversarial_loss=adversarial_loss.to(device)


#%% Train etc
# ----------
#  Training
# ----------
fig,axs = plt.subplots(1,3)
for epoch in range(opt.n_epochs):

    print('Epoch: %d/%d' % (epoch + 1, opt.n_epochs))
    kbar = pkbar.Kbar(target=train_ixs.shape[0], width=8)
    
    for batch_start_ix in np.arange(0,train_ixs.shape[0],batch_size)[:]:
            
        
        tru_data = torch.from_numpy(sub_blk_data_true[train_ixs[batch_start_ix:batch_start_ix+batch_size,0],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,1],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,2],:,:,:]).type(torch.FloatTensor).unsqueeze(1)
        
        lv_data = torch.from_numpy(sub_blk_data_lv[train_ixs[batch_start_ix:batch_start_ix+batch_size,0],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,1],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,2],:,:,:]).type(torch.FloatTensor).unsqueeze(1)
        
        #if cuda:        
        tru_data=tru_data[:,:,blk_depth//2].to(device)
        lv_data=lv_data.to(device)
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)
        
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        est_data0 = generator(lv_data)
        est_data = v2s(est_data0.squeeze(1))

        # Loss measures generator's ability to fool the discriminator
        d_pred = discriminator(est_data.detach())
        #print(d_pred)
        g_loss1 =  (1*generator_loss(est_data,tru_data,msk_slc)) 
        lam = 10**torch.floor(torch.log10(g_loss1))
        g_loss2 = (lam*adversarial_loss(d_pred, valid)) 
        g_loss = g_loss1+g_loss2
        
        
        g_loss.backward()
        optimizer_G.step()
            
        if batch_start_ix%(disc_train_interval*batch_size)==0:
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

            optimizer_D.zero_grad()
    
            # Loss for real images
            d_real_pred = discriminator(tru_data)
            d_real_loss = adversarial_loss(d_real_pred, valid)
    
            # Loss for fake images
            d_fake_pred = discriminator(est_data.detach())
            d_fake_loss = adversarial_loss(d_fake_pred, fake)
    
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()
            
        torch.cuda.empty_cache()
        
        kbar.update(batch_start_ix, values=[("Scaled D loss", g_loss2.item()), ("G loss", g_loss1.item())])
        
    if epoch%1==0:
        torch.save(generator.state_dict(), SAVE_PATH+MODEL_NAME+'_gen.pth')        
        torch.save(v2s.state_dict(), SAVE_PATH+MODEL_NAME+'_v2s.pth')        

#%% extra functions as needed
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
