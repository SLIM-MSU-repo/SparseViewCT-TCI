#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:04:00 2021

@author: anishl
"""
import torch
import torch.nn
import numpy as np
from skimage.transform import resize

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    
    def forward(self, input, target):
        intersection = input*target #
        union = input+target
        
        msk = torch.zeros_like(intersection)
        msk[...,42:412,42:412] = 1.0
        dice = 2*(intersection*msk).sum()/((union*msk).sum()+1e-5) # summed over all batches
        return 1 - dice

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        mse_loss = torch.nn.MSELoss(reduction='none')
        mse=mse_loss(input,target)
        
        mse = (mse * mask.float()).sum() # gives \sigma_euclidean over unmasked elements

        non_zero_elements = mask.sum()
        loss = mse / non_zero_elements
        
        return loss
    
class WalnutBounds(torch.nn.Module):
    def __init__(self):
        super(WalnutBounds,self).__init__()
        
    def return_mask(self,size,bounds):
        self.mask = torch.zeros(size)
        self.mask[bounds[4]:bounds[5],bounds[2]:bounds[3],bounds[0]:bounds[1]]=1
        return self
        
    def getBounds(wal_num):
#         print('www',wal_num)
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
    
    def _padMask(self,sz):
        des_h = sz[0]
        des_w = sz[1]
    
        h = self.mask.shape[0]
        w = self.mask.shape[1]
    
        a = (des_h - h) // 2
        aa = des_h - a - h
    
        b = (des_w - w) // 2
        bb = des_w - b - w
        
        if a<0 or b<0: # make more general non-square padding later
            return self.mask[-a:h+a+1,-b:h+b+1]
        else: 
            return np.pad(self.mask, pad_width=((a, aa), (b, bb)), mode='constant')
    
    def rescaleMask(self,wal_num,mask):
        mask = mask.numpy()
        og_shape = mask.shape
        scale_f = abs(np.float(wal_num)-np.floor(wal_num))
        if scale_f != 0:
            if scale_f<0.5:
                scale_f = 1+scale_f
            mask_sz = tuple(np.ceil(scale_f*np.asarray(mask.shape)))
            res = resize(mask,mask_sz)
            self.mask = res
            res_mask = self._padMask(og_shape)
        else:
            res_mask = mask
        return torch.from_numpy(res_mask)
  
