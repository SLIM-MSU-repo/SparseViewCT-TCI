#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:27:47 2020

@author: anishl
"""
import torch.nn as nn
import torch.nn.functional as F


class vol2slice(nn.Module):
    """
    Converts a 3D block to a 2D slice
    
    """
    def __init__(self,nChannels):
        super(vol2slice, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(nChannels, 16, 3, padding=0)
        self.conv2 = nn.Conv2d(16, 16, 3,padding=0)
        
        self.conv3 = nn.ConvTranspose2d(16, 16, 3,padding=0)
        # self.conv4 = nn.ConvTranspose2d(16, 1, 3, padding = 0,bias=False)
        self.conv4 = nn.ConvTranspose2d(16, 1, 3, padding = 0)
        
        

    def forward(self, x):
        x = ((self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        
        x = (F.relu(self.conv3(x)))
        x = ((self.conv4(x)))
        
        return x
