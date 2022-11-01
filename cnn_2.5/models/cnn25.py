import torch.nn as nn
import numpy as np
import os


class cnn_25(nn.Module):

    def __init__(self,num_layers = 14):
        super(cnn_25, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (3,3), 
                                    stride=1, padding=1, dilation=1, 
                                    groups=1,bias=True, padding_mode='zeros')
        self.conv_last = nn.Conv2d(64, 1, (3,3), 
                                    stride=1, padding=1, dilation=1, 
                                    groups=1, bias=True, padding_mode='zeros')
        
        self.module = nn.ModuleList([nn.Sequential(self.conv1,nn.ReLU())])
        
        for _ in range(0,num_layers):
            
            self.module.append(
            nn.Sequential(nn.Conv2d(64, 64, (3,3),stride=1, padding=1, dilation=1), 
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, dtype=None),
            nn.ReLU())
            )
        
        self.module.append(self.conv_last)
                               
    def forward(self, x):
        
        for m in self.module:
            x = m(x)
            
        return x
    