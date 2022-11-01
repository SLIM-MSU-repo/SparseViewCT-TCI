import torch
from utils import *
from train_test.train_test import train,test
from dataset.dataloader import load_walnut,load_hydro_2 
import os
import numpy as np
from models.cnn25 import cnn_25
from torch.utils.data import DataLoader
import argparse
from losses.loss_fn import MaskedMSELoss
def main():
    # NO NEED FOR ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',default='cnn25_14',type=str)
    parser.add_argument('--rev',default=0,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
#     parser.add_argument('--mbir',default=0,type=int)
    parser.add_argument('--test',default=0,type=int)
    parser.add_argument('--num_view',default=0,type=int)
    parser.add_argument('--orbt_strt',default=0,type=str)
    parser.add_argument('--resume',default=0,type=int)
    
    args = parser.parse_args()
    batch_size = args.batch_size
    
    window_size = 3
    lr = 0.0005
    exp_name = args.exp_name+'_rev'+str(args.rev)
    rev = args.rev>0
    
    
    
    train_loader =  DataLoader(load_hydro_2(rev=rev,wal_idx=['walnut101_10_0',
                                                             'walnut101_10_1',
                                                             'walnut101_10_2',
                                                             'walnut101_10_3',
                                                             'walnut101_0_0'], # name of files in data_dir/
                                window_size=window_size,
                                fraction=1,num_view=args.num_view,orbt_strt=args.orbt_strt,
                                slice_length=501,
                                path='../data_dir/'),
                                num_workers=min(4,batch_size),
                                batch_size=batch_size,    
                               shuffle=True)
    
        
    loader_dict = {'train':train_loader
                   
                  }

    mask = torch.zeros((batch_size,1,501,501))
    mask[...,70:430,50:475]=1.0
    criterion = MaskedMSELoss(mask = mask)
    model = cnn_25().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#     args.resume=1
    if args.resume>0:
        chk =torch.load('saved_models/check_point_'+exp_name+'.pth')
        model.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['opt'])
        cur_epochs = chk['epoch']
        train_loss = chk['train_loss']
        val_loss = chk['val_loss']
    else:
        cur_epochs=0
        train_loss = []
        val_loss = []

    
    train(loader_dict,model,optimizer,100,exp_name=exp_name,batch_size=batch_size,criterion=criterion)
    
if __name__=="__main__":
    main()
    
    
