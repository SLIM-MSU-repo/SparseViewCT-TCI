import torch
from utils import *
from train_test.train_test import test
from dataset.dataloader import load_walnut3 
import os
import numpy as np
from models.cnn25 import cnn_25
from torch.utils.data import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',default='cnn25_14',type=str)
    parser.add_argument('--rev',default=0,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--mbir',default=0,type=int)
    parser.add_argument('--num_views',default=0,type=int)
    parser.add_argument('--orbt_strt',default=0,type=str)
        
    args = parser.parse_args()
    batch_size = args.batch_size
    
    window_size = 3
    wal_idx=['106','105','103','104','102','101'][-1:]
#     wal_idx = ['101']
    exp_name = args.exp_name+'_rev'+str(args.rev)
    rev = args.rev>0
    
    test_list = {wal_idx[i] : DataLoader(load_walnut3(wal_idx[i:i+1],window_size=3,fraction=1.0,rev=True,
                                                      mbir=False,num_view=args.num_views,
                                                      orbt_strt='0',slice_length = 501,
                 path='data_dir_test/'),
                             batch_size=batch_size, num_workers=min(4,batch_size), shuffle=False) for i in range(0,len(wal_idx))
                }
    chkp = 'check_point_'+args.exp_name+'_rev'+str(args.rev)+'.pth'
    checkpoint = torch.load('saved_models/'+chkp)
    model = cnn_25()
    model.load_state_dict(checkpoint['model'])
    test(test_list, model.cuda(), save_dir=exp_name+'/image_results/')
    
main()    
