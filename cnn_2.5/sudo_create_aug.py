import sys

# sys.path.append('/home/gabriel/anaconda3/bin/')

import numpy as np


import os
import elasticdeform
import matplotlib.pyplot as plt
import elasticdeform.torch as etorch
import torch
import multiprocessing as mp
from numpy.lib import stride_tricks
import pickle
import scipy.io as sio
from scipy import ndimage

def make_aug(args):
    save_dir = args[0]
    s=args[1]

    path_w = args[2]
    flip=args[3]
    rotate=args[4]
    
    pp_dict={(True,True):0,(True,False):1,(False,False):2,(False,True):3} # flip/no-flip, rotate/no-rotate
    pp = pp_dict[(flip,rotate)]
    np.random.seed(pp)
    
    if 'walnut' in path_w:
        walnut = np.fromfile(path_w,dtype=np.float64).astype(np.float32).reshape((501,501,501)) # walnut is 501x501x501
    else:
        walnut = np.fromfile(path_w,dtype=np.float64).astype(np.float32).reshape((448,448,448)) # hydro data was 448x448x448
        
    walnut_def = elasticdeform.deform_random_grid(walnut, sigma=s,points=5)
    
    if flip:
        walnut_def = walnut_def[:,::-1,:]
    
    if rotate:
        
        num = np.random.randint(10)
#         print(num)
        if num < 5:
            walnut_def = ndimage.rotate(walnut_def,45,reshape=False)
        else:
            walnut_def = ndimage.rotate(walnut_def,90,reshape=False)
            
    print(save_dir+str(s)+'_'+str(pp)+'.mat')
    sio.savemat(save_dir+str(s)+'_'+str(pp)+'.mat',
            {'xtrue':walnut_def})

    
    
def main():
    
    path_w = 'data_dir/walnut101.raw'

    s = [10] # sigma_x and sigma_y for gaussian kernel to generate random deformation field
    
    
    flip=[True,False]
    rotate=[True,False]
    
    walnut_args = [('data_dir/walnut101_',s[i],path_w,flip[k],rotate[l]) for i in range(num) for k in range(2) for l in range(2)]
    batch_size = len(walnut_args)
    
    p = mp.Pool(batch_size)
    p.map(make_aug,walnut_args)
    p.close()
    p.join()
    
        
if "__name__"=="__main__":
    main()
