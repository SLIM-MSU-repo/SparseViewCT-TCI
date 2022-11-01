import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import scipy.io as sio
class load_walnut(Dataset):
    def __init__(self,wal_idx,window_size=3,fraction=1.0,rev=True,mbir=False,num_view=8,orbt_strt='0',slice_length = 501,
                 path='/mnt/shared_b/gabriel/data/HydroSim/'):
        self.path = path
        self.wals = wal_idx # list of walnut ids
        self.window_size = window_size
        self.rev = rev
        self.slice_length=slice_length
        self.mbir = mbir
        self.orbt_strt = orbt_strt
        self.num_view = num_view
        self.index_dict = {i:np.arange(
                                        (i % self.slice_length) - self.window_size//2,
                                       1+ (i%self.slice_length) + self.window_size//2,
                                        1)
                           for i in range(0,self.slice_length*len(self.wals)
                                     )
                          }
#         self.bound = pickle.load(open('walnut_bound.pkl','rb'))
        if fraction<1.0:
            keys = list(self.index_dict.keys())
            num_keys = int(len(keys)*fraction)
            key_idx = np.unique(np.linspace(0,self.slice_length*len(self.wals)-1,num_keys).astype(np.int))
#             print(keys)
#             print(key_idx)
            self.index_dict = {i:self.index_dict[keys[i]] for j,i in enumerate(key_idx)}
        # all walnuts are 501x501x501
    def __len__(self):
        return len(self.index_dict)
    
    def __getitem__(self,idx):
        # create input volume of size window_sizex501x501
        # if idx + 1 < window_size
#         path ='/mnt/Data/Walnut_dataset_processed/'
        path = self.path
        inp_vol = torch.zeros((self.window_size,501,501))
        out_vol = torch.zeros((1,501,501))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//501]
#         path ='/mnt/Data/Walnut_dataset_processed/'
        raw_recon = np.fromfile(path+'/'+'Walnut'+walnut_id+'/'+'walnut'+walnut_id+'.raw',dtype=np.float64).astype(np.float32).reshape((501,501,501))
        
        if self.num_view == 8:
            ep_recon = [i for i in os.listdir(path+'/'+'Walnut'+walnut_id+'/') if ('rot0' in i and 'ep' in i) or 'lv8_ep.raw' in \
                        i][0]
            
            
            walnut_path = path+'/'+'Walnut'+walnut_id+'/'+ep_recon

            walnut_ep = np.fromfile(walnut_path,dtype=np.float64).astype(np.float32).reshape((501,501,501))
            
            walnut_res = raw_recon - walnut_ep
            
            c=0
        
        else:
            ep_recon = [i for i in os.listdir(path+'/'+'Walnut'+walnut_id+'/') if ('lv4rot'+str(self.orbt_strt) in i and 'ep.raw' in i)  \
                        ][0]
#             print(ep_recon)
            walnut_ep = np.fromfile(path+'/'+'Walnut'+walnut_id+'/'+ep_recon,dtype=np.float64).astype(np.float32).reshape((501,501,501))
            walnut_res = raw_recon - walnut_ep

            c=0
            
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < 501:
                inp_vol[c] = torch.Tensor(walnut_ep[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(walnut_res[idx%self.slice_length])
        
        return inp_vol,out_vol,idx
            

class load_walnut3(Dataset):
    def __init__(self,wal_idx,window_size=3,fraction=1.0,rev=True,mbir=False,num_view=8,orbt_strt='0',slice_length = 501,
                 path='/mnt/shared_b/gabriel/data/HydroSim/'):
        self.path = path
        self.wals = wal_idx # list of walnut ids
        self.window_size = window_size
        self.rev = rev
        self.slice_length=slice_length
        self.mbir = mbir
        self.orbt_strt = orbt_strt
        self.num_view = num_view
        self.index_dict = {i:np.arange(
                                        (i % self.slice_length) - self.window_size//2,
                                       1+ (i%self.slice_length) + self.window_size//2,
                                        1)
                           for i in range(0,self.slice_length*len(self.wals)
                                     )
                          }
#         self.bound = pickle.load(open('walnut_bound.pkl','rb'))
        if fraction<1.0:
            keys = list(self.index_dict.keys())
            num_keys = int(len(keys)*fraction)
            key_idx = np.unique(np.linspace(0,self.slice_length*len(self.wals)-1,num_keys).astype(np.int))
#             print(keys)
#             print(key_idx)
            self.index_dict = {i:self.index_dict[keys[i]] for j,i in enumerate(key_idx)}
        # all walnuts are 501x501x501
    def __len__(self):
        return len(self.index_dict)
    
    def __getitem__(self,idx):
        # create input volume of size window_sizex501x501
        # if idx + 1 < window_size
#         path ='/mnt/Data/Walnut_dataset_processed/'
#         idx = (idx + 250)%250 + 250
        path = self.path
        inp_vol = torch.zeros((self.window_size,501,501))
        out_vol = torch.zeros((1,501,501))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//501]
#         path ='/mnt/Data/Walnut_dataset_processed/'
        raw_recon = np.fromfile(path+'/'+'Walnut'+walnut_id+'/'+'walnut'+walnut_id+'.raw',dtype=np.float64).astype(np.float32).reshape((501,501,501))
        
        if self.num_view == 8:
            ep_recon = [i for i in os.listdir(path+'/'+'Walnut'+walnut_id+'/') if 'lv8' in i and 'aug_ep.raw' in i][0]
            
            
            walnut_path = path+'/'+'Walnut'+walnut_id+'/'+ep_recon

            walnut_ep = np.fromfile(walnut_path,dtype=np.float64).astype(np.float32).reshape((501,501,501)).transpose(1,0,2)
            
            walnut_res = raw_recon - walnut_ep
            
            c=0

        else:
            ep_recon = [i for i in os.listdir(path+'/'+'Walnut'+walnut_id+'/') if 'lv4' in i and 'aug_ep.raw' in i][0]
#             print(ep_recon)
            walnut_ep = np.fromfile(path+'/'+'Walnut'+walnut_id+'/'+\
                                    ep_recon,dtype=np.float64).astype(np.float32).reshape((501,501,501)).transpose(1,0,2)
            walnut_res = raw_recon - walnut_ep

            c=0
            
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < 501:
                inp_vol[c] = torch.Tensor(walnut_ep[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(walnut_res[idx%self.slice_length])
        
        return inp_vol,out_vol,idx
        
class load_hydro(load_walnut):
    
    def __getitem__(self,idx):
        path =self.path#'/mnt/Data/HydroSim/HydroSim_new'
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//self.slice_length]

        ep_recon = np.fromfile(path+'/'+'3D_rmi_'+str(walnut_id)+'/3D_rmi_'+str(walnut_id)+'_lv8_ep.raw',
                       dtype='float64').astype('float32').reshape((self.slice_length,self.slice_length,self.slice_length))

        raw_recon = np.fromfile(path+'/'+'3D_rmi_'+str(walnut_id)+'/3D_rmi_'+str(walnut_id)+'_tru.raw',
                       dtype='float64').astype('float32').reshape((self.slice_length,self.slice_length,self.slice_length))

        res_recon = raw_recon - ep_recon
        
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < self.slice_length:
                inp_vol[c] = torch.Tensor(ep_recon[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(res_recon[idx%self.slice_length])
        
        return inp_vol,out_vol,idx
            
        
    
class load_wal2(load_walnut):
    
    def __getitem__(self,idx):
        path=self.path
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//self.slice_length]
        
        path = '/mnt/shared_b/gabriel/data/LVCT_Walnuts_Mar22/'
        
        ep_recon = np.fromfile(path+'/'+str(walnut_id.replace('_')[0]+'.raw')+'/'+walnut_id+'_lv8_ep.raw',
                       dtype='float64').astype('float32').reshape((self.slice_length,self.slice_length,self.slice_length))

        raw_recon = np.fromfile(path+'/'+str(walnut_id.replace('_')[0]+'.raw')+'/'+walnut_id,
                       dtype='float64').astype('float32').reshape((self.slice_length,self.slice_length,self.slice_length))

        res_recon = raw_recon - ep_recon
        
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < self.slice_length:
                inp_vol[c] = torch.Tensor(ep_recon[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(res_recon[idx%self.slice_length])
        
        return inp_vol,out_vol,idx
    
    
class load_hydro_2(load_walnut3):
    # for data aug experiment
    def __getitem__(self,idx):
#         idx = (idx + 250)%250 + 250
        path =self.path#'/mnt/shared_b/gabriel/data/HydroSim/3D_rmi_sim2_1.0.00002050/'
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//self.slice_length]
        
        if self.num_view==8:
            if '10_' in walnut_id:
                ep_recon = np.fromfile(path+'/'+walnut_id+'_lv8_ep.raw',
                                   dtype='float64').astype('float32').reshape((self.slice_length,
                                                                               self.slice_length,
                                                                               self.slice_length)).transpose(2,1,0)
            else:
                ep_recon = np.fromfile(path+'/'+walnut_id+'_lv8_ep.raw',
                               dtype='float64').astype('float32').reshape((self.slice_length,
                                                                           self.slice_length,
                                                                           self.slice_length))#.transpose(2,1,0)
                
        else:
            if '3_0' in walnut_id or '6_0' in walnut_id or '9_0' in walnut_id or '10_0' in walnut_id:

                ep_recon = np.fromfile(path+'/'+walnut_id+'_lv4_ep.raw',
                           dtype='float64').astype('float32').reshape((self.slice_length,
                                                                       self.slice_length,
                                                                       self.slice_length)).transpose(2,1,0)
            else:
                ep_recon = np.fromfile(path+'/'+walnut_id+'_lv4_ep.raw',
                           dtype='float64').astype('float32').reshape((self.slice_length,
                                                                       self.slice_length,
                                                                       self.slice_length))
                
        if os.path.isfile(path+'/'+walnut_id+'.mat'):
            raw_recon = sio.loadmat(path+'/'+walnut_id+'.mat')['xtrue']
            
        elif os.path.isfile(path+'/'+walnut_id+'_tru.raw'):
            raw_recon = np.fromfile(path+'/'+walnut_id+'_tru.raw',
                                    dtype=np.float64).astype(np.float32).reshape((\
                                                                              self.slice_length,
                                                                                self.slice_length,
                                                                                self.slice_length))
        else:
            raw_recon = np.fromfile(path+'/'+walnut_id+'.raw',
                                    dtype=np.float64).astype(np.float32).reshape((\
                                                                              self.slice_length,
                                                                                self.slice_length,
                                                                                self.slice_length))
            
            
        res_recon = raw_recon - ep_recon
        
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < self.slice_length:
                inp_vol[c] = torch.Tensor(ep_recon[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(res_recon[idx%self.slice_length])
        
        return inp_vol,out_vol,idx

            
class load_wal_2(load_walnut):
# for data aug experiment

    def __getitem__(self,idx):

        path =self.path#'/mnt/shared_b/gabriel/data/LVCT_Walnuts_Mar22/LVCT_Walnuts/Walnut101/'
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        # locate walnut
        walnut_id = self.wals[idx//self.slice_length]
        
        if self.num_view==8:
            ep_recon = np.fromfile(path+'/'+walnut_id+'_lv8_ep.raw',
                           dtype='float64').astype('float32').reshape((self.slice_length,
                                                                       self.slice_length,
                                                                       self.slice_length)).transpose(2,1,0)
        else:
            ep_recon = np.fromfile(path+'/'+walnut_id+'_lv4_ep.raw',
                           dtype='float64').astype('float32').reshape((self.slice_length,
                                                                       self.slice_length,
                                                                       self.slice_length)).transpose(2,1,0)
            
        raw_recon = sio.loadmat(path+'/'+walnut_id+'.mat')['xtrue']
        res_recon = raw_recon - ep_recon
        
        inp_vol = torch.zeros((self.window_size,self.slice_length,self.slice_length))
        out_vol = torch.zeros((1,self.slice_length,self.slice_length))
        c = 0
        for i in self.index_dict[list(self.index_dict.keys())[idx]]:
#             print(i)
            if i>0 and i < self.slice_length:
                inp_vol[c] = torch.Tensor(ep_recon[i%self.slice_length])
            c+=1
            
        out_vol[0] = torch.Tensor(res_recon[idx%self.slice_length])
        
        return inp_vol,out_vol,idx
    
    
