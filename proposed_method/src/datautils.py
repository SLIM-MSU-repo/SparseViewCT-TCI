import numpy as np
from numpy.lib import stride_tricks
import os
import itertools
import scipy.io as sio

def ext3Dpatch(data, blck, strd):
    """
    Function to extract 3D patches or subvolumes from an input volume given subvolume size and stride
    Parameters
    ----------
    data : input 3D volume (np float) from which patches are to be extracted
    blck : list of subvolume dimensions
    strd : list of strides along corresponding dimensions

    Returns
    -------
    data6 : view into data with specified block sizes and stride (numpy volume)

    """
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)

def load_walnutdata_raw(path,fname):
    """
    Function to load a walnut volume from prescribed subvolumes

    Parameters
    ----------
    path : path to walnut (string)
    fname : walnut file name (string)

    Returns
    -------
    numpy array containing the walnut image volume

    """
    return np.fromfile(os.path.join(path,fname),
                       dtype='float64').astype('float32')


def load_test_data(wal_num=3,lvl_num=1,
                   lenz=352,lenx=296,leny=400,frac=0.5,blck=[32,32,32],strd=[1,1,1],half='top',depth_z=31,
                   root='/nh/u/gmaliakal/Walnut_dataset/results/Walnut',vw=8,obt_strt='0',
                  dist_sd = '',dist_od = ''):
#                    root='/n/escanaba/w/anishl/WalnutsOwnReconstructions/Walnut',vw=8,obt_strt=0):
    """
    Function to extract 3D patches from input walnut volumes (streaked and ground truth) at test time, across stages of the extreme LV recon algorithm 

    Parameters
    ----------
    wal_num : int, optional
        walnut identifier. The default is 3.
    lvl_num : int, optional
        stage identifier. The default is 1.
    lenz : int, optional
        z-dimension of subvolume for patch extraction. The default is 352.
    lenx : int, optional
        x-dimension of subvolume for patch extraction. The default is 296.
    leny : int, optional
        y-dimension of subvolume for patch extraction. The default is 400.
    frac : float, optional
        split for train-test (now obsolete). The default is 0.5.
    blck : list, optional
        blocksize for patch extraction. The default is [32,32,32].
    strd : list, optional
        stride for patch extraction. The default is [1,1,1].
    half : string, optional
        top or bottom half for input volume. The default is 'top'.
    depth_z : int, optional
        For the depth of the walnut half when top or bottom not specified (now obsolete). The default is 31.

    Returns
    -------
    input walnut volume from previous stage
    ground truth numpy array volume corresponding to input
    sub_blocks_lv : 
        input (FDK or previous stage DC output) numpy array for training subvolumes
    ixs : list of integers for input volume indices.

    """
    
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'

    else:
        lvl_num = str(lvl_num-1)
    
    path=root + str(wal_num)
    fname_tru='walnut'+ str(wal_num) + '.raw'
    
    if len(dist_sd)==0:
        fname_lv='walnut'+ str(wal_num) + '_lv' + str(vw) + '_rot' + str(obt_strt) + '_' + str(lvl_num) + '.raw'
    else:
        fname_lv='walnut'+ str(wal_num) + '_lv' + str(vw) + '_rot' + str(obt_strt) + '_'+dist_sd+'_'+dist_od+'_'+ str(lvl_num) + '.raw'
                
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])

    if flg==1:
        data_lv=data_lv.transpose(2,1,0)
        # data_tru=data_tru.transpose(2,1,0)

    data_lv_copy = data_lv.copy()
        
    if half == 'top': 
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = int(np.floor(0.5*lenz))
        end_z = start_z+depth_z
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)
        
    ixs = np.array([(i,j,k) 
                    for i,j,k in 
                    list(itertools.product(
                        range(sub_blocks_tru.shape[0]),range(sub_blocks_tru.shape[1]),range(sub_blocks_tru.shape[2])))])
    
    # For one volume only
    # ixs = np.array([(i,j) 
    #                 for i,j in 
    #                 list(itertools.product(
    #                     range(sub_blocks_tru.shape[1]),range(sub_blocks_tru.shape[2])))])
    
    # ixs = np.random.permutation(ixs)
    # ndata = ixs.shape[0]
    # ntrain = np.floor((1-frac)*ixs.shape[0]).astype('int32')
    if half == 'top':    
        return data_lv_copy[:int(np.floor(0.5*lenz)),:,:],data_tru[:int(np.floor(0.5*lenz)),:,:],sub_blocks_lv,ixs
    elif half == 'full':
        return data_lv_copy,data_tru,sub_blocks_lv,ixs
    else:
        return data_lv_copy[start_z:end_z,:,:],data_tru[start_z:end_z,:,:],sub_blocks_lv,ixs

if __name__ == "__main__":
    lv,tru,lv_blk,ixs = load_test_data(half='bot')
        
# TODO: get subblocks, get batches, check consistency between LV and FV patches

def load_small_batch_data(nData=500000,wal_num=1,lvl_num=1,
                    lenz=352,lenx=296,leny=400,blck=[8,500,500],strd=[1,1,1],half='top',
                    root='/nh/u/gmaliakal/Walnut_dataset/results/Walnut',vw=8,obt_strt=0):
    """
    Function to extract 3D patches from input walnut volumes (streaked and ground truth) at training time, 
    across stages of the extreme LV recon algorithm 


    Parameters
    ----------
    nData : int, optional
        number of 3D subvolumes/ training pairs. The default is 500000.
    wal_num : int, optional
        identifier for walnut. The default is 1.
    lvl_num : int, optional
        identiier for the stage. The default is 1.
    lenz : int, optional
        z-dimension of subvolume for patch extraction. The default is 352.
    lenx : int, optional
        x-dimension of subvolume for patch extraction. The default is 296.
    leny : int, optional
        y-dimension of subvolume for patch extraction. The default is 400.
    blck : list, optional
        blocksize for patch extraction. The default is [32,32,32].
    strd : list, optional
        stride for patch extraction. The default is [1,1,1].
    half : string, optional
        top or bottom half for input volume. The default is 'top'.
        
        
    Returns
    -------
    sub_blocks_tru : ground truth (full view) numpy array for training subvolumes
        DESCRIPTION.
    sub_blocks_lv : input (FDK or previous stage DC output) numpy array for training subvolumes
    ixs : list of integers for input volume and corresponding ground truth indices.
    
    """
    
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
    else:
        lvl_num = str(lvl_num-1)
    
    path=root + str(wal_num)
    fname_tru='walnut'+ str(wal_num) + '.raw'
    
    fname_lv='walnut'+ str(wal_num) + '_lv' + str(vw) + '_rot' + str(obt_strt) + '_' + str(lvl_num) + '.raw'
            
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])
    
    if flg==1:
        data_lv=data_lv.transpose(2,1,0)

    if half == 'top':
        print('only top half used for training')    
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = np.floor(0.5*lenz)
        end_z = start_z+blck[0]+1
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)

    ixs0=np.random.randint(0,sub_blocks_tru.shape[0],[nData,1])
    ixs1=np.random.randint(0,sub_blocks_tru.shape[1],[nData,1])
    ixs2=np.random.randint(0,sub_blocks_tru.shape[2],[nData,1])
    
    ixs = np.concatenate((ixs0,ixs1,ixs2),axis=1)

    return sub_blocks_tru,sub_blocks_lv,ixs

def load_small_batch_segdata(nData=500000,wal_num=1,lvl_num=1,
                    lenz=352,lenx=296,leny=400,blck=[8,500,500],strd=[1,1,1],half='top',
                    root='/n/escanaba/w/anishl/WalnutsOwnReconstructions/Walnut',vw=8,obt_strt=0):
    """
    Function to extract 3D patches from input walnut volumes (streaked and ground truth) at training time, 
    across stages of the extreme LV recon algorithm 

    
    """
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
    else:
        lvl_num = str(lvl_num-1)
        
    
    
    path=root+str(wal_num)
    fname_tru='walnut'+str(wal_num)+'.raw'
    fname_lv='walnut'+str(wal_num)+'_lv' + str(vw) + '_' + 'rot' + str(obt_strt) + '_' + lvl_num + '.raw'
    
    fname_seg = 'GTseg' + str(wal_num) + '.mat'
    
    
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])
    
    segdata = sio.loadmat(path+'/'+fname_seg)
    seg_gt = segdata['seg']
    
    if flg==1:
        data_lv=data_lv.transpose(2,1,0)

    if half == 'top':
        print('only top half used for training')    
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
        sub_blocks_seg = ext3Dpatch(seg_gt,blck,strd)
    else:
        start_z = np.floor(0.5*lenz)
        end_z = start_z+blck[0]+1
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)

    ixs0=np.random.randint(0,sub_blocks_tru.shape[0],[nData,1])
    ixs1=np.random.randint(0,sub_blocks_tru.shape[1],[nData,1])
    ixs2=np.random.randint(0,sub_blocks_tru.shape[2],[nData,1])


    ixs = np.concatenate((ixs0,ixs1,ixs2),axis=1)

    return sub_blocks_tru,sub_blocks_lv,sub_blocks_seg,ixs

def load_hydro_train_data(nData=500000,hid='1.0.00002301',sid='sim1',lvl_num=1,
                    lenz=448,lenx=448,leny=448,blck=[8,448,448],strd=[1,1,1],half='top',
                    root='/n/escanaba/w/anishl/HydroSim/3D_rmi_',vw=8):
    
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
    else:
        lvl_num = str(lvl_num-1)
        
    
    
    path=root+sid+'_'+hid
    fname_tru='3D_rmi_'+str(sid)+'_'+str(hid)+'_tru.raw'
    fname_lv='3D_rmi_'+str(sid)+'_'+str(hid)+'_lv' + str(vw) + '_' + lvl_num + '.raw'
    
    
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])
    
    if flg==1:
        data_lv=data_lv.transpose(2,1,0)

    if half == 'top':
        print('only top half used for training')    
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = np.floor(0.5*lenz)
        end_z = start_z+blck[0]+1
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)

    ixs0=np.random.randint(0,sub_blocks_tru.shape[0],[nData,1])
    ixs1=np.random.randint(0,sub_blocks_tru.shape[1],[nData,1])
    ixs2=np.random.randint(0,sub_blocks_tru.shape[2],[nData,1])
    
    ixs = np.concatenate((ixs0,ixs1,ixs2),axis=1)

    return sub_blocks_tru,sub_blocks_lv,ixs


def load_hydro_test_data(mhid,hid='1.0.00002301',sid='sim1',lvl_num=1,
                   lenz=448,lenx=448,leny=448,frac=0.5,blck=[8,448,448],strd=[1,1,1],half='top',depth_z=448,
                   root='/n/escanaba/w/anishl/HydroSim/3D_rmi_',vw=8):
    
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
        path=root + sid +'_'+ hid
        fname_tru='3D_rmi_' + sid +'_' + hid + '_tru.raw'
        fname_lv='3D_rmi_'+ sid + '_' + hid + '_lv' + str(vw) + '_' + lvl_num + '.raw'

    else:
        lvl_num = str(lvl_num-1)
    
        path=root + sid +'_'+ hid
        fname_tru='3D_rmi_' +sid +'_' + hid + '_tru.raw'
        fname_lv='3D_rmi_'+  mhid+'_'+sid + '_' + hid + '_lv' + str(vw) + '_' + lvl_num + '.raw'
    
    
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])

    if flg==1:
        data_lv=data_lv.transpose(2,1,0)
        # data_tru=data_tru.transpose(2,1,0)

    data_lv_copy = data_lv.copy()
        
    if half == 'top': 
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = int(np.floor(0.5*lenz))
        end_z = start_z+depth_z
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)
        
    ixs = np.array([(i,j,k) 
                    for i,j,k in 
                    list(itertools.product(
                        range(sub_blocks_tru.shape[0]),range(sub_blocks_tru.shape[1]),range(sub_blocks_tru.shape[2])))])
    
    if half == 'top':    
        return data_lv_copy[:int(np.floor(0.5*lenz)),:,:],data_tru[:int(np.floor(0.5*lenz)),:,:],sub_blocks_lv,ixs
    elif half == 'full':
        return data_lv_copy,data_tru,sub_blocks_lv,ixs
    else:
        return data_lv_copy[start_z:end_z,:,:],data_tru[start_z:end_z,:,:],sub_blocks_lv,ixs

