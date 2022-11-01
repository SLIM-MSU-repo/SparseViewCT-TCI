import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from utils.utils import *
from losses.loss_fn import *
from skimage.metrics import peak_signal_noise_ratio
import scipy.ndimage as ndimage
def train(loader_dict, model, opt, epochs = 100, lr = 1e-3,amp=False,exp_name='unet_12_21_21',
          cur_epoch=0,train_loss = [],val_loss = [],criterion=None,
          batch_size=4,   verbose=True,nrmse = True):

    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    if criterion is None:

        criterion = torch.nn.MSELoss(reduction='none')
        
    best_val_loss = np.inf
    loss_iter = []
    if len(train_loss)==0:
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)
    elif train_loss.shape[0]<epochs:
        train_loss_temp = np.zeros(epochs)
        val_loss_temp = np.zeros_like(train_loss_temp)
        
        train_loss_temp[:cur_epoch+1] = train_loss[:cur_epoch+1]
        val_loss_temp[:cur_epoch+1] = val_loss[:cur_epoch+1]
        val_loss = val_loss_temp
        train_loss = train_loss_temp
        
    model_best = model
    
    for epoch in range(cur_epoch,epochs):
#         print(epoch)
        cc = 0
        for phase in ['train','val'][:1]:
            if phase =='val':
                model.train(False)
            else:
                
                model.train(True)
                
            loader = loader_dict[phase]
            count=0
            for img,gt,name in loader:

                count+=1

                img = img.cuda()
                gt = gt.cuda()

                    
                with torch.cuda.amp.autocast(enabled=False):                
                    out = model(img)
                    
                    loss = criterion(input=out,
                                     target=gt
                                     )
#                     print('idx ',name)
#                     print('img size ',img.size())
#                     print('gt size ',gt.size())
                          
                    loss = (loss).mean()
#                     del(mask)
                    
                if phase =='train':
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_iter.append(loss.detach().cpu().numpy())
                    np.save('lioss_iter.npy',loss_iter)
                    train_loss[epoch]+=loss.detach().cpu().numpy()

                if phase=='val':
                    val_loss[epoch]+=loss.detach().cpu().numpy()
                                
                del(img)
                del(gt)
                del(out)
                del(loss)
                torch.cuda.empty_cache()
#                 break
            if phase =='val':
                print(phase+' Loss - '+str(val_loss[epoch]))
                if val_loss[epoch]<best_val_loss:
                    best_val_loss=val_loss[epoch]
                    checkpoint = {'model':model.state_dict(),'opt':opt.state_dict(),'epoch':epoch,
                                 'val_loss ':val_loss,'train_loss ':train_loss}
                    best_model = model
                    torch.save(checkpoint,'saved_models/best_'+exp_name+'.pth')

                checkpoint = {'model':model.state_dict(),'opt':opt.state_dict(),'epoch':epoch,
                             'val_loss':val_loss,'train_loss':train_loss}

                torch.save(checkpoint, 'saved_models/check_point_'+exp_name+'.pth')
            else:
                model.train(False)
                checkpoint = {'model':model.state_dict(),'opt':opt.state_dict(),'epoch':epoch,
                             'val_loss':val_loss,'train_loss':train_loss}

                torch.save(checkpoint, 'saved_models/check_point_'+exp_name+'.pth')
                model.train(True)
                print(phase+' Loss - '+str(train_loss[epoch]))

            if phase=='train':
                plt.plot([i for i in range(epoch+1)],train_loss[:epoch+1])        
            elif phase=='val':
                plt.plot([i for i in range(epoch+1)],val_loss[:epoch+1])
            plt.savefig('plots/'+exp_name+'loss.png'),plt.close()
        sch.step()
    
    if phase=='train':
        plt.plot([i for i in range(epoch+1)][1:],train_loss[:epoch+1][1:])        

    plt.savefig('plots/'+exp_name+'loss_log_plot.png'),plt.close()
    
def test(loader_dic, model, save_dir='save_res/',slice_length=501):
    
    res_dic = {'name':[]}
    if save_dir is not None and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
#     criterion = MaskedMSELoss()
    criterion = torch.nn.MSELoss()
    model.eval()
    
    res_dic = {'NAME':[],'NMAE':[]}
    
    for loader_name in loader_dic:
        loader = loader_dic[loader_name]
        out_vol = np.zeros((slice_length,slice_length,slice_length))
        img_vol = np.zeros((slice_length,slice_length,slice_length))
        gt_vol = np.zeros((slice_length,slice_length,slice_length))

        c=0
#         print(loader_name)
        
        for img,gt,name in loader:
            
            batch_size = loader.batch_size


            img = img.cuda()
#                 gt = gt.cuda()
            out = model(img)
            out_vol[c:min(c+batch_size,slice_length),:,:] = out.detach().cpu().numpy()[:,0,...]
            del(out)
#                 del(gt)
            img = img.detach().cpu()
#             print(img.size())
            img_vol[c:min(c+batch_size,slice_length),:,:] = img[:,loader.dataset.window_size//2,:,:].detach().cpu().numpy()
            gt_vol[c:min(c+batch_size,slice_length),:,:] = gt.squeeze().detach().cpu().numpy()
            c+=batch_size
            del(img)
        
        
        np.save(save_dir+'/'+'img_vol'+loader_name,img_vol) # EP reconstruction
        np.save(save_dir+'/'+'gt_vol'+loader_name,gt_vol+img_vol) # ground truth 
        np.save(save_dir+'/'+'out_vol'+loader_name,out_vol+img_vol) # 2.5D CNN output

