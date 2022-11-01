import matplotlib.pyplot as plt


def dbgplt(lv_data,out_data,out_data2,axs):

# PLOT FOR DEBUGGING
        # plt.figure(1)
        # slc_num = 200#18
        colmap = 'hot'
        
        # data_true1 = data_true.transpose(2,1,0)
        ax = axs[0]
        im = ax.imshow(lv_data.cpu().detach().numpy()[0,0,4,:,:],
                       cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
        ax.set_title('True');ax.axis('off')
        # fig.colorbar(im,ax = ax,shrink = 0.25)
        
        ax = axs[1]
        im = ax.imshow(out_data.cpu().detach().numpy()[0,4,:,:],
                       cmap=colmap, interpolation='none',vmax=50000*1e-3,vmin=0)
        ax.set_title('intermediate');ax.axis('off')
        # fig.colorbar(im,ax = ax,shrink = 0.25)
        
        
        ax = axs[2]
        im = ax.imshow(out_data2.cpu().detach().numpy()[0,0,:,:],
                       cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
        ax.set_title('out');ax.axis('off')
        # fig.colorbar(im,ax = ax,shrink = 0.25)
    
        plt.show()    
        plt.pause(3)