import torch
import matplotlib.pyplot as plt
    
class MaskedMSELoss(torch.nn.Module):
    def __init__(self,mask):
        super(MaskedMSELoss, self).__init__()
        self.mask=mask
        
    def forward(self, input, target,plot=False):
        mask=self.mask
        mse_loss = torch.nn.MSELoss(reduction='none')

        mse=mse_loss(input,target)
        if mse.shape[0]<mask.shape[0]:
            mask = mask[:mse.shape[0]]
#         print('mask = ',mask.shape)
#         print('mse = ',mse.shape)
        
        mse = (mse * mask.cuda())#.sum() # gives \sigma_euclidean over unmasked elements
        non_zero_elements = mask.sum() + 1e-5
        loss = mse.sum() / non_zero_elements
        if plot:
            plt.imshow(mse[0].squeeze().detach().cpu().numpy()),plt.savefig('loss_image.png'),plt.close()
            plt.imshow(input[0].squeeze().detach().cpu().numpy()),plt.savefig('input_image.png'),plt.close()
            plt.imshow(target[0].squeeze().detach().cpu().numpy()),plt.savefig('target_image.png'),plt.close()
        
        return loss
    