# supervised contrastive loss, implemented according to 
# https://arxiv.org/pdf/2004.11362.pdf

import torch
from torch import nn


class ConstrastiveLoss(nn.Module):
    def __init__(self, temperature=1.5):
        super(ConstrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_tensor, neg_tensor):

        self.pos_tensor = torch.cat(pos_tensor, dim=0)
        self.neg_tensor = torch.cat(neg_tensor, dim=0)
        if self.pos_tensor.is_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'

        # normalize, because i have trust issues
        # ^ actually, normalizing turned out to be really important, it should be done here and not in the pipeline
        self.pos_tensor = (self.pos_tensor - torch.mean(self.pos_tensor, dim=1).unsqueeze(dim=1)) / torch.std(self.pos_tensor, dim=1).unsqueeze(dim=1)
        self.neg_tensor = (self.neg_tensor - torch.mean(self.neg_tensor, dim=1).unsqueeze(dim=1)) / torch.std(self.neg_tensor, dim=1).unsqueeze(dim=1)
        # print(pos_tensor)

        # this computes the inner product of Zi and Zk where i != k., and i is the anchor, and k are negative samples, usually augments of the anchor 

        # we also divide it by the temperature constant they mention in the paper, since that is convenient to do here, and we exponentiate 
        E_Zi_Zk = torch.exp(torch.div(
            torch.matmul(self.pos_tensor, self.neg_tensor.T),
            self.temperature))
        # this computes the inner product of Zi and Zj where i != j, and i is the anchor, and j are other positive samples
        # the diagonal of the output matrix will be the inner product of i and i, so we remove it by making a mask with torch.eye
        # we also exponentiate and sum. 
        eye = torch.eye(len(self.pos_tensor), dtype = torch.bool).to(device)

        Sum_E_Zi_Zj = torch.sum(
            torch.exp(torch.div(
                torch.matmul(self.pos_tensor, self.pos_tensor.T)*~eye,
                self.temperature
            )), dim=1).unsqueeze(dim=1) # row-wise sum, unsqueeze so we divide each row (as opposed to each columnn) in E_Zi_Zk by each element in this list

        loss_magnitude = torch.sum(torch.log(torch.div(E_Zi_Zk, Sum_E_Zi_Zj)))
        loss_coefficient = -1 / (2*(len(pos_tensor)-1) - 1)
        
        batch_loss = loss_coefficient*loss_magnitude
        print('Batch Loss: ', batch_loss.item())
        return batch_loss



