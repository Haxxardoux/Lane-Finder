# supervised contrastive loss, implemented according to 
# https://arxiv.org/pdf/2004.11362.pdf

import torch
from torch import nn
from utils import get_device_from_model

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=3):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_tensor, neg_tensor):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if pos_tensor.is_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'


        # normalize, because i have trust issues
        pos_tensor = (pos_tensor - torch.mean(pos_tensor, dim=1).unsqueeze(dim=1))# / torch.std(pos_tensor, dim=1).unsqueeze(dim=1)
        neg_tensor = (neg_tensor - torch.mean(neg_tensor, dim=1).unsqueeze(dim=1))# / torch.std(neg_tensor, dim=1).unsqueeze(dim=1)

        # this computes the inner product of Zi and Zk where i != k., and i is the anchor, and k are negative samples, usually augments of the anchor 
        # this is the second time i have used matrix multiplication in code and it feels good as FUCK. nuttttt

        # we also divide it by the temperature constant they mention in the paper, since that is convenient to do here, and we exponentiate 
        E_Zi_Zk = torch.exp(torch.div(
            torch.matmul(pos_tensor, neg_tensor.T),
            self.temperature))

        # this computes the inner product of Zi and Zj where i != j, and i is the anchor, and j are other positive samples
        # the diagonal of the output matrix will be the inner product of i and i, so we remove it by making a mask with torch.eye
        # we also exponentiate 
        eye = torch.eye(len(pos_tensor), dtype = torch.bool).to(device)

        Sum_E_Zi_Zj = torch.sum(
            torch.exp(torch.div(
                torch.matmul(pos_tensor, pos_tensor.T)*~eye,
                self.temperature
            )), dim=1).unsqueeze(dim=1) # row-wise sum, unsqueeze so we divide each row (as opposed to each columnn) in E_Zi_Zk by each element in this list

        loss_magnitude = torch.sum(torch.log(torch.div(E_Zi_Zk, Sum_E_Zi_Zj)))
        loss_coefficient = -1 / (2*(len(pos_tensor)-1) - 1)
        print(loss_magnitude)
        return loss_coefficient*loss_magnitude



