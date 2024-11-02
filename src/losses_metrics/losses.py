from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import MSELoss


class L2(Module):
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()

    def forward(self, x_recon: Tensor, x: Tensor, mean, log_var) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        recon_loss = self.loss(x_recon, x)
        return recon_loss, None, None


class VAE(Module):
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()

    def forward(self, x_recon: Tensor, x: Tensor, mean, log_var, beta=1.0) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss
