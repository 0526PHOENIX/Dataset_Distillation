"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

from Utils.PerceptualLoss import PerceptualLoss

from Model import *


"""
========================================================================================================================
Loss Function
========================================================================================================================
"""
class Loss():

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        
        # Device: CPU or GPU
        self.device = device

        # Perceptual Loss
        self.get_per = PerceptualLoss(device = self.device)

        return

    """
    ====================================================================================================================
    Get Pixelwise Loss: MSE Loss [0, 2]
    ====================================================================================================================
    """
    def get_pix_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:
            
        # MSE
        mse = torch.square(fake_g - real_g).sum() / fake_g.numel()

        return mse.mean()

    """
    ====================================================================================================================
    Get Gradient Difference Loss [0, 2]
    ====================================================================================================================
    """
    def get_gdl_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # First Derivative of Predicts
        grad_fake_x = torch.abs(fake_g[:, :, 1:, :] - fake_g[:, :, :-1, :])
        grad_fake_y = torch.abs(fake_g[:, :, :, 1:] - fake_g[:, :, :, :-1])

        # First Derivative of Labels
        grad_real_x = torch.abs(real_g[:, :, 1:, :] - real_g[:, :, :-1, :])
        grad_real_y = torch.abs(real_g[:, :, :, 1:] - real_g[:, :, :, :-1])

        # MAE
        gdl_x = torch.abs(grad_fake_x - grad_real_x).sum() / grad_fake_x.numel()
        gdl_y = torch.abs(grad_fake_y - grad_real_y).sum() / grad_fake_y.numel()

        return (gdl_x.mean() + gdl_y.mean()) / 2
    
    """
    ====================================================================================================================
    Get Similarity Loss [0, 2]
    ====================================================================================================================
    """
    def get_sim_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # SSIM
        ssim = StructuralSimilarityIndexMeasure(kernel_size = 5).to(self.device)(fake_g, real_g)

        return 1 - ssim.mean()

    """
    ====================================================================================================================
    Get Perceptual Loss [0, 2]
    ====================================================================================================================
    """
    def get_per_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # Extent to RGB Form
        fake_g = fake_g.repeat(1, 3, 1, 1)
        real_g = real_g.repeat(1, 3, 1, 1)

        # Perceptual Loss
        perceptual = self.get_per(fake_g, real_g)

        return perceptual.mean()
    

"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    model = Iso_Dilate_Shuffle(8)

    real1 = torch.randn((1, 7, 256, 256), requires_grad = True)
    real2 = torch.randn((1, 1, 256, 256), requires_grad = True)

    fake2 = model(real1)

    loss = Loss().get_pix_loss(fake2, real2)
    loss.backward()

    print()
    print('real1 gradient', '\t', real1.grad.mean().item())
    print('real2 gradient', '\t', real2.grad.mean().item())
    print()