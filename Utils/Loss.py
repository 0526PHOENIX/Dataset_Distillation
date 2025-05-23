"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torchmetrics.image import StructuralSimilarityIndexMeasure

from Utils.PerceptualLoss import PerceptualLoss


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
    Get Pixelwise Loss: MAE Loss [0, 2]
    ====================================================================================================================
    """
    def get_pix_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:
            
        # MAE
        mae = torch.abs(fake_g - real_g).sum() / fake_g.numel()

        return mae.mean()

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

        # SSIM and SSIM Map
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

        perceptual = self.get_per(fake_g, real_g)

        return perceptual.mean()
    
    """
    ====================================================================================================================
    Get Adversarial Loss: RMSE Loss [0, 2]
    ====================================================================================================================
    """
    def get_adv_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # RMSE 
        rmse = torch.sqrt(torch.square(real_g - fake_g).sum() / fake_g.numel())

        return rmse.mean()

    """
    ====================================================================================================================
    Get Cycle Consistency Loss [0, 2]
    ====================================================================================================================
    """
    def get_cyc_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # MAE
        mae = torch.abs(fake_g - real_g).sum() / fake_g.numel()

        return mae.mean()
    
    """
    ====================================================================================================================
    Get Loss Based on Deep Supervision
    ====================================================================================================================
    """
    def get_deep_supervision(self, 
                             mode: str | Literal['pix', 'gdl', 'sim'],
                             fake_g: list[Tensor],
                             real_g: Tensor) -> Tensor:

        # Loss Function Mode
        if mode == 'pix':
            loss_fn = self.get_pix_loss
        elif mode == 'gdl':
            loss_fn = self.get_gdl_loss
        elif mode == 'sim':
            loss_fn = self.get_sim_loss
        else:
            raise ValueError('Invalid Loss Function Mode')

        # Total Loss
        loss = torch.tensor(0.0, requires_grad = True).to(self.device)

        # Total Pixel
        pixels = 0

        # Compute Loss Through Different Scale
        for fake in fake_g:

            # Pixel
            pixel = fake.shape[2] * fake.shape[3]

            # Total Pixel
            pixels += pixel

            # Interpolation
            fake = interpolate(fake, size = real_g.shape[2:], mode = "bilinear")

            # Loss
            loss = loss + (loss_fn(fake, real_g) * pixel)

        # Weighted Mean
        return loss.sum() / pixels