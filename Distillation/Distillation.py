"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import datetime
from typing import Literal
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
from torch import Tensor
from torch.optim import AdamW

from torch.nn import Parameter
from torch.nn.utils.stateless import functional_call

from Utils import *


"""
========================================================================================================================
Distillation
========================================================================================================================
"""
class Distillation():

    """
    ====================================================================================================================
    Critical Parameters
    ====================================================================================================================
    """
    def __init__(self,
                 lr: float                  = 1e-3,
                 batch: int                 = 16,
                 model: torch.nn.Module     = None,
                 device: torch.device       = torch.device('cpu'),
                 data: str                  = "",
                 result: str                = "",
                 *args,
                 **kwargs) -> None:
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print('\n' + 'Training on: ' + str(self.device))

        # Total Epoch & Batch Size
        self.batch = batch

        # Learning Rate
        self.lr = lr

        # Model
        self.model = model.to(self.device)
        print('\n' + 'Training Model: ' + type(self.model).__name__ + ' ' + str(self.model.width))

        # File Path
        self.data = data
        self.result = result

        # Loss and Metrics
        self.get_loss = Loss(device = self.device)
        self.get_metrics = Metrics(device = self.device)

        # Model, Optimizer, Data Loader
        self.initialization()

        return
    
    """
    ====================================================================================================================
    Optimizer, Data Loader
    ====================================================================================================================
    """
    def initialization(self) -> None:

        # Training Data Loader and Sample Index
        self.val_dl = DL(root = self.data, mode = 'Train', device = self.device, batch_size = self.batch)

        return
    
    """
    ====================================================================================================================
    Main Distillation Function
    ====================================================================================================================
    """
    def main(self) -> None:

        syn_real1 = Parameter(torch.randn(4, 7, 256, 256, requires_grad = True).to(self.device))  # 合成 MR
        syn_real2 = Parameter(torch.randn(4, 1, 256, 256, requires_grad = True).to(self.device))  # 合成 CT

        # Optimizer: AdamW
        out_opt = AdamW([syn_real1, syn_real2], lr = self.lr, weight_decay = 0.05)

        for k in range(10):
            
            # Model
            model = self.model

            # Model Paramenters
            theta = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}

            # Inner loop
            for _ in range(3):

                for i in range(0, 4, self.batch):

                    syn_real1_g = syn_real1[i : i + self.batch]
                    syn_real2_g = syn_real2[i : i + self.batch]

                    syn_fake2_g = functional_call(model, theta, (syn_real1_g,))

                    # Total Loss
                    loss = torch.tensor(0.0, requires_grad = True).to(self.device)

                    # Pixelwise Loss
                    loss = loss + self.get_loss.get_pix_loss(syn_fake2_g, syn_real2_g)

                    grads = torch.autograd.grad(loss, theta.values(), create_graph = True)
                    theta = {name: param - self.lr * grad for (name, param), grad in zip(theta.items(), grads)}

            val_loss = torch.tensor(0.0, requires_grad = True).to(self.device)
            for batch_tuple in self.val_dl:

                real1_g, real2_g, _, _ = batch_tuple

                # Outer loop: 評估在驗證集上表現，反向更新 d_MR, d_CT
                fake2_g = functional_call(model, theta, (real1_g,))

                val_loss = val_loss + self.get_loss.get_pix_loss(fake2_g, real2_g)

                break

            # 對 d_MR, d_CT 求梯度
            out_opt.zero_grad()
            val_loss.backward()
            out_opt.step()

            self.save_images(syn_real1_g, syn_real2_g, postfix = str(k))

        return

    """
    ====================================================================================================================
    Save Image
    ====================================================================================================================
    """ 
    def save_images(self, syn_real1_g: Tensor, syn_real2_g: Tensor, postfix: str) -> None:

        syn_real1_a = syn_real1_g.detach().cpu().numpy()
        syn_real2_a = syn_real2_g.detach().cpu().numpy()

        # Create the plot
        _, axs = plt.subplots(2, 2, figsize = (15, 15))

        # Remove Redundancy
        for ax in axs.flat:
            ax.axis('off')

        # Display Input MR Image
        ax = axs[0][0]
        ax.imshow(syn_real1_a[0, 3], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled MR 1')

        # Display Output CT Image
        ax = axs[0][1]
        ax.imshow(syn_real2_a[0, 0], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled CT 1')

        # Display Input MR Image
        ax = axs[1][0]
        ax.imshow(syn_real1_a[1, 3], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled MR 2')

        # Display Output CT Image
        ax = axs[1][1]
        ax.imshow(syn_real2_a[1, 0], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled CT 2')

        # Save Figure
        plt.tight_layout()
        plt.savefig(postfix + '.png', format = 'png', dpi = 300)
        plt.close()